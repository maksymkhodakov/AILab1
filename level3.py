"""
РІВЕНЬ 3 (Складний): Трансформери (ViT/DeiT) для Fashion-MNIST (10 класів)

Що робить цей скрипт:
  1) Завантажує Fashion-MNIST як PIL-зображення (torchvision кешує локально → повторно не качає).
  2) Перетворює grayscale → RGB, масштабує до 224×224, нормалізує під обрану модель (через AutoImageProcessor).
  3) Підтримує два режими:
     - **Повний fine-tune** (базовий варіант для високої якості).
     - **Linear probe** (швидкий варіант: заморожено бекбон, навчається лише класифікаційна голова).
  4) Додає train-аугментації (RandomResizedCrop/Flip/Affine) — це суттєво покращує узагальнення трансформерів.
  5) Розбиває train на train/val (eval під час навчання — за val; test використовується лише вкінці).
  6) Логує деталі (девайс, кількість trainable-параметрів, LR тощо).
  7) Зберігає у вихідну теку:
     - `classification_report.txt` (повний звіт по класах + загальна Accuracy),
     - `confusion_matrix.png` (матриця плутанини),
     - `training_curves.png` (криві втрат та eval-accuracy),
     - `reliability.png` (калібрування, ECE),
     - `confidence_hist.png` (гістограма впевненості),
     - `top_confusions_grid.png` (грід найтиповішої плутанини).
"""

import argparse
import os

# Вимикаємо експериментальний швидкий даунлоадер HF (інколи неконтрольовані варнінги/ретраї)
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
# Трохи тихіше логи всередині HF/Tokenizers
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import time
import random
import numpy as np
import torch
import torch.nn as nn

# Використаємо неблокуючий бекенд для matplotlib → усі графіки йдуть у файли .png
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import Subset, random_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, brier_score_loss

from transformers import (
    AutoImageProcessor,
    ViTForImageClassification,
    TrainingArguments,
    Trainer,
    default_data_collator,
    EarlyStoppingCallback,
    get_linear_schedule_with_warmup
)

# ----------------------------- КОНСТАНТИ (імена класів) -----------------------------
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


# ----------------------------- ДЕТЕРМІНОВАНІСТЬ -----------------------------
def set_seed(seed: int = 42):
    """Фіксуємо зерна генераторів випадкових чисел для максимальної відтворюваності."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------- ДОДАТКОВА УТИЛІТА -----------------------------
# Топ-рівнева функція, яку можна піклити (на відміну від lambda)
def ensure_rgb(pil_img: Image.Image) -> Image.Image:
    """Гарантує, що зображення в RGB (ViT/DeiT очікують 3 канали)."""
    return pil_img.convert("RGB") if pil_img.mode != "RGB" else pil_img


# ----------------------------- ВИБІР ПРИСТРОЮ -----------------------------
def select_device():
    """
    Обираємо найкращий доступний бекенд:
      - CUDA (NVIDIA GPU) → найшвидший варіант;
      - MPS (Apple Silicon) → значне прискорення проти CPU;
      - CPU → якщо GPU немає.
    """
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps"), "mps"
    return torch.device("cpu"), "cpu"


# ----------------------------- ДОПОМОЖНІ ФУНКЦІЇ -----------------------------
def save_text(text: str, path: str):
    """Безпечно зберігаємо текст у файл (створюємо директорію, якщо немає)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def count_parameters(model):
    """
    Підрахунок параметрів моделі:
      - total: усі параметри,
      - trainable: ті, що оновлюються (requires_grad=True),
      - ratio: частка trainable від total (важливо для linear probe).
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = (trainable / total) if total else 0.0
    return total, trainable, ratio


# ----------------------------- ТРАНСФОРМИ ДЛЯ TRAIN/EVAL -----------------------------
class HFTransformTrain:
    """
    TRAIN-трансформації під трансформер + нормалізація:
      - Конвертуємо grayscale → RGB (ViT/DeiT очікують 3 канали).
      - RandomResizedCrop/Flip/Affine — помірні аугментації (покращують узагальнення).
      - Подаємо картинку у AutoImageProcessor, щоб застосувати mean/std із чекпойнта.
    """

    def __init__(self, processor):
        self.processor = processor
        # Витягуємо цільовий розмір із конфігурації препроцесора (зазвичай 224×224).
        size = processor.size
        if isinstance(size, dict):
            self.H = size.get("height", size.get("shortest_edge", 224))
            self.W = size.get("width", size.get("shortest_edge", 224))
        else:
            self.H = self.W = size or 224

        # Набір аугментацій (обережних — без сильних деформацій, щоб не ламати структури одягу).
        self.aug = transforms.Compose([
            transforms.Lambda(ensure_rgb),
            transforms.RandomResizedCrop((self.H, self.W), scale=(0.7, 1.0), ratio=(0.9, 1.1), antialias=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),  # невеликі оберти/зсуви
        ])

    def __call__(self, pil_img: Image.Image) -> torch.Tensor:
        img = self.aug(pil_img)
        # processor повертає dict з тензором 'pixel_values' у нормалізованому просторі моделі
        enc = self.processor(images=[img], return_tensors="pt")
        return enc["pixel_values"].squeeze(0)  # (3, H, W)


class HFTransformEval:
    """
    EVAL/TEST-трансформації (без аугментацій):
      - Лише конвертація в RGB, масштабування до 224×224 та нормалізація під модель.
    """

    def __init__(self, processor):
        self.processor = processor
        size = processor.size
        if isinstance(size, dict):
            self.H = size.get("height", size.get("shortest_edge", 224))
            self.W = size.get("width", size.get("shortest_edge", 224))
        else:
            self.H = self.W = size or 224
        self.resize = transforms.Resize((self.H, self.W), antialias=True)

    def __call__(self, pil_img: Image.Image) -> torch.Tensor:
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        pil_img = self.resize(pil_img)
        enc = self.processor(images=[pil_img], return_tensors="pt")
        return enc["pixel_values"].squeeze(0)


class TV2HF(torch.utils.data.Dataset):
    """
    Обгортка над torchvision-датасетом:
      - на вході: елемент (PIL, label),
      - на виході: dict {"pixel_values": Tensor(3,H,W), "labels": int}
    Це потрібно, щоб Trainer міг напряму зʼєднатися із датасетом.
    """

    def __init__(self, tv_ds, tf):
        self.ds = tv_ds
        self.tf = tf

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        img, y = self.ds[i]
        return {"pixel_values": self.tf(img), "labels": int(y)}


# ----------------------------- ВІЗУАЛІЗАЦІЇ -----------------------------
def plot_training_curves(log_history, out_path):
    """
    Будуємо дві криві:
      - train loss по кроках (Trainer логуватиме кожні logging_steps),
      - eval loss / eval accuracy по епохах.
    Результат зберігаємо у PNG; показ не блокує виконання.
    """
    steps, tloss, eloss, eacc = [], [], [], []
    for rec in log_history:
        if "loss" in rec and "epoch" in rec:
            steps.append(rec.get("step", len(steps)))
            tloss.append(rec["loss"])
        if "eval_loss" in rec:
            eloss.append(rec["eval_loss"])
            eacc.append(rec.get("eval_accuracy", None))

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    if tloss:
        ax[0].plot(tloss, label="train loss")
    if eloss:
        # Масштабуємо eval_points по довжині train loss для візуальної привʼязки
        ax[0].plot(np.linspace(0, max(len(tloss) - 1, 1), len(eloss)), eloss, label="eval loss")
    ax[0].set_title("Loss")
    ax[0].legend()

    if eacc and any(a is not None for a in eacc):
        ax[1].plot([a for a in eacc if a is not None], marker="o", label="eval acc")
    ax[1].set_title("Eval Accuracy")
    ax[1].legend()

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(cm, classes, out_path, title="Матриця плутанини — ViT"):
    """Світла матриця плутанини з підписами у клітинках."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(classes)), yticks=np.arange(len(classes)),
        xticklabels=classes, yticklabels=classes,
        title=title, ylabel="Справжня", xlabel="Передбачена"
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:d}", ha="center", va="center", fontsize=8)
    fig.savefig(out_path, dpi=150, bbox_inches="tight");
    plt.close(fig)


def reliability_bins(y_true, proba, n_bins=10):
    """
    Розрахунок бінів калібрування (надійності):
      - розбиваємо впевненість (max softmax) на n_bins,
      - у кожному біні рахуємо середню точність та середню впевненість,
      - повертаємо також ECE (Expected Calibration Error) та Brier-score.
    """
    y_pred = np.argmax(proba, axis=1)
    conf = np.max(proba, axis=1)
    correct = (y_pred == y_true).astype(float)

    bins = np.linspace(0, 1, n_bins + 1)
    mids, accs, confs, weights = [], [], [], []
    ece = 0.0
    N = len(y_true)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf >= lo) & (conf < hi if i < n_bins - 1 else conf <= hi)
        if mask.sum() == 0:
            continue
        mids.append((lo + hi) / 2)
        accs.append(correct[mask].mean())
        confs.append(conf[mask].mean())
        w = mask.sum() / N
        weights.append(w)
        ece += w * abs(accs[-1] - confs[-1])

    # Brier для мультикласу: середній по one-vs-rest
    K = proba.shape[1]
    Y = np.eye(K)[y_true]
    brier = np.mean([brier_score_loss(Y[:, k], proba[:, k]) for k in range(K)])
    return np.array(mids), np.array(accs), np.array(confs), np.array(weights), float(ece), float(brier)


def plot_reliability(mids, accs, confs, weights, out_path):
    """Графік калібрування (діаграма надійності) + підписи часток спостережень у біні."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="ідеальне калібрування")
    ax.plot(confs, accs, marker="o", label="модель")
    for x, y, w in zip(confs, accs, weights):
        ax.text(x, y, f"{w * 100:.0f}%", fontsize=8, ha="center", va="bottom")
    ax.set_xlabel("Середня впевненість")
    ax.set_ylabel("Точність у біні")
    ax.set_title("Калібрування прогнозів (top-1)")
    ax.legend()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confidence_hist(proba, out_path):
    """Гістограма розподілу впевненості (max softmax по прикладах)."""
    conf = np.max(proba, axis=1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(conf, bins=20)
    ax.set_title("Гістограма впевненості (top-1)")
    ax.set_xlabel("Впевненість")
    ax.set_ylabel("Кількість прикладів")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_top_confusions_grid_from_dataset(ds, y_true, y_pred, classes, out_path, kmax=16):
    """
    Грід найбільш характерної плутанини. Щоб не зʼїсти всю памʼять:
      - НЕ стекуємо весь тестовий набір (10k × 3×224×224).
      - Обчислюємо матрицю плутанини, знаходимо пару класів (i→j) з найбільшим відносним off-diagonal.
      - Витягуємо ЛИШЕ kmax відповідних картинок по індексах із ds, будуємо невеликий грід.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    i_best, j_best, best = 0, 0, -1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i == j:
                continue
            if cm_norm[i, j] > best:
                best = cm_norm[i, j]
                i_best, j_best = i, j

    # Індекси прикладів з найсильнішою плутаниною i_best → j_best
    idx = np.where((y_true == i_best) & (y_pred == j_best))[0][:kmax]
    if len(idx) == 0:
        return  # немає відповідних прикладів — нічого малювати

    cols = 4
    rows = int(np.ceil(len(idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.8 * cols, 1.8 * rows), constrained_layout=True)

    # Малюємо кожне зображення по індексу (витягуємо з ds по одному — економно для памʼяті)
    axes = np.atleast_2d(axes)
    for ax, ii in zip(axes.flat, idx):
        sample = ds[ii]  # dict: {"pixel_values": Tensor(3,H,W), "labels": int}
        img = sample["pixel_values"].detach().cpu()  # (3,H,W), уже нормалізоване під модель
        img = img.permute(1, 2, 0).numpy()  # (H,W,3) для matplotlib
        # Робимо min-max нормалізацію для візуалізації (бо зображення вже нормалізовані mean/std):
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        ax.imshow(img)
        ax.set_axis_off()

    # Якщо грід більший, ніж кількість прикладів — вимикаємо зайві осі
    for ax in axes.flat[len(idx):]:
        ax.set_axis_off()

    fig.suptitle(f"Помилки: {classes[i_best]} → {classes[j_best]}")
    fig.savefig(out_path, dpi=150, bbox_inches="tight");
    plt.close(fig)


# ----------------------------- ОСНОВНА ФУНКЦІЯ -----------------------------
def main():
    # --- 0) ПАРСИНГ АРГУМЕНТІВ КОМАНДНОГО РЯДКА ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="./data", help="Де кешувати torchvision Fashion-MNIST")
    parser.add_argument("--model-id", type=str, default="facebook/deit-tiny-patch16-224",
                        help="HF-модель, напр.: 'facebook/deit-tiny-patch16-224' або 'google/vit-base-patch16-224-in21k'")
    parser.add_argument("--output-dir", type=str, default="./plots/level3",
                        help="Куди зберігати чекпойнти/графіки/звіти")

    # Гіперпараметри та режими
    parser.add_argument("--epochs", type=int, default=10, help="Кількість епох навчання")
    parser.add_argument("--batch-size", type=int, default=64, help="Розмір батчу на пристрій")
    parser.add_argument("--lr", type=float, default=5e-4, help="Базовий LR для бекбону (якщо він НЕ заморожений)")
    parser.add_argument("--lr-head", type=float, default=5e-3, help="Вищий LR для класифікаційної голови")
    parser.add_argument("--weight-decay", type=float, default=0.05, help="L2-регуляризація (AdamW)")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Частка warmup-кроків для планувальника LR")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing у Trainer")
    parser.add_argument("--seed", type=int, default=42, help="Seed для відтворюваності")

    parser.add_argument("--freeze-base", action="store_true",
                        help="Linear probe: заморозити бекбон (тренується лише класифікаційна голова)")

    # Продуктивність/логістика
    parser.add_argument("--plots", type=str, default="save", choices=["save", "none", "show"],
                        help="save=писати PNG у вихідну теку; none=не будувати графіки; show=інтерактивно (може блокувати)")
    parser.add_argument("--num-workers", type=int, default=4, help="Потоки DataLoader (підготовка батчів)")
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="Опційно обрізати train для швидких прогонів")
    parser.add_argument("--max-eval-samples", type=int, default=None, help="Опційно обрізати test (лише для швидкості)")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Частка train, виділена під validation")
    parser.add_argument("--early-stop", type=int, default=2, help="Early stopping patience (0=вимкнено)")
    args = parser.parse_args()

    # --- 1) ВСТАНОВЛЮЄМО SEED + ВИЗНАЧАЄМО ПРИСТРІЙ ---
    set_seed(args.seed)
    device, dev_name = select_device()
    # pin_memory має сенс на CUDA; на MPS воно ігнорується (PyTorch просто попереджає)
    pin_mem = (dev_name == "cuda")

    # --- 2) ЗАВАНТАЖЕННЯ ДАТАСЕТІВ (torchvision кешує у data-dir → повторно не качає) ---
    full_train = datasets.FashionMNIST(root=args.data_dir, train=True, download=True, transform=None)
    test_tv = datasets.FashionMNIST(root=args.data_dir, train=False, download=True, transform=None)

    # Опційно обрізаємо для швидких експериментів (лише кількість елементів, порядок той самий)
    if args.max_train_samples is not None and args.max_train_samples > 0:
        full_train = Subset(full_train, range(min(args.max_train_samples, len(full_train))))
    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        test_tv = Subset(test_tv, range(min(args.max_eval_samples, len(test_tv))))

    # --- 3) ВАЛІДАЦІЙНИЙ СПЛІТ З TRAIN (eval під час навчання буде саме на val) ---
    n_total = len(full_train)
    n_val = int(n_total * args.val_fraction)
    n_tr = n_total - n_val
    train_tv, val_tv = random_split(full_train, [n_tr, n_val],
                                    generator=torch.Generator().manual_seed(args.seed))

    # --- 4) СТВОРЮЄМО ПРОЦЕСОР HF І ТРАНСФОРМИ ДЛЯ TRAIN/EVAL ---
    processor = AutoImageProcessor.from_pretrained(args.model_id, use_fast=True)
    tf_train = HFTransformTrain(processor)  # аугментації + нормалізація
    tf_eval = HFTransformEval(processor)  # тільки resize + нормалізація

    # Обгортки, які перетворюють (PIL, label) → {"pixel_values": Tensor, "labels": int}
    train_ds = TV2HF(train_tv, tf_train)
    val_ds = TV2HF(val_tv, tf_eval)
    test_ds = TV2HF(test_tv, tf_eval)

    # --- 5) МОДЕЛЬ (10 класів) + опційна заморозка бекбону ---
    id2label = {i: n for i, n in enumerate(CLASS_NAMES)}
    label2id = {n: i for i, n in enumerate(CLASS_NAMES)}
    model = ViTForImageClassification.from_pretrained(
        args.model_id,
        num_labels=10,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True  # важливо: у ImageNet-моделей 1000 класів → автоматично ре-ініт голови під 10
    )

    if args.freeze_base:
        # Linear probe: тренуємо ТІЛЬКИ класифікаційну голову; бекбон фіксований
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith("classifier.")
        # (не обовʼязково) акуратна ре-ініціалізація ваг голови
        if isinstance(model.classifier, nn.Linear):
            nn.init.xavier_uniform_(model.classifier.weight)
            if model.classifier.bias is not None:
                nn.init.zeros_(model.classifier.bias)

    # Лог: скільки параметрів справді навчається
    total_p, train_p, ratio = count_parameters(model)

    # --- 6) НАЛАШТУВАННЯ TRAINER (TrainingArguments) ---
    common_args = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,  # базовий LR (для бекбону)
        weight_decay=args.weight_decay,
        logging_strategy="steps",
        logging_steps=20,
        disable_tqdm=False,
        save_strategy="epoch",
        evaluation_strategy="epoch",  # оцінюємо на val в кінці кожної епохи
        load_best_model_at_end=True,  # повертаємо найкращу версію після тренування
        metric_for_best_model="accuracy",
        greater_is_better=True,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=pin_mem,  # тільки CUDA дає виграш
        label_smoothing_factor=args.label_smoothing,
        report_to=[]  # без зовнішніх трекерів (W&B тощо)
    )

    try:
        training_args = TrainingArguments(**common_args)
    except TypeError:
        # Підтримка старих версій transformers (eval_strategy замість evaluation_strategy)
        common_args.pop("evaluation_strategy", None)
        common_args["eval_strategy"] = "epoch"
        training_args = TrainingArguments(**common_args)

    # --- 7) ОКРЕМИЙ LR ДЛЯ ГОЛОВИ КЛАСИФІКАЦІЇ ---
    #   Робимо 2 групи параметрів: head (вищий LR) і base (нижчий LR).
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("classifier.")]
    base_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("classifier.")]

    from torch.optim import AdamW
    optimizer = AdamW([
        {"params": base_params, "lr": args.lr},
        {"params": head_params, "lr": args.lr_head}
    ], weight_decay=args.weight_decay)

    # Планувальник навчання: лінійний warmup → лінійний decay (стабілізує початок навчання)
    num_update_steps_per_epoch = int(np.ceil(len(train_ds) / args.batch_size))
    max_train_steps = num_update_steps_per_epoch * args.epochs
    num_warmup_steps = int(max_train_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, max_train_steps)

    # --- 8) МЕТРИКА ДЛЯ MODEL SELECTION (Trainer бере її з compute_metrics) ---
    def compute_metrics(eval_pred):
        """
        Trainer може передавати або EvalPrediction, або tuple(predictions, labels) — підтримуємо обидва.
        Повертаємо dict з 'accuracy', щоб Trainer міг обрати найкращу епоху.
        """
        logits = getattr(eval_pred, "predictions", eval_pred[0])
        labels = getattr(eval_pred, "label_ids", eval_pred[1])
        preds = np.argmax(logits, axis=1)
        return {"accuracy": float((preds == labels).mean())}

    # Early stopping (не обовʼязково, але корисно на валідaції)
    callbacks = []
    if args.early_stop and args.early_stop > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stop))

    # --- 9) СТВОРЮЄМО TRAINER ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,  # ВАЖЛИВО: eval під час навчання — на валідації, не на тесті
        tokenizer=processor,  # не обовʼязково, але ок
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        optimizers=(optimizer, scheduler)  # наш оптимізатор з 2 LR + планувальник
    )

    # --- 10) ІНФОРМАТИВНІ ЛОГИ ПЕРЕД СТАРТОМ ---
    print("\n=== НАЛАШТУВАННЯ (ViT) ===")
    print(f"Модель:           {args.model_id}")
    print(f"Пристрій:         {dev_name}")
    print(f"Train/Val/Test:   {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")
    print(f"Batch/Epochs:     {args.batch_size}/{args.epochs}")
    print(f"LR base/head:     {args.lr} / {args.lr_head}  | warmup_ratio={args.warmup_ratio}")
    print(f"Reg/LS:           weight_decay={args.weight_decay}, label_smoothing={args.label_smoothing}")
    print(f"Параметри:        total={total_p:,} | trainable={train_p:,} ({ratio * 100:.1f}%)")
    print(f"Режим:            {'Linear probe (freeze base)' if args.freeze_base else 'Full fine-tune'}")

    # --- 11) НАВЧАННЯ ---
    t0 = time.perf_counter()
    trainer.train()
    t_train = time.perf_counter() - t0
    print(f"[TIME] Навчання завершено за: {t_train:.2f} с")

    # --- 12) КРИВІ НАВЧАННЯ (LOSS/ACCURACY) ---
    if args.plots != "none":
        os.makedirs(args.output_dir, exist_ok=True)
        plot_training_curves(trainer.state.log_history, os.path.join(args.output_dir, "training_curves.png"))

    # --- 13) ФІНАЛЬНА ОЦІНКА НА ТЕСТІ ---
    preds = trainer.predict(test_ds)  # ВАЖЛИВО: test не використовувався під час тренування
    logits = preds.predictions
    y_true = preds.label_ids
    y_pred = np.argmax(logits, axis=1)

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, digits=4, zero_division=0, target_names=CLASS_NAMES
    )
    cm = confusion_matrix(y_true, y_pred)

    print("\n=== ПІДСУМКИ (TEST) ===")
    print(f"Accuracy: {acc:.4f}")
    print(report)

    # Зберігаємо звіт
    os.makedirs(args.output_dir, exist_ok=True)
    save_text(report + f"\nAccuracy: {acc:.4f}\n", os.path.join(args.output_dir, "classification_report.txt"))

    # --- 14) ДОДАТКОВІ ВІЗУАЛІЗАЦІЇ ---
    if args.plots != "none":
        # 14.1 Матриця плутанини
        plot_confusion(cm, CLASS_NAMES, os.path.join(args.output_dir, "confusion_matrix.png"))

        # 14.2 Калібрування/гістограма впевненості (через softmax)
        proba = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        mids, accs, confs, weights, ece, brier = reliability_bins(y_true, proba, n_bins=10)
        plot_reliability(mids, accs, confs, weights, os.path.join(args.output_dir, "reliability.png"))
        plot_confidence_hist(proba, os.path.join(args.output_dir, "confidence_hist.png"))
        print(f"[CALIB] ECE={ece:.4f}, Brier={brier:.4f}")

        # 14.3 Грід найтиповішої плутанини (економно по памʼяті, завантажуємо лише потрібні приклади)
        plot_top_confusions_grid_from_dataset(
            test_ds, y_true, y_pred, CLASS_NAMES,
            os.path.join(args.output_dir, "top_confusions_grid.png"),
            kmax=16
        )

    # --- 15) ФІНАЛЬНИЙ ЛОГ ---
    print("\n[OK] Артефакти збережено у:", args.output_dir)
    print("[OK] Файли: classification_report.txt, confusion_matrix.png, training_curves.png, reliability.png, "
          "confidence_hist.png, top_confusions_grid.png")


# ЗАПУСК:
"""
python level3.py \
  --model-id facebook/deit-tiny-patch16-224 \
  --epochs 6 --batch-size 128 \
  --lr 5e-4 --lr-head 5e-3 --warmup-ratio 0.1 --label-smoothing 0.1 \
  --max-train-samples 30000 --val-fraction 0.1 --num-workers 4 \
  --plots save --output-dir ./plots/level3


Або швидший але менш точний варіант

python level3.py \
  --model-id facebook/deit-tiny-patch16-224 \
  --freeze-base --epochs 5 --batch-size 128 \
  --lr 5e-4 --lr-head 1e-2 --warmup-ratio 0.1 --label-smoothing 0.05 \
  --max-train-samples 30000 --val-fraction 0.1 --num-workers 4 \
  --plots save --output-dir ./plots/level3_probe_fast

"""

if __name__ == "__main__":
    main()
