"""
СЕРЕДНІЙ РІВЕНЬ — Fashion-MNIST (28x28, 10 класів)
Методи:
  1) SVM (лінійний / RBF) + (опційно) PCA для пришвидшення/стабілізації
  2) EM (Expectation-Maximization) через GaussianMixture (GMM) як байєсівський класифікатор
  3) Невелика CNN (2 згорткові блоки)

Дизайн:
  • Жодних блокувань: графіки зберігаємо у PNG (за замовчуванням --plots save).
  • Параметр --run-all запускає всі методи послідовно.
  • SVM/EM працюють на плоских ознаках (784), CNN — на тензорах (1×28×28).
  • Для SVM/EM: StandardScaler + (рекомендовано) PCA (типово 80 компонент).
  • Для CNN: криві навчання, матриця плутанини, надійність (ECE/Brier), грід помилок.

Запуск:
python level2.py --run-all --max-samples 20000 --pca 80 --epochs 5 --batch-size 128 --plots save --out-dir ./plots/level2
"""

import argparse
import os
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib

matplotlib.use("Agg")  # ← гарантія, що нічого не блокує (без GUI-бекенду)
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, brier_score_loss
from sklearn.svm import SVC, LinearSVC
from sklearn.mixture import GaussianMixture

# ----------------------------- НАЗВИ КЛАСІВ -----------------------------
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


# ----------------------------- ВІДТВОРЮВАНІСТЬ -----------------------------
def set_seed(seed: int = 42):
    """Фіксуємо seed для Python/NumPy/PyTorch — для відтворюваності."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # трішки детермінізму для CNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------- ДАНІ: TENSORS + FLAT -----------------------------
def load_fmnist_tensors(data_dir: str):
    """
    Завантаження Fashion-MNIST як тензорів (для CNN).
    Нормалізуємо до [-1,1], що допомагає швидше/стабільніше навчати CNN.
    """
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_ds = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=tfm)
    test_ds = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=tfm)
    return train_ds, test_ds


def tensor_to_numpy_flat(ds, max_samples=None):
    """
    Перетворення тензорного датасету у NumPy-масиви плоских ознак (784) для класичних методів (SVM/GMM).
    Вхідні тензори мають нормалізацію [-1,1], тому повертаємо у [0,1] для «сирих» пікселів.
    """
    n = len(ds) if max_samples is None else min(max_samples, len(ds))
    X = np.zeros((n, 28 * 28), dtype=np.float32)
    y = np.zeros((n,), dtype=np.int64)
    for i in range(n):
        img, label = ds[i]  # img: [1,28,28] у [-1,1]
        img = (img * 0.5 + 0.5)  # назад у [0,1]
        X[i] = img.view(-1).numpy()
        y[i] = int(label)
    return X, y


# ----------------------------- УТИЛІТИ ГРАФІКІВ -----------------------------
def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)


def savefig(fig, out_dir: str, fname: str):
    """Зберігаємо фігуру у PNG та закриваємо її (щоб не накопичувати пам’ять)."""
    ensure_outdir(out_dir)
    fig.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(cm, classes, title="Матриця плутанини"):
    """Легка, «світла» матриця плутанини з підписами класів."""
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(classes)), yticks=np.arange(len(classes)),
        xticklabels=classes, yticklabels=classes,
        title=title, ylabel="Справжня мітка", xlabel="Передбачена мітка"
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:d}", ha="center", va="center", color="black", fontsize=8)
    return fig


def plot_overall_bars(metrics_list, out_dir, fname="overall_bars.png"):
    """
    Зведений барчарт порівняння методів за: Accuracy, Macro-F1, Weighted-F1.
    metrics_list: список dict, кожен має keys: name, acc, macro_f1, weighted_f1
    """
    labels = ["Accuracy", "Macro-F1", "Weighted-F1"]
    names = [m["name"] for m in metrics_list]
    vals = np.array([[m["acc"], m["macro_f1"], m["weighted_f1"]] for m in metrics_list])  # shape: (M, 3)

    x = np.arange(len(labels))
    w = 0.8 / len(metrics_list)  # акуратне групування
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for i, nm in enumerate(names):
        ax.bar(x + (i - (len(names) - 1) / 2) * w, vals[i], width=w, label=nm)
    ax.set_xticks(x);
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Значення метрики")
    ax.set_title("Порівняння методів (тест)")
    ax.legend()
    savefig(fig, out_dir, fname)


# ----------------------------- ЗВІТИ/МЕТРИКИ -----------------------------
def compute_report(y_true, y_pred):
    """
    Повертає:
      - словник зі зведеними метриками,
      - масив F1 по класах (у порядку CLASS_NAMES),
      - повний текстовий звіт sklearn (для друку).
    """
    rep_dict = classification_report(
        y_true, y_pred, digits=4, zero_division=0, target_names=CLASS_NAMES, output_dict=True
    )
    macro_f1 = rep_dict["macro avg"]["f1-score"]
    weighted_f1 = rep_dict["weighted avg"]["f1-score"]
    per_class_f1 = np.array([rep_dict[name]["f1-score"] for name in CLASS_NAMES], dtype=float)
    rep_text = classification_report(y_true, y_pred, digits=4, zero_division=0, target_names=CLASS_NAMES)
    return {"macro_f1": macro_f1, "weighted_f1": weighted_f1}, per_class_f1, rep_text


def reliability_bins(y_true, proba, n_bins=10):
    """
    Калібрування top-1: розкладаємо впевненість моделі по бінках і порівнюємо з фактичною точністю.
    Повертаємо (mids, accs, confs, weights, ECE, Brier).
    """
    y_pred = np.argmax(proba, axis=1)
    conf = np.max(proba, axis=1)
    corr = (y_pred == y_true).astype(np.float32)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    mids, accs, confs, weights = [], [], [], []
    ece = 0.0
    N = len(y_true)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf >= lo) & (conf <= hi if i == n_bins - 1 else conf < hi)
        if mask.sum() == 0:
            continue
        mids.append((lo + hi) / 2.0)
        acc_bin = corr[mask].mean()
        conf_bin = conf[mask].mean()
        w = mask.sum() / N
        ece += w * abs(acc_bin - conf_bin)
        accs.append(acc_bin);
        confs.append(conf_bin);
        weights.append(w)

    # Brier (усереднений OVR для багатокласової задачі)
    K = proba.shape[1]
    Y = np.eye(K)[y_true]
    brier = np.mean([brier_score_loss(Y[:, k], proba[:, k]) for k in range(K)])

    return np.array(mids), np.array(accs), np.array(confs), np.array(weights), float(ece), float(brier)


def plot_reliability(mids, accs, confs, weights, title="Калібрування (top-1)"):
    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    ax.plot([0, 1], [0, 1], "--", color="gray", label="ідеально")
    ax.plot(confs, accs, marker="o", label="модель")
    for x, y, w in zip(confs, accs, weights):
        ax.text(x, y, f"{w * 100:.0f}%", fontsize=8, ha="center", va="bottom")
    ax.set_xlabel("Середня впевненість");
    ax.set_ylabel("Точність у біні")
    ax.set_title(title);
    ax.legend()
    return fig


# ----------------------------- SVM -----------------------------
def run_svm(X_train, y_train, X_test, y_test, kernel="rbf", C=2.0, gamma="scale",
            pca_components=80, out_dir="./plots_level2"):
    """
    Навчання/оцінка SVM:
      • Масштабування ознак (StandardScaler) обов'язкове для стабільності.
      • PCA (типово 80) часто радикально пришвидшує RBF і зменшує «висіння».
      • Для 'linear' використовуємо LinearSVC (швидко у високій розмірності).
    """
    t0 = time.perf_counter()
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    if pca_components is not None and pca_components > 0:
        pca = PCA(n_components=pca_components, random_state=42)
        Xtr = pca.fit_transform(Xtr)
        Xte = pca.transform(Xte)

    if kernel == "linear":
        # LinearSVC — оптимізований лінійний SVM, працює напряму у високій розмірності
        clf = LinearSVC(C=C, max_iter=5000, dual=True, verbose=0, random_state=42)
    else:
        # Класичний SVM з RBF-ядром (або іншим ядром із SVC)
        clf = SVC(C=C, kernel=kernel, gamma=gamma, probability=False, random_state=42)

    clf.fit(Xtr, y_train)
    y_pred = clf.predict(Xte)

    acc = accuracy_score(y_test, y_pred)
    rep_dict, f1_per_class, rep_text = compute_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    dt = time.perf_counter() - t0

    print("\n=== SVM ===")
    print(f"Kernel={kernel}, C={C}, gamma={gamma}, PCA={pca_components}, TrainTime={dt:.2f}s")
    print(f"Accuracy: {acc:.4f}")
    print(rep_text)

    fig = plot_confusion_matrix(cm, CLASS_NAMES, title=f"Матриця плутанини — SVM ({kernel})")
    savefig(fig, out_dir, f"svm_cm_{kernel}.png")

    return {"name": f"SVM-{kernel}", "acc": acc, "macro_f1": rep_dict["macro_f1"],
            "weighted_f1": rep_dict["weighted_f1"]}


# ----------------------------- EM / GMM -----------------------------
def run_em_gmm_classifier(X_train, y_train, X_test, y_test, n_components=2, pca_components=80,
                          covariance_type="diag", out_dir="./plots_level2"):
    """
    Байєсівський класифікатор на основі сумішей Гауса (GMM) з EM:
      • Окремий GMM для кожного класу у пониженому ПК-просторі.
      • Передбачення через argmax log P(x | class) + log P(class).
    """
    t0 = time.perf_counter()

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(X_train)
    Xte_s = scaler.transform(X_test)

    if pca_components is not None and pca_components > 0:
        pca = PCA(n_components=pca_components, random_state=42)
        Xtr = pca.fit_transform(Xtr_s)
        Xte = pca.transform(Xte_s)
    else:
        Xtr, Xte = Xtr_s, Xte_s

    # Оцінюємо апріорні ймовірності класів з тренувальних частот
    classes, counts = np.unique(y_train, return_counts=True)
    priors_log = {int(c): np.log(counts[i] / len(y_train)) for i, c in enumerate(classes)}

    # Тренуємо по GMM на кожен клас
    gmms = {}
    for c in classes:
        Xc = Xtr[y_train == c]
        gmm = GaussianMixture(
            n_components=n_components, covariance_type=covariance_type,
            random_state=42, reg_covar=1e-6, max_iter=200
        )
        gmm.fit(Xc)
        gmms[int(c)] = gmm

    log_post = np.zeros((len(Xte), len(classes)), dtype=np.float64)
    for idx, c in enumerate(classes):
        log_post[:, idx] = gmms[int(c)].score_samples(Xte) + priors_log[int(c)]
    y_pred = classes[np.argmax(log_post, axis=1)]

    acc = accuracy_score(y_test, y_pred)
    rep_dict, f1_per_class, rep_text = compute_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    dt = time.perf_counter() - t0

    print("\n=== EM / GMM Класифікатор ===")
    print(f"n_components={n_components}, PCA={pca_components}, cov='{covariance_type}', TrainTime={dt:.2f}s")
    print(f"Accuracy: {acc:.4f}")
    print(rep_text)

    fig = plot_confusion_matrix(cm, CLASS_NAMES, title=f"Матриця плутанини — EM/GMM ({n_components}×/клас)")
    savefig(fig, out_dir, f"gmm_cm_{covariance_type}_{n_components}.png")

    return {"name": f"GMM-{covariance_type}", "acc": acc, "macro_f1": rep_dict["macro_f1"],
            "weighted_f1": rep_dict["weighted_f1"]}


# ----------------------------- НЕВЕЛИКА CNN -----------------------------
class SmallCNN(nn.Module):
    """Проста CNN: 2 згорткові блоки + 2 повнозв’язних шари."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),  # 28→14
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2)  # 14→7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def select_device():
    """Вибір пристрою: CUDA → MPS (Apple) → CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def plot_training_curves(history, out_dir):
    """Криві навчання: loss та accuracy по епохах."""
    epochs = np.arange(1, len(history["loss"]) + 1)
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot(epochs, history["loss"], marker="o", label="train loss")
    ax.plot(epochs, history["acc"], marker="o", label="train acc")
    ax.set_xlabel("Епоха");
    ax.set_title("Криві навчання (CNN)");
    ax.set_ylim(0, 1.05)
    ax.legend()
    savefig(fig, out_dir, "cnn_training_curves.png")


def run_cnn(train_ds, test_ds, batch_size=128, epochs=5, lr=1e-3, out_dir="./plots_level2"):
    """
    Навчання/оцінка невеликої CNN:
      • Adam + CrossEntropyLoss.
      • Зберігаємо: криві навчання, матрицю плутанини, діаграму надійності, грід помилок.
    """
    device = select_device()
    print(f"[CNN] Пристрій: {device}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2,
                              pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2,
                             pin_memory=(device.type == "cuda"))

    model = SmallCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {"loss": [], "acc": []}

    # --- Навчання ---
    t0 = time.perf_counter()
    model.train()
    for epoch in range(1, epochs + 1):
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        history["loss"].append(train_loss)
        history["acc"].append(train_acc)
        print(f"[CNN] Епоха {epoch:02d}/{epochs}: loss={train_loss:.4f}, acc={train_acc:.4f}")

    train_time = time.perf_counter() - t0
    plot_training_curves(history, out_dir)

    # --- Оцінка на тесті ---
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(labels.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    proba = np.concatenate(all_probs, axis=0)

    acc = accuracy_score(y_true, y_pred)
    rep_dict, f1_per_class, rep_text = compute_report(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print("\n=== Невелика CNN ===")
    print(f"Епох: {epochs}, batch_size={batch_size}, lr={lr}, TrainTime={train_time:.2f}s")
    print(f"Accuracy: {acc:.4f}")
    print(rep_text)

    # збереження: матриця плутанини
    fig_cm = plot_confusion_matrix(cm, CLASS_NAMES, title="Матриця плутанини — CNN")
    savefig(fig_cm, out_dir, "cnn_cm.png")

    # калібрування (ECE/Brier) + графік
    mids, accs, confs, weights, ece, brier = reliability_bins(y_true, proba, n_bins=10)
    print(f"[CNN] Калібрування: ECE={ece:.4f}, Brier={brier:.4f}")
    fig_rel = plot_reliability(mids, accs, confs, weights, title="Калібрування (CNN)")
    savefig(fig_rel, out_dir, "cnn_reliability.png")

    # грід помилок за найпоширенішою плутаниною (top off-diagonal)
    cm_norm = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True).clip(min=1)
    i_best, j_best = 0, 0
    best_val = -1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i == j:
                continue
            if cm_norm[i, j] > best_val:
                best_val = cm_norm[i, j];
                i_best, j_best = i, j

    # збираємо індекси помилок i_best → j_best
    # (пройдёмось ще раз по тест-лоадеру з обчисленням предиктів)
    mis_imgs = []
    with torch.no_grad():
        for images, labels in test_loader:
            logits = model(images.to(device))
            preds = logits.argmax(dim=1).cpu().numpy()
            lbls = labels.numpy()
            mask = (lbls == i_best) & (preds == j_best)
            if mask.any():
                ims = images[mask].cpu().numpy()
                mis_imgs.append(ims)
            if sum(len(a) for a in mis_imgs) >= 16:
                break
    mis_imgs = np.concatenate(mis_imgs, axis=0) if len(mis_imgs) > 0 else np.empty((0, 1, 28, 28))
    k = min(16, len(mis_imgs))
    if k > 0:
        cols, rows = 4, math.ceil(k / 4)
        fig, axes = plt.subplots(rows, cols, figsize=(1.8 * cols, 1.8 * rows), constrained_layout=True)
        for ax, img in zip(axes.flat, mis_imgs[:k]):
            ax.imshow((img[0] * 0.5 + 0.5), cmap="gray")  # назад у [0,1] для візуалізації
            ax.axis("off")
        for ax in axes.flat[k:]:
            ax.axis("off")
        fig.suptitle(f"Помилки: {CLASS_NAMES[i_best]} → {CLASS_NAMES[j_best]} (CNN)")
        savefig(fig, out_dir, "cnn_top_confusion_examples.png")

    return {"name": "CNN", "acc": acc, "macro_f1": rep_dict["macro_f1"], "weighted_f1": rep_dict["weighted_f1"]}


# ----------------------------- ГОЛОВНА ФУНКЦІЯ -----------------------------
def main():
    parser = argparse.ArgumentParser()
    # Загальне
    parser.add_argument("--data-dir", type=str, default="./data", help="Каталог для кешу Fashion-MNIST")
    parser.add_argument("--seed", type=int, default=42, help="Seed для відтворюваності")
    parser.add_argument("--plots", type=str, default="save", choices=["save", "none", "show"],
                        help="Керування графіками: save=PNG у --out-dir; none=без графіків; show=інтерактивно (не радимо)")
    parser.add_argument("--out-dir", type=str, default="./plots_level2", help="Куди зберігати PNG, якщо --plots save")
    parser.add_argument("--run-all", action="store_true", help="Запустити SVM + EM/GMM + CNN послідовно")

    # Для SVM/EM (плоскі ознаки)
    parser.add_argument("--max-samples", type=int, default=20000,
                        help="Обмежити train для SVM/EM (швидше). None = повністю")
    parser.add_argument("--pca", type=int, default=80, help="К-сть компонент PCA для SVM/EM (рекомендовано 50–150)")

    # SVM
    parser.add_argument("--run-svm", action="store_true", help="Запустити SVM")
    parser.add_argument("--svm-kernel", type=str, default="rbf", choices=["rbf", "linear"], help="Тип ядра SVM")
    parser.add_argument("--svm-C", type=float, default=2.0, help="Регуляризація SVM (більше C → слабша)")
    parser.add_argument("--svm-gamma", type=str, default="scale", help="gamma для RBF ('scale'/'auto' або число)")

    # EM/GMM
    parser.add_argument("--run-em", action="store_true", help="Запустити EM/GMM класифікатор")
    parser.add_argument("--gmm-components", type=int, default=2, help="Кількість гаусів на клас")
    parser.add_argument("--gmm-cov", type=str, default="diag", choices=["full", "tied", "diag", "spherical"],
                        help="Тип коваріації GMM (diag/tied часто стабільні й швидкі)")

    # CNN
    parser.add_argument("--run-cnn", action="store_true", help="Запустити невелику CNN")
    parser.add_argument("--epochs", type=int, default=5, help="Кількість епох CNN")
    parser.add_argument("--batch-size", type=int, default=128, help="Розмір батчу CNN")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate для Adam")

    args = parser.parse_args()
    set_seed(args.seed)

    # якщо обрано run-all — увімкнемо всі три прапорці
    if args.run_all:
        args.run_svm = True
        args.run_em = True
        args.run_cnn = True

    # Вимкнути всі виклики plt.show(), якщо обрано save/none
    if args.plots != "show":
        plt.ioff()  # safety: ніяких інтерактивних вікон

    # 1) Завантажуємо датасети як тензори (1×28×28) — для CNN.
    train_ds_t, test_ds_t = load_fmnist_tensors(args.data_dir)

    # 2) Для SVM/EM також приготуємо плоскі ознаки (784). Обмежимо train, щоб «не висло».
    if args.run_svm or args.run_em:
        X_train, y_train = tensor_to_numpy_flat(train_ds_t, max_samples=args.max_samples)
        X_test, y_test = tensor_to_numpy_flat(test_ds_t, max_samples=None)

    all_metrics = []  # для зведеного барчарту

    # --- SVM ---
    if args.run_svm:
        gamma = args.svm_gamma
        if isinstance(gamma, str) and gamma not in ("scale", "auto"):
            try:
                gamma = float(gamma)
            except ValueError:
                gamma = "scale"
        m = run_svm(
            X_train, y_train, X_test, y_test,
            kernel=args.svm_kernel, C=args.svm_C, gamma=gamma,
            pca_components=args.pca, out_dir=args.out_dir
        )
        all_metrics.append(m)

    # --- EM/GMM ---
    if args.run_em:
        m = run_em_gmm_classifier(
            X_train, y_train, X_test, y_test,
            n_components=args.gmm_components,
            pca_components=args.pca,
            covariance_type=args.gmm_cov,
            out_dir=args.out_dir
        )
        all_metrics.append(m)

    # --- CNN ---
    if args.run_cnn:
        m = run_cnn(
            train_ds_t, test_ds_t,
            batch_size=args.batch_size, epochs=args.epochs, lr=args.lr,
            out_dir=args.out_dir
        )
        all_metrics.append(m)

    # 3) Зведений барчарт (якщо запущено ≥1 метод)
    if args.plots != "none" and len(all_metrics) >= 1:
        plot_overall_bars(all_metrics, args.out_dir, fname="overall_bars.png")

    # 4) Фінальні підсумки у консоль
    if all_metrics:
        print("\n=== ЗВЕДЕННЯ (тест) ===")
        for m in all_metrics:
            print(
                f"{m['name']:<12} | Acc={m['acc']:.4f} | MacroF1={m['macro_f1']:.4f} | WeightedF1={m['weighted_f1']:.4f}")
        if args.plots == "save":
            print(f"\n[OK] Усі графіки збережено у: {args.out_dir}")
    else:
        print("Не обрано жодного методу. Додай --run-svm / --run-em / --run-cnn або --run-all.")


if __name__ == "__main__":
    main()
