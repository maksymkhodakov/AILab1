"""
Складний рівень: Трансформери для класифікації Fashion-MNIST (10 класів).
Модель за замовчуванням: ViT (google/vit-base-patch16-224-in21k) з HuggingFace.
Підтримка повного fine-tune та linear probe (freeze бекбону).

Що робить скрипт:
  1) Завантажує Fashion-MNIST (28x28, grayscale) через torchvision.
  2) Конвертує у RGB та змінює розмір до 224×224, нормалізує за статистикою моделі.
  3) Файнтюнить трансформер під 10 класів.
  4) Рахує accuracy, classification report (з назвами класів) і будує матрицю плутанини (cmap="Blues").

Примітки:
  - Повторне завантаження датасету НЕ відбувається (torchvision кешує у data_dir).
  - При наявності CUDA автоматично вмикається fp16 (можна вимкнути ключем).
"""

import argparse

import os
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")  # вимкнути швидкий (Rust) даунлоадер


import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torchvision import datasets
from torchvision.transforms import functional as TF
from PIL import Image

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from transformers import (
    AutoImageProcessor,
    ViTForImageClassification,
    TrainingArguments,
    Trainer,
    default_data_collator,
)

# ----------------------------- НАЗВИ КЛАСІВ -----------------------------
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


# ----------------------------- ВІДТВОРЮВАНІСТЬ -----------------------------
def set_seed(seed: int = 42):
    """
    Фіксуємо зерна генераторів випадкових чисел, щоб отримувати стабільні результати
    (наскільки це можливо при тих самих налаштуваннях).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------- ПІДГОТОВКА ДАНИХ -----------------------------
class HFTransformWrapper:
    """
    Обгортка над HuggingFace ImageProcessor для застосування до зображень torchvision.
    Завдання:
      - Перетворити grayscale → RGB (трансформерам потрібні 3 канали).
      - Змінити розмір до очікуваного (наприклад, 224×224).
      - Нормалізувати за mean/std моделі.
      - Повернути тензор 'pixel_values' (C,H,W).
    """

    def __init__(self, processor):
        self.processor = processor
        # Зазвичай у ViT очікується 224×224; processor.size може містити dict.
        size = processor.size
        if isinstance(size, dict):
            self.resize = size.get("shortest_edge", 224)
        else:
            self.resize = size if size is not None else 224

    def __call__(self, pil_img: Image.Image) -> torch.Tensor:
        # Якщо зображення в градаціях сірого — конвертуємо у RGB
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        # Зміна розміру з антиаліасингом
        pil_img = TF.resize(pil_img, [self.resize, self.resize], antialias=True)
        # У process передаємо список зображень і повертаємо тензор
        enc = self.processor(images=[pil_img], return_tensors="pt")
        # enc["pixel_values"] має форму [1, 3, H, W] — прибираємо першу розмірність
        return enc["pixel_values"].squeeze(0)


class TorchvisionHFDataset(torch.utils.data.Dataset):
    """
    Об'єднує torchvision FashionMNIST із HuggingFace-процесором:
      - на виході повертає dict із 'pixel_values' (тензор) та 'labels' (int).
    """

    def __init__(self, tv_dataset, hf_transform: HFTransformWrapper):
        self.ds = tv_dataset
        self.tf = hf_transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]  # img: PIL (через FashionMNIST(..., transform=None))
        pixel_values = self.tf(img)
        return {"pixel_values": pixel_values, "labels": int(label)}


def plot_confusion_matrix(cm, classes, title="Матриця плутанини"):
    """
    Відмалювати матрицю плутанини у світлій палітрі 'Blues'.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")  # світліша карта
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes, yticklabels=classes,
        title=title,
        ylabel="Справжня мітка",
        xlabel="Передбачена мітка"
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # Підписи значень у клітинках
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:d}", ha="center", va="center", color="black")
    fig.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="./data", help="Каталог для кешу Fashion-MNIST (torchvision)")
    parser.add_argument("--model-id", type=str, default="google/vit-base-patch16-224-in21k",
                        help="HuggingFace модель (наприклад, 'google/vit-base-patch16-224-in21k' або 'facebook/deit-tiny-patch16-224')")
    parser.add_argument("--output-dir", type=str, default="./vit_fmnist_ckpt", help="Куди зберігати чекпоїнти/логи")
    parser.add_argument("--epochs", type=int, default=5, help="Кількість епох навчання")
    parser.add_argument("--batch-size", type=int, default=64, help="Розмір батчу")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate (AdamW)")
    parser.add_argument("--weight-decay", type=float, default=0.05, help="L2-регуляризація (weight decay)")
    parser.add_argument("--seed", type=int, default=42, help="Seed для відтворюваності")
    parser.add_argument("--freeze-base", action="store_true", help="Заморозити бекбон (linear probe)")
    parser.add_argument("--fp16", action="store_true", help="Примусово увімкнути fp16 (інакше автоматично якщо CUDA)")
    parser.add_argument("--no-plots", action="store_true", help="Не показувати матрицю плутанини")
    args = parser.parse_args()

    set_seed(args.seed)

    # 1) Завантажуємо torchvision-датасет як PIL-зображення (без трансформів тут!)
    #    torchvision сам кешує дані у data_dir, тож повторного скачування не буде.
    train_tv = datasets.FashionMNIST(root=args.data_dir, train=True, download=True, transform=None)
    test_tv = datasets.FashionMNIST(root=args.data_dir, train=False, download=True, transform=None)

    # 2) ImageProcessor і перетворення під модель (resize, normalize, RGB)
    processor = AutoImageProcessor.from_pretrained(args.model_id, use_fast=True)
    hf_transform = HFTransformWrapper(processor)

    # 3) Обгортаємо у датаcети для Trainer (повертаємо dict з pixel_values та labels)
    train_ds = TorchvisionHFDataset(train_tv, hf_transform)
    test_ds = TorchvisionHFDataset(test_tv, hf_transform)

    # 4) Завантажуємо модель класифікації та ініціалізуємо під 10 класів
    num_labels = 10
    id2label = {i: name for i, name in enumerate(CLASS_NAMES)}
    label2id = {name: i for i, name in enumerate(CLASS_NAMES)}

    model = ViTForImageClassification.from_pretrained(
        args.model_id,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )

    # 4.1) (Опційно) Заморозити всі шари, крім класифікаційної голови — швидкий linear probe
    if args.freeze_base:
        for name, param in model.named_parameters():
            if not name.startswith("classifier."):
                param.requires_grad = False
        # Для надійності можна ініціалізувати голову заново (не обов'язково)
        if hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
            nn.init.xavier_uniform_(model.classifier.weight)
            if model.classifier.bias is not None:
                nn.init.zeros_(model.classifier.bias)

    # 5) TrainingArguments — налаштування тренування
    use_fp16 = args.fp16 or torch.cuda.is_available()
    training_args = TrainingArguments(
        dataloader_num_workers=0,
        logging_strategy="steps",
        logging_steps=10,
        disable_tqdm=False,
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",  # оцінювати наприкінці кожної епохи
        save_strategy="epoch",  # зберігати чекпоїнт кожної епохи
        load_best_model_at_end=True,  # завантажити найкращий за метрикою (eval_loss) після тренування
        metric_for_best_model="accuracy",  # ми повернемо accuracy у compute_metrics
        greater_is_better=True,
        save_total_limit=2,  # зберігати лише кілька останніх чекпоїнтів
        fp16=use_fp16,  # пришвидшення на CUDA (або вимкнеться на CPU)
        report_to=[]  # без WandB/Hub за замовчуванням
    )

    # 6) Функція метрик для Trainer
    def compute_metrics(eval_preds):
        """
        HuggingFace Trainer передає логіти/ймовірності та істинні мітки.
        Тут рахуємо accuracy, але ще окремо нижче виведемо full report і CM.
        """
        logits, labels = eval_preds
        preds = np.argmax(logits, axis=1)
        acc = (preds == labels).mean().item()
        return {"accuracy": acc}

    # 7) Колатор (склеює батчі). Для VisionTransformer достатньо стандартного data_collator.
    data_collator = default_data_collator

    # 8) Створюємо Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,  # будемо дивитися на тест одразу (для простоти навчальної задачі)
        tokenizer=processor,  # не обов'язково, але не зашкодить
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # 9) Навчання
    trainer.train()

    # 10) Оцінювання — беремо прогноз на тесті, рахуємо докладні метрики
    preds_output = trainer.predict(test_ds)
    y_logits = preds_output.predictions
    y_true = preds_output.label_ids
    y_pred = np.argmax(y_logits, axis=1)

    acc = accuracy_score(y_true, y_pred)
    print(f"\n=== ViT — Підсумкова якість на тесті ===\nAccuracy: {acc:.4f}")

    # Повний звіт з назвами класів
    report = classification_report(y_true, y_pred, digits=4, zero_division=0, target_names=CLASS_NAMES)
    print(report)

    # Матриця плутанини
    cm = confusion_matrix(y_true, y_pred)
    if not args.no_plots:
        plot_confusion_matrix(cm, CLASS_NAMES, title="Матриця плутанини — ViT")

    print("\nГотово. Найкраща модель (за accuracy) збережена у:", args.output_dir)


if __name__ == "__main__":
    main()
