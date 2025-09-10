"""
СЕРЕДНІЙ РІВЕНЬ — Fashion-MNIST (28x28, 10 класів)
Методи:
  1) SVM (лінійний / RBF) + (опційно) PCA для пришвидшення/стабілізації
  2) EM (Expectation-Maximization) через GaussianMixture (GMM) як байєсівський класифікатор
  3) Невелика CNN (2 згорткові блоки)

Вивід:
  - Accuracy, classification report з назвами класів, матриця плутанини
"""

import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC, LinearSVC
from sklearn.mixture import GaussianMixture

# ----------------------------- НАЗВИ КЛАСІВ -----------------------------
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


# ----------------------------- ВІДТВОРЮВАНІСТЬ -----------------------------
def set_seed(seed: int = 42):
    """
    Фіксуємо seed для Python/NumPy/PyTorch для повторюваності експериментів.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------- ЗАВАНТАЖЕННЯ ДАНИХ -----------------------------
def load_fmnist_tensors(data_dir: str):
    """
    Завантаження Fashion-MNIST як тензорів (для CNN).
    Трансформації:
      - ToTensor(): перетворення у тензор з діапазоном [0,1]
      - Normalize(mean=0.5, std=0.5): проста нормалізація до [-1,1]
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
    Перетворити датасет тензорів у NumPy-масиви плоских ознак (784) для класичних методів (SVM/GMM).
    Якщо max_samples задано — обмежуємо train-вибірку (для швидкості).
    """
    n = len(ds) if max_samples is None else min(max_samples, len(ds))
    X = np.zeros((n, 28 * 28), dtype=np.float32)
    y = np.zeros((n,), dtype=np.int64)
    for i in range(n):
        img, label = ds[i]  # img: тензор [1,28,28] у [-1,1] після Normalize
        img = (img * 0.5 + 0.5)  # повернемо у [0,1] (для коректного трактування як "сирих" пікселів)
        X[i] = img.view(-1).numpy()
        y[i] = label
    return X, y


# ----------------------------- ВІЗУАЛІЗАЦІЯ -----------------------------
def plot_confusion_matrix(cm, classes, title="Матриця плутанини"):
    """
    Побудова матриці плутанини з підписами класів.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Oranges")
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
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:d}", ha="center", va="center")
    fig.tight_layout()
    plt.show()


# ----------------------------- SVM КЛАСИФІКАЦІЯ -----------------------------
def run_svm(X_train, y_train, X_test, y_test, kernel="rbf", C=2.0, gamma="scale", pca_components=None, no_plots=False):
    """
    Навчання та оцінювання SVM.
      - Для RBF доцільно застосувати стандартизацію (StandardScaler) + (опційно) PCA (для пришвидшення).
      - Для 'linear' використовуємо LinearSVC (швидше на великій кількості ознак).
    """
    # Масштабування завжди корисне для SVM
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Опційно: PCA (зменшує розмірність, пришвидшує навчання та робить ядрові методи стабільнішими)
    if pca_components is not None and pca_components > 0:
        pca = PCA(n_components=pca_components, random_state=42)
        X_train_s = pca.fit_transform(X_train_s)
        X_test_s = pca.transform(X_test_s)

    if kernel == "linear":
        # LinearSVC — оптимізований лінійний SVM, працює напряму у високій розмірності
        clf = LinearSVC(C=C, max_iter=5000, dual=True, verbose=0, random_state=42)
    else:
        # Класичний SVM з RBF-ядром (або іншим ядром із SVC)
        clf = SVC(C=C, kernel=kernel, gamma=gamma, probability=False, random_state=42)

    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4, zero_division=0, target_names=CLASS_NAMES)
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== SVM ===")
    print(f"Kernel: {kernel}, C={C}, gamma={gamma}, PCA={pca_components}")
    print(f"Accuracy: {acc:.4f}")
    print(report)
    if not no_plots:
        plot_confusion_matrix(cm, CLASS_NAMES, title=f"Матриця плутанини — SVM ({kernel})")


# ----------------------------- EM / GMM КЛАСИФІКАЦІЯ -----------------------------
def run_em_gmm_classifier(X_train, y_train, X_test, y_test, n_components=2, pca_components=50, covariance_type="full",
                          no_plots=False):
    """
    Байєсівський класифікатор на основі сумішей Гауса (GMM) з EM:
      - Для КОЖНОГО класу тренуємо окрему GaussianMixture із n_components (EM-алгоритм оцінює параметри).
      - Для передбачення: r_k(x) = log P(x | class=k) + log P(class=k), обираємо клас з максимумом.
      - Щоб уникнути проблем у високій розмірності, зазвичай корисно попередньо зменшити розмірність через PCA.

    Пояснення:
      - Це "generative" підхід: ми моделюємо розподіл P(x | y=k), а потім обчислюємо апостеріорні ймовірності через правило Байєса.
      - Навіть якщо класи не гаусові, сума кількох гаусів часто добре апроксимує реальні розподіли.

    Параметри:
      - n_components: кількість гаусів на клас (2–5 часто достатньо для Fashion-MNIST у пониженій розмірності).
      - pca_components: розмірність після PCA (типово 30–100).
      - covariance_type: 'full' (найгнучкіше), 'tied', 'diag', 'spherical' (швидші/стійкіші варіанти).
    """
    # Масштабування перед PCA (стандартизація до нульового середнього та одиничної дисперсії)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # PCA для уникнення "прокляття розмірності" у GMM
    if pca_components is not None and pca_components > 0:
        pca = PCA(n_components=pca_components, random_state=42)
        X_train_p = pca.fit_transform(X_train_s)
        X_test_p = pca.transform(X_test_s)
    else:
        X_train_p, X_test_p = X_train_s, X_test_s

    # Оцінюємо апріорні ймовірності класів з тренувальних частот
    classes, counts = np.unique(y_train, return_counts=True)
    priors = {int(c): np.log(counts[i] / len(y_train)) for i, c in enumerate(classes)}

    # Тренуємо по GMM на кожен клас
    gmms = {}
    for c in classes:
        X_c = X_train_p[y_train == c]
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=42,
            reg_covar=1e-6,  # невелика регуляризація для числової стабільності
            max_iter=200
        )
        gmm.fit(X_c)
        gmms[int(c)] = gmm

    # Передбачення: обчислюємо log P(x | class=c) + log P(class=c) для кожного класу
    log_post = np.zeros((len(X_test_p), len(classes)), dtype=np.float64)
    for idx, c in enumerate(classes):
        log_likelihood = gmms[int(c)].score_samples(X_test_p)  # log P(x | class=c)
        log_post[:, idx] = log_likelihood + priors[int(c)]  # + log P(class=c)
    y_pred = classes[np.argmax(log_post, axis=1)]

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4, zero_division=0, target_names=CLASS_NAMES)
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== EM / GMM Класифікатор ===")
    print(f"n_components={n_components}, PCA={pca_components}, covariance_type={covariance_type}")
    print(f"Accuracy: {acc:.4f}")
    print(report)
    if not no_plots:
        plot_confusion_matrix(cm, CLASS_NAMES, title=f"Матриця плутанини — EM/GMM ({n_components}×/клас)")


# ----------------------------- НЕВЕЛИКА CNN -----------------------------
class SmallCNN(nn.Module):
    """
    Дуже компактна CNN для Fashion-MNIST:
      - Блок1: Conv(1→32, 3x3) → ReLU → MaxPool(2x2)
      - Блок2: Conv(32→64, 3x3) → ReLU → MaxPool(2x2)
      - FC:   64*7*7 → 128 → 10
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28→14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 14→7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def run_cnn(train_ds, test_ds, batch_size=128, epochs=5, lr=1e-3, no_plots=False):
    """
    Навчання та оцінювання невеликої CNN.
      - Використовуємо Adam, крос-ентропійні втрати.
      - Для швидких експериментів достатньо 3–5 епох на CPU; на GPU можна більше.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[CNN] Пристрій: {device}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2,
                              pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2,
                             pin_memory=torch.cuda.is_available())

    model = SmallCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # --- Цикл навчання ---
    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
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
        print(f"Епоха {epoch:02d}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}")

    # --- Оцінювання на тесті ---
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.numpy())
    y_pred = np.concatenate(all_preds)
    y_test = np.concatenate(all_labels)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4, zero_division=0, target_names=CLASS_NAMES)
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== Невелика CNN ===")
    print(f"Епох: {epochs}, batch_size={batch_size}, lr={lr}")
    print(f"Accuracy: {acc:.4f}")
    print(report)
    if not no_plots:
        plot_confusion_matrix(cm, CLASS_NAMES, title="Матриця плутанини — CNN")


# ----------------------------- ГОЛОВНА ФУНКЦІЯ -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="./data", help="Каталог для збереження/читання Fashion-MNIST")
    parser.add_argument("--seed", type=int, default=42, help="Seed для відтворюваності")
    parser.add_argument("--max-samples", type=int, default=None, help="Обмеження train для класичних методів (SVM/GMM)")
    parser.add_argument("--no-plots", action="store_true", help="Не будувати матриці плутанини/графіки")

    # SVM параметри
    parser.add_argument("--run-svm", action="store_true", help="Запустити SVM")
    parser.add_argument("--svm-kernel", type=str, default="rbf", choices=["rbf", "linear"], help="Ядро для SVM")
    parser.add_argument("--svm-C", type=float, default=2.0, help="Регуляризація SVM (більше C -> слабша регуляризація)")
    parser.add_argument("--svm-gamma", type=str, default="scale", help="Параметр gamma для RBF ('scale' або число)")
    parser.add_argument("--pca", type=int, default=None, help="К-сть компонент PCA для SVM/EM (напр., 50 або 100)")

    # EM / GMM параметри
    parser.add_argument("--run-em", action="store_true", help="Запустити EM/GMM-класифікатор")
    parser.add_argument("--gmm-components", type=int, default=2, help="Кількість гаусів на клас")
    parser.add_argument("--gmm-cov", type=str, default="full", choices=["full", "tied", "diag", "spherical"],
                        help="Тип коваріації GMM")

    # CNN параметри
    parser.add_argument("--run-cnn", action="store_true", help="Запустити невелику CNN")
    parser.add_argument("--epochs", type=int, default=5, help="Кількість епох для CNN")
    parser.add_argument("--batch-size", type=int, default=128, help="Розмір батчу для CNN")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate для Adam")

    args = parser.parse_args()
    set_seed(args.seed)

    # Завантажуємо датасет як тензори один раз (і для класики, і для CNN)
    train_ds_t, test_ds_t = load_fmnist_tensors(args.data_dir)

    # Для SVM/EM потрібні плоскі ознаки (784). Обмеження train за бажанням.
    if args.run_svm or args.run_em:
        X_train, y_train = tensor_to_numpy_flat(train_ds_t, max_samples=args.max_samples)
        X_test, y_test = tensor_to_numpy_flat(test_ds_t, max_samples=None)

    # --- SVM ---
    if args.run_svm:
        gamma = args.svm_gamma
        # Якщо gamma задана числом (рядком), спробуємо привести до float
        if isinstance(gamma, str) and gamma not in ("scale", "auto"):
            try:
                gamma = float(gamma)
            except ValueError:
                gamma = "scale"
        run_svm(
            X_train, y_train, X_test, y_test,
            kernel=args.svm_kernel, C=args.svm_C, gamma=gamma,
            pca_components=args.pca, no_plots=args.no_plots
        )

    # --- EM/GMM ---
    if args.run_em:
        run_em_gmm_classifier(
            X_train, y_train, X_test, y_test,
            n_components=args.gmm_components,
            pca_components=args.pca if args.pca is not None else 50,  # за замовчуванням корисно робити PCA≈50
            covariance_type=args.gmm_cov,
            no_plots=args.no_plots
        )

    # --- CNN ---
    if args.run_cnn:
        # Для CNN використовуємо весь train, оскільки DataLoader сам шардить батчі.
        run_cnn(
            train_ds_t, test_ds_t,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            no_plots=args.no_plots
        )


if __name__ == "__main__":
    main()
