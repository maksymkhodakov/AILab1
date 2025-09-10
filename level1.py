"""
Базовий (простий) класифікатор зображень на Fashion-MNIST.
Моделі: Логістична регресія (multinomial, softmax) та Наївний Байєс (GaussianNB).
Задача: 10-класова класифікація 28x28 сірих зображень (одяг).
Метрики: Accuracy, classification report, матриця плутанини.
"""

import argparse
import random
import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt


# ----------------------------- ВІДТВОРЮВАНІСТЬ -----------------------------
def set_seed(seed: int = 42):
    """
    Фіксуємо зерно (seed) для генераторів випадкових чисел у Python, NumPy та PyTorch.
    Це допомагає отримувати однакові результати при повторних запусках (якщо інші умови не змінюються).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------- ЗАВАНТАЖЕННЯ ДАНИХ -----------------------------
def load_fashion_mnist(data_dir: str, max_samples: int = None):
    """
    Завантажуємо (якщо потрібно — автоматично докачуються файли) датасет Fashion-MNIST.
    Повертаємо дані у вигляді плоских (flatten) векторів довжини 784 (28*28) та відповідних міток класів.

    Пояснення:
    - torchvision.datasets.FashionMNIST повертає пари (зображення, мітка);
    - transforms.ToTensor() перетворює кожне зображення у тензор з діапазоном значень [0, 1];
    - далі ми розплющуємо (flatten) 28x28 у вектор 784 для класичних моделей (LR, NB).
    """
    transform = transforms.Compose([
        transforms.ToTensor(),  # конвертація у тензор з нормалізацією до [0, 1]
        # Додаткової нормалізації тут не робимо — для LR використаємо StandardScaler нижче
    ])

    # Створюємо об’єкти датасету для навчальної та тестової частин
    train_ds = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)

    def to_numpy(ds, limit=None):
        """
        Перетворюємо PyTorch-датасет у NumPy-масиви:
        - X має форму (N, 784), де N — кількість прикладів;
        - y — вектор міток довжини N.
        Параметр limit дозволяє обмежити кількість прикладів (для швидших експериментів).
        """
        n = len(ds) if limit is None else min(limit, len(ds))
        X = np.zeros((n, 28 * 28), dtype=np.float32)
        y = np.zeros((n,), dtype=np.int64)
        for i in range(n):
            img, label = ds[i]  # img: тензор [1, 28, 28] зі значеннями в [0, 1]
            X[i] = img.view(-1).numpy()  # перетворення у плоский вектор (784,)
            y[i] = label  # ціле число від 0 до 9
        return X, y

    X_train, y_train = to_numpy(train_ds, limit=max_samples)  # навчальна вибірка (з обмеженням, якщо вказано)
    X_test, y_test = to_numpy(test_ds, limit=None)  # тестова вибірка (повна, 10k прикладів)

    return X_train, y_train, X_test, y_test


# ----------------------------- МОДЕЛІ ТА ОЦІНЮВАННЯ -----------------------------
def run_logistic_regression(X_train, y_train, X_test, y_test, C=1.0, max_iter=1000):
    """
    Навчаємо та оцінюємо Логістичну регресію (багатокласову, через softmax).
    Ключові моменти:
    - Перед LR виконуємо стандартизацію ознак (StandardScaler): віднімаємо середнє та ділимо на стандартне відхилення.
      Це робить простір ознак більш “однорідним” і покращує збіжність оптимізації.
    - Використовуємо solver="lbfgs"
    - Повертаємо accuracy, текстовий звіт sklearn та матрицю плутанини.
    """
    # Масштабування (навчаємось на train, застосовуємо однакове перетворення до test)
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Ініціалізуємо та навчаємо модель
    lr = LogisticRegression(
        C=C,  # обернена сила регуляризації (більший C -> слабша регуляризація)
        max_iter=max_iter,  # максимум ітерацій оптимізатора
        n_jobs=-1,  # використовувати всі ядра CPU, де можливо
        solver="lbfgs",  # оптимізатор, добре працює для багатокласових задач
        verbose=0
    )
    lr.fit(X_train_scaled, y_train)

    # Прогнозуємо мітки на тесті
    y_pred = lr.predict(X_test_scaled)

    # Рахуємо метрики якості
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        digits=4,
        zero_division=0,
        target_names=CLASS_NAMES
    )
    cm = confusion_matrix(y_test, y_pred)

    return acc, report, cm


def run_gaussian_nb(X_train, y_train, X_test, y_test):
    """
    Навчаємо та оцінюємо Наївного Байєса з Гаусовим припущенням (GaussianNB).
    Ключові моменти:
    - Не стандартизуємо ознаки (залишаємо пікселі в [0, 1]), оскільки GNB робить припущення
      про нормальний розподіл кожної ознаки в кожному класі та часто працює краще на “сирих” значеннях.
    - Повертаємо accuracy, текстовий звіт та матрицю плутанини.
    """
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        digits=4,
        zero_division=0,
        target_names=CLASS_NAMES
    )
    cm = confusion_matrix(y_test, y_pred)

    return acc, report, cm


# ----------------------------- ВІЗУАЛІЗАЦІЯ -----------------------------
def plot_confusion_matrix(cm, classes, title="Матриця плутанини"):
    """
    Будуємо матрицю плутанини:
    - По осі Y — істинні класи.
    - По осі X — передбачені класи.
    - У клітинках — кількість прикладів, що потрапили у відповідну комірку.
    Це дозволяє побачити типові помилки (які класи найчастіше плутаються).
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Oranges")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes, yticklabels=classes,
        title=title,
        ylabel="Справжня мітка (True)",
        xlabel="Передбачена мітка (Pred)"
    )
    # Повертаємо підписи класів по осі X, щоб не накладались
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Підписуємо кожну клітинку її значенням (ціле число прикладів)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:d}", ha="center", va="center")

    fig.tight_layout()
    plt.show()


# ----------------------------- ТОЧНІ НАЗВИ КЛАСІВ -----------------------------
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


# ----------------------------- ГОЛОВНА ФУНКЦІЯ -----------------------------
def main():
    """
    Точка входу:
    1) Парсимо аргументи командного рядка (шляхи, seed, обмеження кількості прикладів, гіперпараметри LR).
    2) Фіксуємо seed для відтворюваності.
    3) Завантажуємо та готуємо дані (flatten 28x28 -> 784).
    4) Навчаємо та оцінюємо Логістичну регресію (з попереднім масштабуванням ознак).
    5) Навчаємо та оцінюємо GaussianNB (на “сирих” пікселях).
    6) За потреби, будуємо матриці плутанини для обох моделей.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Каталог для збереження/завантаження Fashion-MNIST")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed для відтворюваності")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Максимальна кількість train-зразків (наприклад, 20000 для швидкості)")
    parser.add_argument("--lr-C", type=float, default=1.0,
                        help="Параметр C (обернена сила регуляризації) для Логістичної регресії")
    parser.add_argument("--lr-max-iter", type=int, default=200,
                        help="Максимум ітерацій оптимізатора для Логістичної регресії")
    parser.add_argument("--no-plots", action="store_true",
                        help="Якщо вказано — не будувати графіки (матриці плутанини)")
    args = parser.parse_args()

    # 1) Фіксуємо seed
    set_seed(args.seed)

    # 2) Завантажуємо дані
    print("Завантаження Fashion-MNIST...")
    X_train, y_train, X_test, y_test = load_fashion_mnist(args.data_dir, max_samples=args.max_samples)
    print(f"Розмірності: Train X={X_train.shape}, y={y_train.shape}; Test X={X_test.shape}, y={y_test.shape}")

    # 3) Логістична регресія (Multinomial Softmax)
    print("\n=== Логістична регресія (multinomial, softmax) ===")
    acc_lr, rep_lr, cm_lr = run_logistic_regression(
        X_train, y_train, X_test, y_test,
        C=args.lr_C, max_iter=args.lr_max_iter
    )
    print(f"Accuracy: {acc_lr:.4f}")
    print(rep_lr)

    # 4) Наївний Байєс (GaussianNB)
    print("\n=== Наївний Байєс (GaussianNB) ===")
    acc_nb, rep_nb, cm_nb = run_gaussian_nb(X_train, y_train, X_test, y_test)
    print(f"Accuracy: {acc_nb:.4f}")
    print(rep_nb)

    # 5) Матриці плутанини (за бажанням)
    if not args.no_plots:
        plot_confusion_matrix(cm_lr, CLASS_NAMES, title="Матриця плутанини — Логістична регресія")
        plot_confusion_matrix(cm_nb, CLASS_NAMES, title="Матриця плутанини — GaussianNB")

    print("\nГотово.")


# ----------------------------- ЗАПУСК -----------------------------
if __name__ == "__main__":
    main()
