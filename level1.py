"""
======================================================================
БАЗОВИЙ (ПРОСТИЙ) КЛАСИФІКАТОР ЗОБРАЖЕНЬ ДЛЯ Fashion-MNIST (РІВЕНЬ 1)
======================================================================

Мета:
  • Поставити чесний, відтворюваний бейслайн на двох класичних моделях:
      1) Логістична регресія (багатокласова softmax)
      2) Наївний Байєс (GaussianNB)
  • Порахувати ключові метрики на офіційному тест-наборі (10 000 зразків).
  • Побудувати легкі, неблокувальні графіки (PNG-файли), щоб проаналізувати результати.

Коротко про дизайн:
  • Завантаження Fashion-MNIST через torchvision, перетворення у плоскі вектори 28×28=784.
  • Для ЛР — обов’язкове масштабування ознак (StandardScaler) для стабільної збіжності.
  • Для NB — «сирі» пікселі в [0,1] без стандартизації (як правило, так краще для GNB).
"""

import argparse
import os
import time
import random
import numpy as np
import torch
from torchvision import datasets, transforms

# Імпортуємо моделі та метрики зі sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, brier_score_loss
)

# ----------------------------- НАЗВИ КЛАСІВ -----------------------------
# Важливо: ці назви відповідають міткам 0..9 у Fashion-MNIST.
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


# ----------------------------- ВІДТВОРЮВАНІСТЬ -----------------------------
def set_seed(seed: int = 42):
    """
    Фіксуємо зерно для, щоб запуски були відтворюваними:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------- ЗАВАНТАЖЕННЯ ДАНИХ -----------------------------
def load_fashion_mnist(data_dir: str, max_samples: int = None):
    """
    Завантажуємо Fashion-MNIST (якщо файлів немає — буде автоматичне докачування).
    Повертаємо X_train, y_train, X_test, y_test, де X — плоскі вектори довжини 784.

    Аргумент max_samples дозволяє обмежити навчальну вибірку (напр., 20000) для швидкого прогона.
    Тест залишаємо повним (10 000), щоб метрики були співставними.
    """
    # ToTensor() перетворює PIL Image → torch.Tensor у [0,1]
    transform = transforms.ToTensor()
    train_ds = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)

    def to_numpy(ds, limit=None):
        """
        Переносимо дані з PyTorch-датасету у NumPy-масиви:
          X: (N, 784), y: (N,)
        Картинки мають форму [1, 28, 28], тож робимо .view(-1) для flatten.
        """
        n = len(ds) if limit is None else min(limit, len(ds))
        X = np.zeros((n, 28 * 28), dtype=np.float32)
        y = np.zeros((n,), dtype=np.int64)
        for i in range(n):
            img, lab = ds[i]  # img ∈ [0,1] розміром [1,28,28]
            X[i] = img.view(-1).numpy()  # перетворюємо у (784,)
            y[i] = int(lab)  # ціла мітка 0..9
        return X, y

    # Навчальна частина — обмежена/повна; тест — завжди повний.
    X_train, y_train = to_numpy(train_ds, limit=max_samples)
    X_test, y_test = to_numpy(test_ds, limit=None)
    return X_train, y_train, X_test, y_test


# ----------------------------- МЕТРИКИ / ЗВІТИ -----------------------------
def compute_report(y_true, y_pred):
    """
    Рахуємо докладний звіт sklearn і дістаємо з нього:
      • macro_f1: середній F1 по класах (без ваг),
      • weighted_f1: середній F1 із вагами класів,
      • per_class_f1: масив F1 у порядку CLASS_NAMES,
      • rep_text: форматований текстовий звіт (для виводу в консоль).
    """
    rep_dict = classification_report(
        y_true, y_pred, digits=4, zero_division=0, target_names=CLASS_NAMES, output_dict=True
    )
    macro_f1 = rep_dict["macro avg"]["f1-score"]
    weighted_f1 = rep_dict["weighted avg"]["f1-score"]
    per_class_f1 = np.array([rep_dict[name]["f1-score"] for name in CLASS_NAMES], dtype=float)
    rep_text = classification_report(
        y_true, y_pred, digits=4, zero_division=0, target_names=CLASS_NAMES
    )
    return {"macro_f1": macro_f1, "weighted_f1": weighted_f1}, per_class_f1, rep_text


def reliability_bins(y_true, proba, n_bins=10):
    """
    Діаграма надійності (калібрування) для top-1:
      1) Для кожного зразка беремо максимальну ймовірність серед 10 класів (confidence).
      2) Обчислюємо, чи передбачення правильне (corr ∈ {0,1}).
      3) Ділимо conf на бінки (0..1), у кожному біні рахуємо:
           – середню впевненість (conf_bin),
           – фактичну точність (acc_bin),
           – частку прикладів (weight),
         і сумуємо Expected Calibration Error (ECE).
      4) Додатково рахуємо багатокласовий Brier score (усереднений OVR).
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
        # Останній бін включає праву межу (<=), щоб захопити conf=1.0
        mask = (conf >= lo) & (conf <= hi if i == n_bins - 1 else conf < hi)
        if mask.sum() == 0:
            continue
        mids.append((lo + hi) / 2.0)
        acc_bin = corr[mask].mean()
        conf_bin = conf[mask].mean()
        w = mask.sum() / N
        ece += w * abs(acc_bin - conf_bin)
        accs.append(acc_bin)
        confs.append(conf_bin)
        weights.append(w)

    # Багатокласовий Brier: середній по всіх класах (One-vs-Rest)
    K = proba.shape[1]
    Y = np.eye(K)[y_true]  # one-hot матриця (N,K)
    brier = np.mean([brier_score_loss(Y[:, k], proba[:, k]) for k in range(K)])
    return np.array(mids), np.array(accs), np.array(confs), np.array(weights), float(ece), float(brier)


# ----------------------------- ПЛОТИ (save/show/none) -----------------------
def save_or_show(fig, mode, out_dir, fname):
    """
    Безпечний вивід графіків:
      • mode == "save": зберегти у PNG (не блокує виконання, створює директорію за потреби);
      • mode == "show": показати інтерактивно (може блокувати до закриття вікна);
      • після цього фігуру закриваємо, щоб не накопичувати об’єкти в пам’яті.
    """
    import matplotlib.pyplot as plt
    if mode == "show":
        plt.show()
    elif mode == "save":
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(cm, classes, title, mode, out_dir, fname, cmap="Blues"):
    """
    Матриця плутанини (у світлій палітрі, не блокуючи виконання).
    constrained_layout=True — альтернатива tight_layout (менше попереджень).
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(classes)), yticks=np.arange(len(classes)),
        xticklabels=classes, yticklabels=classes,
        title=title, ylabel="Справжня мітка", xlabel="Передбачена мітка"
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # Підписуємо кожну клітинку значенням (ціле число прикладів)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:d}", ha="center", va="center", color="black")
    save_or_show(fig, mode, out_dir, fname)


def plot_overall_bars(metrics_lr, metrics_nb, mode, out_dir, fname):
    """
    Зведені барчарти: Accuracy, Macro-F1, Weighted-F1 — порівняння LR vs NB.
    Це швидкий візуальний спосіб побачити, де саме ЛР виграє у NB (або навпаки).
    """
    import matplotlib.pyplot as plt
    labels = ["Accuracy", "Macro-F1", "Weighted-F1"]
    lr_vals = [metrics_lr["acc"], metrics_lr["macro_f1"], metrics_lr["weighted_f1"]]
    nb_vals = [metrics_nb["acc"], metrics_nb["macro_f1"], metrics_nb["weighted_f1"]]
    x = np.arange(len(labels));
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.bar(x - w / 2, lr_vals, width=w, label="LogReg")
    ax.bar(x + w / 2, nb_vals, width=w, label="GaussianNB")
    ax.set_xticks(x);
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Значення метрики")
    ax.set_title("Порівняння загальних метрик")
    ax.legend()
    save_or_show(fig, mode, out_dir, fname)


def plot_per_class_f1_bars(f1_lr, f1_nb, classes, mode, out_dir, fname):
    """
    Барчарт F1 по кожному класу (10 класів): дві колонки на клас (LR та NB).
    """
    import matplotlib.pyplot as plt
    x = np.arange(len(classes));
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.bar(x - w / 2, f1_lr, width=w, label="LogReg")
    ax.bar(x + w / 2, f1_nb, width=w, label="GaussianNB")
    ax.set_xticks(x);
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1")
    ax.set_title("F1 по класах (LR vs NB)")
    ax.legend()
    save_or_show(fig, mode, out_dir, fname)


def plot_reliability(mids, accs, confs, weights, title, mode, out_dir, fname):
    """
    Діаграма надійності: ідеальна лінія y=x + фактична точність.
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    ax.plot([0, 1], [0, 1], "--", color="gray", label="ідеально")
    ax.plot(confs, accs, marker="o", label="модель")
    for x, y, w in zip(confs, accs, weights):
        ax.text(x, y, f"{w * 100:.0f}%", fontsize=8, ha="center", va="bottom")
    ax.set_xlabel("Середня впевненість");
    ax.set_ylabel("Точність у біні")
    ax.set_title(title);
    ax.legend()
    save_or_show(fig, mode, out_dir, fname)


# ----------------------------- НАВЧАННЯ МОДЕЛЕЙ -----------------------------
def run_logistic_regression(X_train, y_train, X_test, y_test,
                            C=1.0, max_iter=600, solver="newton-cg", tol=1e-3, use_sgd=False):
    """
    Логістична регресія (багатокласова softmax) з передмасштабуванням ознак:
      • StandardScaler покращує геометрію простору та збіжність оптимізатора.

    Повертаємо:
      – metrics: словник з acc, macro_f1, weighted_f1,
      – rep_text: повний текстовий classification report,
      – cm: матриця плутанини (цілі числа),
      – clf: навчена модель (LogisticRegression або SGDClassifier),
      – scaler: навчений StandardScaler,
      – train_time: час навчання в секундах,
      – used_iters: скільки ітерацій реально використано (None для SGD),
      – per_class_f1: масив F1 по класах,
      – calib: кортеж для діаграми надійності (mids, accs, confs, weights, ece, brier).
    """
    t0 = time.perf_counter()

    # 1) Масштабуємо ознаки (навчимо scaler на train і застосовуємо до test)
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    # 2) Обираємо реалізацію ЛР: «класичну» або швидку стохастичну
    if use_sgd:
        # SGDClassifier з log_loss — це стохастична логістична регресія.
        # Має ранню зупинку, добре підходить для швидких прогонів.
        clf = SGDClassifier(
            loss="log_loss", alpha=1e-4, penalty="l2",
            learning_rate="optimal", early_stopping=True,
            n_iter_no_change=5, tol=tol, random_state=42
        )
        clf.fit(Xtr, y_train)
        used_iters = getattr(clf, "t_", None)  # к-сть оновлень (не завжди інтерпретовано як «ітерації»)
        proba = clf.predict_proba(Xte)
    else:
        # Класична ЛР з обраним solver/tol/max_iter. Не задаємо multi_class (щоб не ловити FutureWarning).
        clf = LogisticRegression(
            C=C, max_iter=max_iter, solver=solver, tol=tol, n_jobs=-1, verbose=0
        )
        clf.fit(Xtr, y_train)
        # n_iter_ може бути масивом (по класах); беремо максимум як «фактичні ітерації»
        n_iter_arr = np.array(clf.n_iter_)
        used_iters = int(n_iter_arr.max()) if n_iter_arr.ndim else int(n_iter_arr)
        proba = clf.predict_proba(Xte)

    train_time = time.perf_counter() - t0

    # 3) Оцінювання: передбачення, метрики, матриця плутанини
    y_pred = clf.predict(Xte)
    acc = accuracy_score(y_test, y_pred)
    rep_dict, per_class_f1, rep_text = compute_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # 4) Збір метрик у зручний формат
    metrics = {"acc": acc, **rep_dict}
    calib = reliability_bins(y_test, proba, n_bins=10)
    return metrics, rep_text, cm, clf, scaler, train_time, used_iters, per_class_f1, calib


def run_gaussian_nb(X_train, y_train, X_test, y_test):
    """
    Наївний Байєс (GaussianNB) — проста генеративна модель.
    Працює напряму на «сирих» пікселях [0,1] без стандартизації.
    Повертаємо той самий набір артефактів, що й для ЛР (де це має сенс).
    """
    t0 = time.perf_counter()
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    y_pred = gnb.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    rep_dict, per_class_f1, rep_text = compute_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # NB має predict_proba → теж можемо будувати діаграму надійності
    proba = gnb.predict_proba(X_test)
    calib = reliability_bins(y_test, proba, n_bins=10)
    metrics = {"acc": acc, **rep_dict}
    return metrics, rep_text, cm, gnb, train_time, per_class_f1, calib


# ----------------------------- ГОЛОВНА ТОЧКА ВХОДУ -----------------------------
def main():
    parser = argparse.ArgumentParser()
    # Де кешувати/читати Fashion-MNIST
    parser.add_argument("--data-dir", type=str, default="./data")
    # Seed для відтворюваності
    parser.add_argument("--seed", type=int, default=42)
    # Обмеження train для швидких прогонів (тест завжди повний)
    parser.add_argument("--max-samples", type=int, default=None)
    # Гіперпараметри ЛР
    parser.add_argument("--lr-C", type=float, default=1.0, help="Обернена сила L2: більше C → слабша регуляризація")
    parser.add_argument("--lr-max-iter", type=int, default=600,
                        help="Ліміт ітерацій для ЛР (уникаємо ConvergenceWarning)")
    parser.add_argument("--lr-solver", type=str, default="newton-cg",
                        choices=["lbfgs", "newton-cg", "saga", "sgd"],
                        help="'sgd' — стохастична ЛР (SGDClassifier), інші — LogisticRegression")
    parser.add_argument("--lr-tol", type=float, default=1e-3,
                        help="Критерій зупинки оптимізації (чим більше — тим швидше зупинка)")
    # Куди дівати графіки
    parser.add_argument("--plots", type=str, default="save", choices=["none", "save", "show"],
                        help="none=без графіків; save=PNG у --out-dir; show=показати інтерактивно (може блокувати)")
    parser.add_argument("--out-dir", type=str, default="./plots/level1", help="Директорія для PNG, коли --plots save")
    args = parser.parse_args()

    # 0) Фіксуємо seed і друкуємо базову інфу про запуск
    set_seed(args.seed)
    print("Завантаження Fashion-MNIST...")
    X_train, y_train, X_test, y_test = load_fashion_mnist(args.data_dir, max_samples=args.max_samples)
    print(f"Розмірності: Train X={X_train.shape}, Test X={X_test.shape}")

    # 1) ЛОГІСТИЧНА РЕГРЕСІЯ (із масштабуванням ознак)
    print("\n=== Логістична регресія (multinomial, softmax) ===")
    use_sgd = (args.lr_solver == "sgd")
    m_lr, rep_lr, cm_lr, lr, scaler, t_lr, it_lr, f1_lr, calib_lr = run_logistic_regression(
        X_train, y_train, X_test, y_test,
        C=args.lr_C, max_iter=args.lr_max_iter,
        solver=(None if use_sgd else args.lr_solver),  # якщо 'sgd', solver для LogisticRegression не використовується
        tol=args.lr_tol, use_sgd=use_sgd
    )
    # Логи часу/ітерацій допомагають контролювати збіжність без попереджень ConvergenceWarning
    print(
        f"[LR] Час навчання: {t_lr:.2f} c; використано ітерацій: {it_lr if it_lr is not None else '-'} / {args.lr_max_iter if not use_sgd else '(SGD)'}")
    print(f"Accuracy: {m_lr['acc']:.4f}")
    print(rep_lr)
    mids_lr, accs_lr, confs_lr, weights_lr, ece_lr, brier_lr = calib_lr
    print(f"[LR] Калібрування: ECE={ece_lr:.4f}, Brier={brier_lr:.4f}")

    # 2) НАЇВНИЙ БАЙЄС (на «сирих» пікселях)
    print("\n=== Наївний Байєс (GaussianNB) ===")
    m_nb, rep_nb, cm_nb, nb, t_nb, f1_nb, calib_nb = run_gaussian_nb(X_train, y_train, X_test, y_test)
    print(f"[NB] Час навчання: {t_nb:.2f} c")
    print(f"Accuracy: {m_nb['acc']:.4f}")
    print(rep_nb)
    mids_nb, accs_nb, confs_nb, weights_nb, ece_nb, brier_nb = calib_nb
    print(f"[NB] Калібрування: ECE={ece_nb:.4f}, Brier={brier_nb:.4f}")

    # 3) ГРАФІКИ (збереження/показ/вимкнено — залежно від --plots)
    if args.plots != "none":
        # Матриці плутанини — видно структуру помилок по класах
        plot_confusion_matrix(cm_lr, CLASS_NAMES, "Матриця плутанини — ЛР", args.plots, args.out_dir, "cm_lr.png",
                              cmap="Blues")
        plot_confusion_matrix(cm_nb, CLASS_NAMES, "Матриця плутанини — NB", args.plots, args.out_dir, "cm_nb.png",
                              cmap="Oranges")

        # Загальні барчарти — наочне порівняння метрик між моделями
        plot_overall_bars(
            {"acc": m_lr["acc"], "macro_f1": m_lr["macro_f1"], "weighted_f1": m_lr["weighted_f1"]},
            {"acc": m_nb["acc"], "macro_f1": m_nb["macro_f1"], "weighted_f1": m_nb["weighted_f1"]},
            args.plots, args.out_dir, "overall_bars.png"
        )

        # По-класах: де саме одна модель краща/гірша
        _, f1_lr_again, _ = compute_report(y_test, lr.predict(scaler.transform(X_test)))
        _, f1_nb_again, _ = compute_report(y_test, nb.predict(X_test))
        plot_per_class_f1_bars(f1_lr_again, f1_nb_again, CLASS_NAMES, args.plots, args.out_dir, "per_class_f1.png")

        # Діаграми надійності — наскільки «правдиві» ймовірності
        plot_reliability(mids_lr, accs_lr, confs_lr, weights_lr, "Калібрування (LR)", args.plots, args.out_dir,
                         "reliability_lr.png")
        plot_reliability(mids_nb, accs_nb, confs_nb, weights_nb, "Калібрування (NB)", args.plots, args.out_dir,
                         "reliability_nb.png")

    if args.plots == "save":
        print(f"Графіки збережено у: {args.out_dir}")
    elif args.plots == "none":
        print("Графіки вимкнено (--plots none)")


# ЗАПУСК: python level1.py
if __name__ == "__main__":
    main()
