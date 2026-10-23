"""Train and compare the three classifiers, then persist the best one.

Usage: python train_model.py  (thin wrapper at the project root)
"""

import json
import warnings
from pathlib import Path

import joblib
import matplotlib

# PassiveAggressiveClassifier is deprecated in sklearn >=1.8 in favour of
# SGDClassifier(loss='hinge', ...), but it's the algorithm this project is
# built around (and the spec's viva questions are about it), so keep using it.
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

matplotlib.use("Agg")  # headless-safe backend for saving figures to disk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier

from utils.data_prep import LABEL_NAMES, prepare_dataset

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "model"
REPORTS_DIR = ROOT_DIR / "reports"

MODEL_BUILDERS = {
    "Passive Aggressive": lambda: PassiveAggressiveClassifier(max_iter=1000, random_state=42),
    "Multinomial Naive Bayes": lambda: MultinomialNB(),
    "Logistic Regression": lambda: LogisticRegression(max_iter=1000),
}


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }
    cm = confusion_matrix(y_test, y_pred)
    return metrics, cm


def save_confusion_matrix(name, cm, out_dir: Path):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["FAKE", "REAL"])
    fig, ax = plt.subplots(figsize=(4, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {name}")
    fig.tight_layout()
    fig.savefig(out_dir / f"confusion_matrix_{name.lower().replace(' ', '_')}.png", dpi=150)
    plt.close(fig)


def save_comparison_chart(results: dict, out_dir: Path):
    metrics = ["accuracy", "precision", "recall", "f1"]
    names = list(results.keys())

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(names))
    width = 0.2
    for i, metric in enumerate(metrics):
        values = [results[name][metric] for name in names]
        ax.bar([p + i * width for p in x], values, width=width, label=metric)

    ax.set_xticks([p + width * 1.5 for p in x])
    ax.set_xticklabels(names, rotation=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "model_comparison.png", dpi=150)
    plt.close(fig)


def save_wordclouds(df, out_dir: Path):
    from wordcloud import WordCloud

    for label, name in LABEL_NAMES.items():
        text = " ".join(df.loc[df["label"] == label, "clean_text"])
        if not text.strip():
            continue
        wc = WordCloud(width=800, height=400, background_color="white").generate(text)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"Most Frequent Words — {name}")
        fig.tight_layout()
        fig.savefig(out_dir / f"wordcloud_{name.lower()}.png", dpi=150)
        plt.close(fig)


def train_and_evaluate(data_dir=None, test_size=0.2, random_state=42, save_artifacts=True):
    df = prepare_dataset(data_dir) if data_dir else prepare_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"], test_size=test_size, random_state=random_state, stratify=df["label"]
    )

    vectorizer = TfidfVectorizer(max_df=0.7)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    results = {}
    trained_models = {}
    confusion_matrices = {}

    for name, build in MODEL_BUILDERS.items():
        model = build()
        model.fit(X_train_tfidf, y_train)
        metrics, cm = evaluate_model(name, model, X_test_tfidf, y_test)
        results[name] = metrics
        trained_models[name] = model
        confusion_matrices[name] = cm
        print(f"{name:25s} acc={metrics['accuracy']:.4f} prec={metrics['precision']:.4f} "
              f"rec={metrics['recall']:.4f} f1={metrics['f1']:.4f}")

    best_name = max(results, key=lambda n: results[n]["f1"])
    best_model = trained_models[best_name]
    print(f"\nBest model by F1-score: {best_name}")

    if save_artifacts:
        MODEL_DIR.mkdir(exist_ok=True)
        REPORTS_DIR.mkdir(exist_ok=True)

        joblib.dump(best_model, MODEL_DIR / "model.pkl")
        joblib.dump(vectorizer, MODEL_DIR / "tfidf.pkl")
        (MODEL_DIR / "model_name.txt").write_text(best_name, encoding="utf-8")

        with open(REPORTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
            json.dump({"best_model": best_name, "results": results}, f, indent=2)

        for name, cm in confusion_matrices.items():
            save_confusion_matrix(name, cm, REPORTS_DIR)
        save_comparison_chart(results, REPORTS_DIR)
        try:
            save_wordclouds(df, REPORTS_DIR)
        except ImportError:
            print("wordcloud not installed — skipping word cloud plots.")

        print(f"\nSaved model/vectoriser to {MODEL_DIR}, charts/metrics to {REPORTS_DIR}")

    return results, best_name


if __name__ == "__main__":
    train_and_evaluate()
