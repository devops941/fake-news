"""Load, label, and clean the Fake.csv / True.csv dataset."""

from pathlib import Path

import pandas as pd

from utils.clean_text import preprocess, remove_reuters_prefix

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

FAKE_LABEL = 0
REAL_LABEL = 1
LABEL_NAMES = {FAKE_LABEL: "FAKE", REAL_LABEL: "REAL"}


def load_raw(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load Fake.csv and True.csv, tag each row with its label, and concatenate."""
    fake_df = pd.read_csv(data_dir / "Fake.csv")
    true_df = pd.read_csv(data_dir / "True.csv")

    fake_df["label"] = FAKE_LABEL
    true_df["label"] = REAL_LABEL

    df = pd.concat([fake_df, true_df], ignore_index=True)
    return df


def prepare_dataset(data_dir: Path = DATA_DIR, random_state: int = 42) -> pd.DataFrame:
    """Full data-prep pipeline: merge, shuffle, de-leak, and clean.

    Returns a DataFrame with columns: title, text, subject, date, label, content, clean_text
    """
    df = load_raw(data_dir)

    # Shuffle so fake/real rows aren't grouped together before the train/test split.
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    df["text"] = df["text"].fillna("")
    df["title"] = df.get("title", "").fillna("")

    # Data-leakage fix: strip the "CITY (Reuters) - " dateline that almost every
    # REAL article starts with, otherwise the model just learns to spot "Reuters".
    df["text"] = df["text"].apply(remove_reuters_prefix)

    # Combine headline + body before cleaning, since both carry signal.
    df["content"] = (df["title"] + " " + df["text"]).str.strip()
    df["clean_text"] = df["content"].apply(preprocess)

    # Drop rows that end up empty after cleaning (e.g. non-text-only rows).
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)

    return df


if __name__ == "__main__":
    dataset = prepare_dataset()
    print(dataset["label"].value_counts().rename(index=LABEL_NAMES))
    print(dataset[["content", "clean_text", "label"]].head())
