# 01_load_data.py  (STEP 1 - load SemEval-2024 Task 8 Subtask A, monolingual)
import pandas as pd
from config import ROOT, PROC

SRC = ROOT / "dataset"
TRAIN_SOURCES = {"wikipedia", "wikihow", "reddit", "arxiv", "peerread"}

def load(split):
    df = pd.read_parquet(SRC / f"{split}-00000-of-00001.parquet")
    df["origin"] = f"semeval_{split}"
    return df

if __name__ == "__main__":
    data = {s: load(s) for s in ("train", "dev", "test")}

    for name, d in data.items():
        print(f"\n=== {name}: {d.shape} ===")
        print("columns:", d.columns.tolist())
        print(d.dtypes)
        print("nulls:\n", d.isna().sum())
        if "label" in d:
            print("label counts (0=human, 1=AI):\n", d["label"].value_counts(dropna=False))
        if "source" in d:
            print("sources:\n", d["source"].value_counts())
        if "model" in d:
            print("generators:\n", d["model"].value_counts())
        print(d.head(2).T)

    # ---- sanity checks: is this really the English monolingual set? ----
    tr, dv = data["train"], data["dev"]
    assert set(tr["label"].unique()) <= {0, 1}, "train labels are not binary - wrong subtask?"
    extra = set(tr["source"].str.lower().unique()) - TRAIN_SOURCES
    assert not extra, f"unexpected sources in train: {extra} (multilingual file?)"
    overlap = len(set(tr["text"]) & set(dv["text"]))
    print(f"\nSanity OK. train/dev exact-text overlap: {overlap}")

    tr.to_csv(PROC / "semeval_train_raw.csv", index=False)
    dv.to_csv(PROC / "semeval_dev_raw.csv", index=False)
    data["test"].to_csv(PROC / "semeval_test_raw.csv", index=False)  # kept aside for generalization tests
    print("Saved raw CSVs to", PROC)