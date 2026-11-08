"""STEP 2 - Cleaning, splitting, EDA.

Design note: we keep TWO text versions.
  text        -> lightly cleaned (unicode/whitespace only). Used for stylometric features,
                 perplexity and transformers (punctuation/case are signal!).
  text_tfidf  -> lowercased copy for TF-IDF.
We do NOT strip punctuation/stopwords: that would destroy exactly the stylistic cues we want.
"""
import re, unicodedata
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from config import PROC, FIG, SEED, MIN_WORDS, SAMPLE_N


def clean(t):
    t = unicodedata.normalize("NFKC", str(t))
    t = re.sub(r"<[^>]+>", " ", t)           # html tags
    t = re.sub(r"http\S+|www\.\S+", " ", t)  # urls
    t = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def prepare(df):
    n0 = len(df)
    df = df.dropna(subset=["text", "label"]).copy()
    df["label"] = df["label"].astype(int)
    df["text"] = df["text"].map(clean)
    df["n_words"] = df["text"].str.split().str.len()
    df = df[df["n_words"] >= MIN_WORDS]
    # remove exact duplicates; drop texts that appear with conflicting labels
    conflict = df.groupby("text")["label"].nunique()
    df = df[~df["text"].isin(conflict[conflict > 1].index)]
    df = df.drop_duplicates("text")
    df["text_tfidf"] = df["text"].str.lower()
    print(f"cleaned {n0} -> {len(df)} rows")
    return df.reset_index(drop=True)


def eda(df, tag):
    df = df.reset_index(drop=True)   # FIX: concat of splits leaves duplicate index labels
    sns.set_theme(style="whitegrid")
    df = df.assign(cls=df["label"].map({0: "human", 1: "AI"}),
                   n_chars=df["text"].str.len())

    # 1) class balance + source breakdown
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    sns.countplot(data=df, x="cls", ax=ax[0]); ax[0].set_title("Class balance")
    sns.countplot(data=df, y="source", hue="cls", ax=ax[1]); ax[1].set_title("Source x class")
    plt.tight_layout(); plt.savefig(FIG / f"{tag}_balance.png", dpi=150); plt.close()

    # 2) length distributions
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(data=df, x="n_words", hue="cls", bins=60, log_scale=True, ax=ax[0]); ax[0].set_title("Words")
    sns.boxplot(data=df, x="source", y="n_words", hue="cls", ax=ax[1]); ax[1].set_yscale("log")
    ax[1].tick_params(axis="x", rotation=30)
    plt.tight_layout(); plt.savefig(FIG / f"{tag}_length.png", dpi=150); plt.close()

    # 3) generator distribution (AI only)
    if "model" in df and (df["label"] == 1).any():
        df[df.label == 1]["model"].value_counts().plot.bar(title="AI generators", figsize=(7, 3.5))
        plt.tight_layout(); plt.savefig(FIG / f"{tag}_generators.png", dpi=150); plt.close()

    # 4) discriminative words (log-odds)
    s = df.sample(min(len(df), 30000), random_state=SEED)
    cv = CountVectorizer(stop_words="english", min_df=20, max_features=20000)
    X = cv.fit_transform(s["text_tfidf"]); y = s["label"].values
    ai = np.asarray(X[y == 1].sum(0)).ravel() + 1
    hu = np.asarray(X[y == 0].sum(0)).ravel() + 1
    lo = np.log(ai / ai.sum()) - np.log(hu / hu.sum())
    w = np.array(cv.get_feature_names_out())
    order = np.argsort(lo)
    top = pd.concat([pd.Series(lo[order[:15]], w[order[:15]]),
                     pd.Series(lo[order[-15:]], w[order[-15:]])])
    top.plot.barh(figsize=(6, 7), color=["tab:blue"] * 15 + ["tab:red"] * 15,
                  title="Log-odds: blue=human-leaning, red=AI-leaning")
    plt.tight_layout(); plt.savefig(FIG / f"{tag}_logodds_words.png", dpi=150); plt.close()

    print(df.groupby("cls")["n_words"].describe())
    print(pd.crosstab(df["source"], df["cls"]))


if __name__ == "__main__":
    tr = prepare(pd.read_csv(PROC / "semeval_train_raw.csv"))
    te = prepare(pd.read_csv(PROC / "semeval_dev_raw.csv"))   # official dev -> our held-out TEST
    te = te[~te["text"].isin(set(tr["text"]))]                # leakage guard
    if SAMPLE_N:
        tr = tr.sample(min(SAMPLE_N, len(tr)), random_state=SEED)
    tr, va = train_test_split(tr, test_size=0.15, stratify=tr["label"], random_state=SEED)
    tr = tr.assign(split="train"); va = va.assign(split="val"); te = te.assign(split="test")

    out = pd.concat([tr, va, te], ignore_index=True)
    eda(out, "semeval")
    out.to_csv(PROC / "semeval_clean.csv", index=False)
    print(out["split"].value_counts())
    print(out.groupby("split")["label"].value_counts())

    try:
        es = pd.read_csv(PROC / "essays_raw.csv")
        es = prepare(es); es["split"] = "essay_all"
        eda(es, "essays"); es.to_csv(PROC / "essays_clean.csv", index=False)
    except FileNotFoundError:
        pass