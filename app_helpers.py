"""Pure-Python helpers for app.py (no Streamlit imports, so they are easy to test)."""
import re
from html import escape
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RANGE = {"tfidf_logreg": (0.0, 1.0), "tfidf_linsvm": (-3.0, 3.0), "rf_features": (0.0, 1.0), "distilbert": (0.0, 1.0)}
LABEL = {"tfidf_logreg": "TF-IDF + Logistic Regression",
         "tfidf_linsvm": "TF-IDF + Linear SVM",
         "rf_features": "Random Forest (writing-style features)",
         "distilbert": "DistilBERT (fine-tuned transformer)"}


def zone(score, default, strict):
    """'high' = above strict (1% FPR) cut-off, 'mid' = above default only, 'low' = below both."""
    if score > strict:
        return "high"
    if score > default:
        return "mid"
    return "low"


def gauge_fig(name, score, default, strict):
    lo, hi = RANGE[name]
    fig, ax = plt.subplots(figsize=(6.2, 0.95))
    if strict > default:
        ax.axvspan(lo, default, color="#2e7d32", alpha=.30)
        ax.axvspan(default, strict, color="#f9a825", alpha=.40)
        ax.axvspan(strict, hi, color="#c62828", alpha=.30)
    else:
        ax.axvspan(lo, strict, color="#2e7d32", alpha=.30)
        ax.axvspan(strict, hi, color="#c62828", alpha=.30)
    ax.axvline(min(max(score, lo), hi), color="black", lw=3)
    ax.axvline(strict, color="black", lw=1, ls="--")
    ax.set_xlim(lo, hi); ax.set_yticks([])
    ax.set_title(f"{LABEL[name]}  |  score {score:.3f}", fontsize=9, loc="left")
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    return fig


def make_chunks(P, raw, target=120, max_chunks=60):
    """Split into ~target-word chunks, respecting paragraph breaks where they exist."""
    paras = [P.prep.clean(p) for p in re.split(r"\n\s*\n", raw) if p.strip()]
    if len(paras) < 3:
        paras = [P.prep.clean(raw)]
    chunks = []
    for p in paras:
        cur, n = [], 0
        for s in re.split(r"(?<=[.!?])\s+", p):
            cur.append(s); n += len(s.split())
            if n >= target:
                chunks.append(" ".join(cur)); cur, n = [], 0
        if cur:
            chunks.append(" ".join(cur))
    merged = []
    for c in chunks:
        if merged and len(c.split()) < 40:
            merged[-1] += " " + c
        else:
            merged.append(c)
    return merged[:max_chunks]


def chunk_table(P, raw):
    """Score every chunk with the TF-IDF models (+ DistilBERT when available); the RF needs whole-text features."""
    chunks = make_chunks(P, raw)
    db = P.distilbert_scores(chunks) if getattr(P, "HAS_DISTILBERT", False) and chunks else None
    rows = []
    for i, c in enumerate(chunks):
        X = P.vec.transform(P.style.prepare(P.PREP_MODE, [c], nlp=P.feat.nlp))
        sc = {"tfidf_logreg": float(P.clfs["tfidf_logreg"].predict_proba(X)[0, 1]),
              "tfidf_linsvm": float(P.clfs["tfidf_linsvm"].decision_function(X)[0])}
        if db is not None:
            sc["distilbert"] = db[i]
        n_strong = sum(zone(v, P.DEFAULT_THR[k], P.LOW_FPR_THR[k]) == "high" for k, v in sc.items())
        n_weak = sum(zone(v, P.DEFAULT_THR[k], P.LOW_FPR_THR[k]) != "low" for k, v in sc.items())
        n = len(c.split())
        if n < 40:
            level = "short"
        elif n_strong >= len(sc) // 2 + 1:
            level = "high"
        elif n_weak >= 1:
            level = "mid"
        else:
            level = "low"
        rows.append({"text": c, "words": n, "logreg": sc["tfidf_logreg"], "svm": sc["tfidf_linsvm"],
                     "distilbert": sc.get("distilbert", np.nan), "level": level})
    return pd.DataFrame(rows)


def chunks_html(df):
    bg = {"high": "#f8d7da", "mid": "#fff3cd", "low": "#d4edda", "short": "#e9ecef"}
    tag = {"high": "strong AI signal", "mid": "weak AI signal", "low": "no signal", "short": "too short to judge"}
    out = []
    for r in df.itertuples():
        out.append(f'<div style="background:{bg[r.level]};color:#111;padding:8px 10px;margin:5px 0;'
                   f'border-radius:6px;font-size:0.92rem">{escape(r.text)}'
                   f'<br><small style="color:#444">{tag[r.level]} | LogReg {r.logreg:.2f} | SVM {r.svm:.2f}'
                   f'{"" if pd.isna(r.distilbert) else f" | DistilBERT {r.distilbert:.2f}"}</small></div>')
    return "".join(out)


def explain_words(P, raw, k=10):
    """Words in THIS text that push the Logistic Regression towards AI / human."""
    X = P.vec.transform([P.prep.clean(raw).lower()]).tocoo()
    coef = P.clfs["tfidf_logreg"].coef_[0]
    names = np.array(P.vec.get_feature_names_out())
    contrib = X.data * coef[X.col]
    order = np.argsort(contrib)
    ai = [(names[X.col[i]], float(contrib[i])) for i in order[::-1][:k] if contrib[i] > 0]
    hu = [(names[X.col[i]], float(contrib[i])) for i in order[:k] if contrib[i] < 0]
    return (pd.DataFrame(ai, columns=["word / phrase", "pushes towards AI"]),
            pd.DataFrame(hu, columns=["word / phrase", "pushes towards human"]))


def explain_style(P, row, ref, k=10):
    """Top Random-Forest features: this text's value vs. the human and AI averages in training data."""
    imp = pd.Series(P.clfs["rf_features"].feature_importances_, index=P.num_cols).sort_values(ascending=False)
    out = []
    for f in imp.index[:k]:
        v, h, a = float(row[f]), ref.loc["human", f], ref.loc["ai", f]
        sd = ref.loc["sd", f] or 1.0
        out.append({"feature": f, "this text": round(v, 3), "human avg": round(h, 3), "AI avg": round(a, 3),
                    "closer to": "AI" if abs(v - a) / sd < abs(v - h) / sd else "human",
                    "importance": round(float(imp[f]), 3)})
    return pd.DataFrame(out)


def reference_stats(features_csv, num_cols):
    """Human / AI feature means from train+val rows only (never the test split)."""
    ft = pd.read_csv(features_csv)
    ft = ft[ft["split"].isin(["train", "val"])]
    ref = pd.DataFrame({"human": ft[ft.label == 0][num_cols].mean(),
                        "ai": ft[ft.label == 1][num_cols].mean(),
                        "sd": ft[num_cols].std()}).T
    return ref
