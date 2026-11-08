"""Score a text with the trained models: TF-IDF LogReg, TF-IDF Linear SVM, Random Forest (selected features
if 09_feature_selection.py has been run) and the fine-tuned DistilBERT (if 06_finetune_distilbert.py has been run).

Usage:
  py predict.py                      (opens a file-picker window: .txt / .pdf / .docx)
  py predict.py --file essay.txt
  py predict.py "paste text here"
"""
import sys, json, importlib
import numpy as np, pandas as pd, joblib, torch
from config import MODELS, PRED, MAX_CHARS_FEATURES, MIN_WORDS, TARGET_FPR

# modules whose filenames start with a digit can't be imported normally
feat = importlib.import_module("03_features")
prep = importlib.import_module("02_clean_eda")
dev = feat.dev


def _cutoff(neg_scores, fpr=TARGET_FPR):
    """Lowest threshold whose false-positive rate on human texts is <= fpr."""
    s = np.sort(np.asarray(neg_scores))
    return float(s[int(np.ceil((1 - fpr) * len(s))) - 1])


# ------------------------------------------------------------------ traditional models
import style_text as style
_prep = MODELS / "tfidf_prep.json"          # written by 17_style_ngrams.py save <variant>; absent = word 1-2-grams
PREP_MODE = json.load(open(_prep))["mode"] if _prep.exists() else "word"
vec = joblib.load(MODELS / "tfidf_vectorizer.joblib")
num_cols = joblib.load(MODELS / "numeric_feature_names.joblib")          # all 32 features
clfs = {n: joblib.load(MODELS / f"{n}.joblib") for n in ("tfidf_logreg", "tfidf_linsvm", "rf_features")}
KIND = {"tfidf_logreg": "proba", "tfidf_linsvm": "decision", "rf_features": "proba", "distilbert": "proba"}
DEFAULT_THR = {"tfidf_logreg": 0.5, "tfidf_linsvm": 0.0, "rf_features": 0.5, "distilbert": 0.5}

pr = pd.read_csv(PRED / "traditional_predictions.csv")
LOW_FPR_THR = {}
for n in clfs:
    v = pr[(pr.model_name == n) & (pr.split == "val")]
    LOW_FPR_THR[n] = _cutoff(v.loc[v.y_true == 0, "score"].values)

# Random Forest on the selected features replaces the 32-feature forest when it exists
RF_DESC = f"all {len(num_cols)} features"
_sel = MODELS / "selected_features.json"
if _sel.exists() and (MODELS / "rf_features_selected.joblib").exists():
    meta = json.load(open(_sel))
    clfs["rf_features"] = joblib.load(MODELS / "rf_features_selected.joblib")
    num_cols = meta["features"]
    LOW_FPR_THR["rf_features"] = float(meta["strict_threshold"])
    RF_DESC = f"selected {len(num_cols)} features"

# ------------------------------------------------------------------ DistilBERT (optional)
HAS_DISTILBERT = False
DB_WINDOW, DB_MAX_WINDOWS = 510, 2          # up to ~1,000 tokens, scored in 510-token windows (+CLS/SEP = 512)
_db_dir = MODELS / "distilbert"
if (_db_dir / "config.json").exists() and (PRED / "distilbert_predictions.csv").exists():
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        _tok = AutoTokenizer.from_pretrained(str(_db_dir))
        _db = AutoModelForSequenceClassification.from_pretrained(str(_db_dir)).to(dev).eval()
        _p = pd.read_csv(PRED / "distilbert_predictions.csv")
        LOW_FPR_THR["distilbert"] = _cutoff(_p[(_p.split == "val") & (_p.y_true == 0)].score.values)
        HAS_DISTILBERT = True
    except Exception as e:                                              # keep the app usable without it
        print("DistilBERT could not be loaded:", e)


@torch.no_grad()
def distilbert_scores(texts):
    """AI-probability per text: mean over up to DB_MAX_WINDOWS windows of 510 tokens."""
    out = []
    for t in texts:
        ids = _tok(t, add_special_tokens=False, truncation=True,
                   max_length=DB_WINDOW * DB_MAX_WINDOWS)["input_ids"] or [_tok.unk_token_id]
        batch = [[_tok.cls_token_id] + ids[i:i + DB_WINDOW] + [_tok.sep_token_id]
                 for i in range(0, len(ids), DB_WINDOW)]
        L = max(map(len, batch))
        inp = torch.tensor([b + [_tok.pad_token_id] * (L - len(b)) for b in batch]).to(dev)
        att = torch.tensor([[1] * len(b) + [0] * (L - len(b)) for b in batch]).to(dev)
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            logits = _db(input_ids=inp, attention_mask=att).logits
        out.append(float(torch.softmax(logits.float(), -1)[:, 1].mean().cpu()))
    return out


# ------------------------------------------------------------------ scoring
def score_text(raw):
    text = prep.clean(raw)
    n_words = len(text.split())
    t = text[:MAX_CHARS_FEATURES]
    row = feat.lexical_statistical(t)
    if any(c.startswith("syn_") for c in num_cols):
        row.update(feat.syntactic(feat.nlp.pipe([t])).iloc[0].to_dict())
    row.update(feat.perplexity_feats([t]).iloc[0].to_dict())            # always: shown in the app
    if any(c.startswith("sem_adj_") for c in num_cols):
        row.update(feat.coherence_feats([t]).iloc[0].to_dict())
    X_num = pd.DataFrame([row]).reindex(columns=num_cols).replace([np.inf, -np.inf], np.nan).fillna(0).values
    X_tfidf = vec.transform(style.prepare(PREP_MODE, [text], nlp=feat.nlp))

    out = {}
    for n, clf in clfs.items():
        X = X_tfidf if n.startswith("tfidf") else X_num
        s = clf.predict_proba(X)[0, 1] if KIND[n] == "proba" else clf.decision_function(X)[0]
        out[n] = float(s)
    if HAS_DISTILBERT:
        out["distilbert"] = distilbert_scores([text])[0]
    return n_words, out, row


TRADITIONAL = ("tfidf_logreg", "tfidf_linsvm", "rf_features")
TRANSFORMER = ("distilbert",)
LABELS = {"LIKELY": "Likely AI-generated", "UNCERTAIN": "Uncertain (mixed signals)",
          "NONE": "No strong evidence of AI"}


def group_verdict(scores, names):
    """Verdict for one group of models. Majority rule inside the group (2 of 3 traditional models; the single
    DistilBERT model decides alone). 'strong' = above the 1%-false-positive threshold, 'weak' = above the default."""
    names = [n for n in names if n in scores]
    if not names:
        return None
    maj = len(names) // 2 + 1
    strong = sum(scores[m] > LOW_FPR_THR[m] for m in names)
    weak = sum(scores[m] > DEFAULT_THR[m] for m in names)
    level = "LIKELY" if strong >= maj else "UNCERTAIN" if (strong >= 1 or weak >= maj) else "NONE"
    return dict(level=level, label=LABELS[level], strong=strong, weak=weak, n=len(names), names=names)


def verdicts(scores):
    """Two separate verdicts: traditional models and DistilBERT (None if DistilBERT is not installed)."""
    return {"Traditional models": group_verdict(scores, TRADITIONAL),
            "DistilBERT": group_verdict(scores, TRANSFORMER)}


def verdict(scores):
    """Text summary of both verdicts (used by the command-line version)."""
    parts = []
    for name, v in verdicts(scores).items():
        if v:
            parts.append(f"{name}: {v['label'].upper()} ({v['strong']} of {v['n']} above the 1%-false-positive threshold)")
    return " | ".join(parts)


def read_file(path):
    p = path.lower()
    if p.endswith(".pdf"):
        from pypdf import PdfReader
        return " ".join((pg.extract_text() or "") for pg in PdfReader(path).pages)
    if p.endswith(".docx"):
        from docx import Document
        return " ".join(par.text for par in Document(path).paragraphs)
    return open(path, encoding="utf-8", errors="ignore").read()


def pick_file():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Select a text file to analyze",
            filetypes=[("Text / PDF / Word", "*.txt *.pdf *.docx"), ("All files", "*.*")])
        root.destroy()
        return path
    except Exception:
        return input("File picker unavailable. Enter the full path to your file: ").strip().strip('"')


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--file":
        raw = read_file(sys.argv[2])
    elif len(sys.argv) > 1:
        raw = " ".join(sys.argv[1:])
    else:
        path = pick_file()
        if not path:
            sys.exit("No file selected.")
        print("Selected:", path)
        raw = read_file(path)

    n_words, scores, row = score_text(raw)
    print(f"\nwords: {n_words} | random forest: {RF_DESC} | distilbert: {'on' if HAS_DISTILBERT else 'off'}")
    if n_words < MIN_WORDS:
        print(f"Text is shorter than {MIN_WORDS} words - the models were not trained on text this short. Result unreliable.")
    print(f"\n{'model':14s} {'score':>8s}  default-thr  1%FPR-thr")
    for n, s in scores.items():
        print(f"{n:14s} {s:8.3f}  {'AI' if s > DEFAULT_THR[n] else 'human':>10s}  {'AI' if s > LOW_FPR_THR[n] else 'human':>9s}")
    print(f"\nperplexity: {row['sem_perplexity']:.1f} | burstiness: {row['stat_burstiness']:.2f}")
    print("\nRESULTS (reported separately):")
    for name, v in verdicts(scores).items():
        if v:
            print(f"  {name:20s} {v['label']}  ({v['strong']} of {v['n']} above the 1%-FPR threshold)")
