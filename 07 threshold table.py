"""STEP 7 - Threshold table for the three traditional models.

Reads outputs/predictions/traditional_predictions.csv, finds the score cut-offs that give a chosen
false-positive rate on VALIDATION humans, and reports what each cut-off does on val and on the
(unseen-generator) test set. Also prints the decision bands used by predict.py.
Output: console tables + outputs/predictions/threshold_table.csv
"""
import numpy as np, pandas as pd
from config import PRED

pd.set_option("display.width", 220)
DEFAULT = {"tfidf_logreg": 0.5, "tfidf_linsvm": 0.0, "rf_features": 0.5}
FPR_LEVELS = [("5% FPR", 0.05), ("1% FPR  <- used by predict.py", 0.01), ("0.1% FPR", 0.001)]

pr = pd.read_csv(PRED / "traditional_predictions.csv")

def cutoff(neg_scores, fpr):
    s = np.sort(neg_scores)
    return float(s[int(np.ceil((1 - fpr) * len(s))) - 1])

def rates(d, thr):
    h, a = d[d.y_true == 0].score, d[d.y_true == 1].score
    return (h > thr).mean(), (a > thr).mean()      # false-positive rate, recall (AI detected)

rows = []
for m in DEFAULT:
    val = pr[(pr.model_name == m) & (pr.split == "val")]
    test = pr[(pr.model_name == m) & (pr.split == "test")]
    neg = val[val.y_true == 0].score.values
    options = [("default", DEFAULT[m])] + [(n, cutoff(neg, f)) for n, f in FPR_LEVELS]
    for name, thr in options:
        vf, vr = rates(val, thr); tf, trc = rates(test, thr)
        rows.append({"model": m, "rule": name, "threshold": round(thr, 4),
                     "val_FPR": round(vf, 4), "val_recall": round(vr, 4),
                     "test_FPR": round(tf, 4), "test_recall": round(trc, 4)})
T = pd.DataFrame(rows)
T.to_csv(PRED / "threshold_table.csv", index=False)
print("\n=== Threshold table (FPR = human text wrongly flagged; recall = AI text caught) ===")
print(T.to_string(index=False))

print("\n=== Decision bands per model (score -> label) ===")
for m in DEFAULT:
    strict = float(T[(T.model == m) & (T.rule.str.startswith("1% FPR"))].threshold.iloc[0])
    d = DEFAULT[m]
    if strict > d:
        print(f"{m:14s} score <= {d:.4f}: no strong evidence | {d:.4f} < score <= {strict:.4f}: possible AI (uncertain) "
              f"| score > {strict:.4f}: likely AI")
    else:   # strict cut-off at or below the default one: the uncertain band is empty
        print(f"{m:14s} score <= {strict:.4f}: no strong evidence | score > {strict:.4f}: likely AI (no uncertain band)")

print("""
=== Overall verdict rule (predict.py / app.py) ===
  Likely AI-generated  : at least 2 of 3 models above their 1% FPR threshold
  Uncertain            : exactly 1 model above its 1% FPR threshold, OR at least 2 above their default threshold
  No strong evidence   : everything else (this does NOT prove the text is human-written)""")