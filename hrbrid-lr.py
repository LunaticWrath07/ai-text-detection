"""STEP 20 - Hybrid (feature-fusion) model: ONE Logistic Regression on TF-IDF words + the 32 style features.
Nothing is re-extracted and no existing model is touched: it reuses the saved TF-IDF vectorizer and semeval_features.csv.
Strict threshold = cut-off flagging <=1% of VAL humans (same rule as every other model). Test is only reported.
Run:  py 20_hybrid_lr.py        (a few minutes)   Output: outputs/predictions/hybrid_predictions.csv, hybrid_metrics.csv"""
import numpy as np, pandas as pd, joblib
from scipy.sparse import hstack, csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from config import PROC, PRED, MODELS, TARGET_FPR

df = pd.read_csv(PROC / "semeval_clean.csv"); ft = pd.read_csv(PROC / "semeval_features.csv")
assert (df["id"].values == ft["id"].values).all()
num_cols = joblib.load(MODELS / "numeric_feature_names.joblib")
vec = joblib.load(MODELS / "tfidf_vectorizer.joblib")
y = df["label"].values
sp = {k: (df["split"] == k).values for k in ("train", "val", "test")}

Xt = vec.transform(df["text_tfidf"])
Xn = ft[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
sc = StandardScaler().fit(Xn[sp["train"]])
Xn = np.clip(sc.transform(Xn), -5, 5) * 0.1          # small scale so the 32 dense columns do not swamp 300k sparse ones
X = hstack([Xt, csr_matrix(Xn)]).tocsr()

clf = LogisticRegression(C=4, max_iter=3000, class_weight="balanced").fit(X[sp["train"]], y[sp["train"]])
s = clf.predict_proba(X)[:, 1]
neg = np.sort(s[sp["val"] & (y == 0)]); thr = float(neg[int(np.ceil((1 - TARGET_FPR) * len(neg))) - 1])

rows, preds = [], []
for k in ("val", "test"):
    m = sp[k]
    for tag, t in (("default", 0.5), ("strict", thr)):
        p = s[m] > t
        rows.append(dict(split=k, threshold=tag, AUC=roc_auc_score(y[m], s[m]), acc=(p == y[m]).mean(),
                         recall=p[y[m] == 1].mean(), FPR=p[y[m] == 0].mean()))
    preds.append(pd.DataFrame({"id": df.loc[m, "id"], "split": k, "model_name": "hybrid_lr", "y_true": y[m],
                               "score": s[m], "source": df.loc[m, "source"]}))
pd.concat(preds).to_csv(PRED / "hybrid_predictions.csv", index=False)
res = pd.DataFrame(rows); res.to_csv(PRED / "hybrid_metrics.csv", index=False)
print(res.round(3).to_string(index=False)); print("strict cut-off:", round(thr, 3))