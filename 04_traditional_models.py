"""STEP 4 - Traditional ML.
  A) TF-IDF + Logistic Regression
  B) TF-IDF + Linear SVM
  C) Random Forest on numeric writing features
Saves: predictions/scores for every model & split, fitted models, metrics, ROC + confusion plots,
and a feature-importance plot for the RF.
Emphasis: false-positive rate (human text flagged as AI). A threshold is tuned on VAL so that
FPR <= TARGET_FPR, then applied unchanged to TEST.
"""
import numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, roc_auc_score,
                             confusion_matrix, roc_curve)
from config import PROC, PRED, MODELS, FIG, SEED, TARGET_FPR

def tune_threshold(y, s, target_fpr=TARGET_FPR):
    """Lowest threshold whose FPR on humans (y==0) is <= target."""
    neg = np.sort(s[y == 0])
    return float(neg[int(np.ceil((1 - target_fpr) * len(neg))) - 1]) if len(neg) else 0.5

def metrics(y, s, thr):
    p = (s > thr).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return dict(acc=accuracy_score(y, p), precision=pr, recall=rc, f1=f1,
                auc=roc_auc_score(y, s), fpr=fp / max(fp + tn, 1), fnr=fn / max(fn + tp, 1))

if __name__ == "__main__":
    df = pd.read_csv(PROC / "semeval_clean.csv")
    ft = pd.read_csv(PROC / "semeval_features.csv")
    assert (df["id"].values == ft["id"].values).all()
    sp = {k: (df["split"] == k).values for k in ("train", "val", "test")}
    y = df["label"].values
    num_cols = [c for c in ft.columns if c.startswith(("lex_", "syn_", "stat_", "sem_"))]

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=300_000, sublinear_tf=True)
    vec.fit(df.loc[sp["train"], "text_tfidf"])        # fit vocabulary on TRAIN only (no leakage)
    Xt = vec.transform(df["text_tfidf"])
    Xn = ft[num_cols].values

    models = {
        "tfidf_logreg": (LogisticRegression(C=4, max_iter=2000, class_weight="balanced"), Xt, "proba"),
        "tfidf_linsvm": (LinearSVC(C=0.5, class_weight="balanced"), Xt, "decision"),
        "rf_features": (RandomForestClassifier(n_estimators=400, n_jobs=-1, min_samples_leaf=2,
                                               class_weight="balanced", random_state=SEED), Xn, "proba"),
    }
    all_pred, rows = [], []
    fig_roc, ax_roc = plt.subplots(figsize=(5.5, 5))
    for name, (clf, X, kind) in models.items():
        clf.fit(X[sp["train"]], y[sp["train"]])
        score = clf.predict_proba(X)[:, 1] if kind == "proba" else clf.decision_function(X)
        thr_default = 0.5 if kind == "proba" else 0.0
        thr_fpr = tune_threshold(y[sp["val"]], score[sp["val"]])
        for split in ("train", "val", "test"):
            m = sp[split]
            all_pred.append(pd.DataFrame({
                "id": df.loc[m, "id"], "split": split, "model_name": name, "y_true": y[m],
                "score": score[m], "pred_default": (score[m] > thr_default).astype(int),
                "pred_lowFPR": (score[m] > thr_fpr).astype(int), "source": df.loc[m, "source"]}))
            if split != "train":
                rows.append({"model": name, "split": split, "threshold": "default",
                             **metrics(y[m], score[m], thr_default)})
                rows.append({"model": name, "split": split, "threshold": f"FPR<={TARGET_FPR}",
                             **metrics(y[m], score[m], thr_fpr)})
        f, t, _ = roc_curve(y[sp["test"]], score[sp["test"]]); ax_roc.plot(f, t, label=name)
        joblib.dump(clf, MODELS / f"{name}.joblib")
        if name == "rf_features":
            imp = pd.Series(clf.feature_importances_, num_cols).sort_values()
            imp.tail(20).plot.barh(figsize=(6, 6), title="RF feature importance (top 20)")
            plt.tight_layout(); plt.savefig(FIG / "rf_importance.png", dpi=150); plt.close()
    joblib.dump(vec, MODELS / "tfidf_vectorizer.joblib")
    joblib.dump(num_cols, MODELS / "numeric_feature_names.joblib")
    ax_roc.plot([0, 1], [0, 1], "k--"); ax_roc.set_xlabel("FPR"); ax_roc.set_ylabel("TPR")
    ax_roc.set_title("ROC - test"); ax_roc.legend(); fig_roc.tight_layout()
    fig_roc.savefig(FIG / "roc_test.png", dpi=150)

    pd.concat(all_pred).to_csv(PRED / "traditional_predictions.csv", index=False)
    res = pd.DataFrame(rows); res.to_csv(PRED / "traditional_metrics.csv", index=False)
    pd.set_option("display.width", 200); print(res.round(4).to_string(index=False))
    # per-source breakdown on test (where does each model fail? arxiv/peerread = academic)
    pp = pd.concat(all_pred); pp = pp[pp.split == "test"]
    print(pp.groupby(["model_name", "source"]).apply(
        lambda g: pd.Series({"acc": (g.pred_default == g.y_true).mean(),
                             "FPR": ((g.pred_default == 1) & (g.y_true == 0)).sum() / max((g.y_true == 0).sum(), 1)})
    ).round(3))
