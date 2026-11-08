"""STEP 9 - Feature selection for the Random Forest (and an optional TF-IDF vocabulary sweep).

  py 09_feature_selection.py          # Random Forest features
  py 09_feature_selection.py tfidf    # TF-IDF vocabulary size sweep

Design: features are judged by how well they GENERALISE TO AN UNSEEN GENERATOR, not by a random split:
  1. correlation filter           - of two near-duplicate features (|r| > 0.95) keep the more informative one
  2. leave-one-generator-out CV   - train without one AI generator, test on it (+ held-out humans)
  3. permutation importance       - measured on that unseen generator, averaged over folds
  4. candidate sets               - all / corr-filtered / no length / top-k; pick the SMALLEST set whose LOGO AUC is
                                    within TOL of the best. The test split is only REPORTED, never used to choose.
Outputs: outputs/models/selected_features.json, rf_features_selected.joblib, outputs/predictions/feature_selection.csv,
         outputs/figures/feature_selection_permutation.png
"""
import sys, json, time
import numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from config import PROC, PRED, MODELS, FIG, SEED, TARGET_FPR

SEL_N, TOL, PERM_ROWS = 40000, 0.005, 4000
LM_FEATS = {"sem_perplexity", "sem_mean_logprob", "sem_std_logprob", "sem_top1_token_rate"}
pd.set_option("display.width", 220)


def rf(n=100, leaf=3):
    return RandomForestClassifier(n_estimators=n, min_samples_leaf=leaf, n_jobs=-1,
                                  class_weight="balanced", random_state=SEED)


def needs(cols):
    out = []
    if any(c.startswith("syn_") for c in cols): out.append("spaCy")
    if any(c in LM_FEATS for c in cols): out.append("distilgpt2")
    if any(c.startswith("sem_adj_") for c in cols): out.append("MiniLM")
    return "+".join(out) if out else "none (regex only)"


def logo_folds(pool, rng):
    gens = sorted(pool.loc[pool.label == 1, "model"].unique())
    hum = pool.index[pool.label == 0].values
    for g in gens:
        hold = set(rng.choice(hum, int(0.2 * len(hum)), replace=False)) | set(pool.index[(pool.label == 1) & (pool.model == g)])
        mask = pool.index.isin(hold)
        yield g, np.where(~mask)[0], np.where(mask)[0]


def logo_auc(pool, cols, folds):
    s = []
    for g, tr, ho in folds:
        m = rf().fit(pool.iloc[tr][cols], pool.iloc[tr]["label"])
        s.append(roc_auc_score(pool.iloc[ho]["label"], m.predict_proba(pool.iloc[ho][cols])[:, 1]))
    return float(np.mean(s)), float(np.std(s))


def run_rf():
    ft = pd.read_csv(PROC / "semeval_features.csv")
    num = [c for c in ft.columns if c.startswith(("lex_", "syn_", "stat_", "sem_"))]
    trn, val, tst = (ft[ft.split == s] for s in ("train", "val", "test"))
    pool = ft[ft.split.isin(["train", "val"])]
    pool = pool.groupby("model", group_keys=False).sample(frac=min(1, SEL_N / len(pool)), random_state=SEED).reset_index(drop=True)
    print(f"{len(num)} candidate features | selection pool {len(pool)} rows | generators: "
          f"{sorted(pool.loc[pool.label == 1, 'model'].unique())}")

    # 1) correlation filter
    single = {c: abs(roc_auc_score(pool.label, pool[c]) - 0.5) for c in num}
    corr, drop = pool[num].corr().abs(), set()
    for i, a in enumerate(num):
        for b in num[i + 1:]:
            if a not in drop and b not in drop and corr.loc[a, b] > 0.95:
                drop.add(a if single[a] < single[b] else b)
    keep = [c for c in num if c not in drop]
    print(f"correlation filter dropped {len(drop)}: {sorted(drop)}")

    # 2) permutation importance on each unseen generator
    rng = np.random.RandomState(SEED)
    folds = list(logo_folds(pool, rng))
    imp = []
    for g, tr, ho in folds:
        m = rf().fit(pool.iloc[tr][keep], pool.iloc[tr]["label"])
        h = pool.iloc[ho].sample(min(PERM_ROWS, len(ho)), random_state=SEED)
        p = permutation_importance(m, h[keep], h["label"], scoring="roc_auc", n_repeats=3, random_state=SEED, n_jobs=-1)
        imp.append(p.importances_mean)
        print(f"  permutation importance on held-out generator '{g}' done")
    imp = pd.DataFrame(imp, columns=keep)
    ranked = imp.mean().sort_values(ascending=False)
    ranked.iloc[::-1].plot.barh(xerr=imp.std()[ranked.index[::-1]], figsize=(7, 8),
                                title="Permutation importance on an unseen generator (drop in AUC)")
    plt.tight_layout(); plt.savefig(FIG / "feature_selection_permutation.png", dpi=150); plt.close()
    print("\nRanked features (mean AUC drop when shuffled; <=0 means useless or harmful on unseen generators):")
    print(ranked.round(4).to_string())

    # 3) candidate sets
    sets = {f"all ({len(num)})": num, f"corr-filtered ({len(keep)})": keep,
            "all minus lex_n_words": [c for c in num if c != "lex_n_words"]}
    for k in (6, 8, 10, 12, 16, 20):
        if k < len(keep):
            sets[f"top-{k}"] = list(ranked.index[:k])
    rows = []
    for name, cols in sets.items():
        lm, ls = logo_auc(pool, cols, folds)
        t0 = time.time(); m = rf().fit(trn[cols], trn.label); fit_s = time.time() - t0
        rows.append({"set": name, "n": len(cols), "LOGO_auc": round(lm, 4), "LOGO_sd": round(ls, 4),
                     "val_auc": round(roc_auc_score(val.label, m.predict_proba(val[cols])[:, 1]), 4),
                     "test_auc(report only)": round(roc_auc_score(tst.label, m.predict_proba(tst[cols])[:, 1]), 4),
                     "fit_s": round(fit_s, 1), "extraction needs": needs(cols)})
        print(f"  evaluated {name}")
    R = pd.DataFrame(rows); R.to_csv(PRED / "feature_selection.csv", index=False)
    print("\n", R.to_string(index=False))

    # 4) choose: smallest set within TOL of the best LOGO AUC (chosen on LOGO only, never on test)
    best = R.LOGO_auc.max()
    pick = R[R.LOGO_auc >= best - TOL].sort_values(["n", "LOGO_auc"], ascending=[True, False]).iloc[0]
    cols = sets[pick["set"]]
    print(f"\nSELECTED: {pick['set']}  ({pick['n']} features, LOGO AUC {pick.LOGO_auc} vs best {best})")
    print("features:", cols, "\nextraction needs:", pick["extraction needs"])

    # 5) tune forest complexity (bigger min_samples_leaf = simpler trees) on the same unseen-generator folds
    print("\nForest complexity (LOGO AUC on the selected features):")
    leaf_res = []
    for leaf in (2, 10, 30, 60):
        sc = []
        for g, tr_i, ho in folds:
            m = rf(100, leaf).fit(pool.iloc[tr_i][cols], pool.iloc[tr_i]["label"])
            sc.append(roc_auc_score(pool.iloc[ho]["label"], m.predict_proba(pool.iloc[ho][cols])[:, 1]))
        leaf_res.append((leaf, float(np.mean(sc))))
        print(f"  min_samples_leaf={leaf:3d}  LOGO AUC {leaf_res[-1][1]:.4f}")
    best_auc = max(a for _, a in leaf_res)
    leaf = max(l for l, a in leaf_res if a >= best_auc - TOL)       # simplest forest within TOL of the best
    print(f"chosen min_samples_leaf = {leaf}")

    final = rf(400, leaf).fit(trn[cols], trn.label)
    sv = final.predict_proba(val[cols])[:, 1]
    neg = np.sort(sv[val.label.values == 0])
    strict = float(neg[int(np.ceil((1 - TARGET_FPR) * len(neg))) - 1])    # <= TARGET_FPR false positives on val humans
    st_ = final.predict_proba(tst[cols])[:, 1]
    print(f"strict threshold (val, {TARGET_FPR:.0%} FPR) = {strict:.4f} | test AUC {roc_auc_score(tst.label, st_):.4f} | "
          f"test FPR {(st_[tst.label.values == 0] > strict).mean():.4f} | test recall {(st_[tst.label.values == 1] > strict).mean():.4f}")

    joblib.dump(final, MODELS / "rf_features_selected.joblib"); joblib.dump(cols, MODELS / "selected_feature_names.joblib")
    json.dump({"set": pick["set"], "features": cols, "min_samples_leaf": leaf, "strict_threshold": strict,
               "rule": f"smallest set within {TOL} of best LOGO AUC",
               "extraction_needs": pick["extraction needs"]}, open(MODELS / "selected_features.json", "w"), indent=2)
    out = pd.concat([pd.DataFrame({"id": d["id"].values, "split": n, "model_name": "rf_selected",
                                   "y_true": d["label"].values, "score": p, "source": d["source"].values})
                     for n, d, p in (("val", val, sv), ("test", tst, st_))])
    out.to_csv(PRED / "rf_selected_predictions.csv", index=False)
    print("saved rf_features_selected.joblib, selected_feature_names.joblib, selected_features.json, rf_selected_predictions.csv")


def run_tfidf():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_selection import chi2
    from sklearn.linear_model import LogisticRegression
    df = pd.read_csv(PROC / "semeval_clean.csv", usecols=["split", "label", "text_tfidf"])
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=300_000, sublinear_tf=True).fit(tr.text_tfidf)
    Xtr, Xva, Xte = (vec.transform(d.text_tfidf) for d in (tr, va, te))
    order = np.argsort(-np.nan_to_num(chi2(Xtr, tr.label)[0]))     # chi2 on TRAIN only
    rows = []
    for k in (5_000, 20_000, 50_000, 100_000, Xtr.shape[1]):
        k = min(k, Xtr.shape[1]); c = order[:k]
        t0 = time.time(); m = LogisticRegression(C=4, max_iter=2000, class_weight="balanced").fit(Xtr[:, c], tr.label)
        rows.append({"vocab": k, "fit_s": round(time.time() - t0, 1),
                     "val_auc": round(roc_auc_score(va.label, m.decision_function(Xva[:, c])), 4),
                     "test_auc(report only)": round(roc_auc_score(te.label, m.decision_function(Xte[:, c])), 4)})
    R = pd.DataFrame(rows).drop_duplicates("vocab"); R.to_csv(PRED / "tfidf_vocab_sweep.csv", index=False)
    print(R.to_string(index=False))


if __name__ == "__main__":
    run_tfidf() if len(sys.argv) > 1 and sys.argv[1] == "tfidf" else run_rf()
