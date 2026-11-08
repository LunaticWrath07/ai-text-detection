"""STEP 6 - Fine-tune DistilBERT on the SAME train/val/test splits as Step 4 and compare.

Defaults are sized for an 8 GB laptop GPU (bf16 autocast, max_len 256, batch 16).
First run: keep TRAIN_N = 30000 to get a result in a reasonable time; set None for all rows.

Outputs:
  outputs/models/distilbert/                      fine-tuned weights + tokenizer
  outputs/predictions/distilbert_predictions.csv  same columns as traditional_predictions.csv
  outputs/predictions/distilbert_metrics.csv      val/test metrics, default and 1% FPR thresholds
"""
import importlib, time
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import PROC, PRED, MODELS, SEED, TARGET_FPR

trad = importlib.import_module("04_traditional_models")   # reuse tune_threshold() and metrics()

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN, BS, EPOCHS, LR = 512, 8, 3, 2e-5
TRAIN_N = None          # None = use every training row
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED); np.random.seed(SEED)

tok = AutoTokenizer.from_pretrained(MODEL_NAME)

def make_loader(texts, labels, shuffle, bs):
    items = list(zip(texts, labels))
    def collate(batch):
        t, y = zip(*batch)
        enc = tok(list(t), truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor(y)
        return enc
    return DataLoader(items, batch_size=bs, shuffle=shuffle, collate_fn=collate, num_workers=0)

@torch.no_grad()
def predict_scores(model, texts):
    model.eval()
    order = np.argsort([len(t) for t in texts])               # length-sorted = less padding, faster
    loader = make_loader([texts[i] for i in order], [0] * len(texts), False, 64)
    out = []
    for b in loader:
        b = {k: v.to(dev) for k, v in b.items() if k != "labels"}
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            logits = model(**b).logits
        out.append(torch.softmax(logits.float(), -1)[:, 1].cpu().numpy())
    scores = np.empty(len(texts)); scores[order] = np.concatenate(out)
    return scores

if __name__ == "__main__":
    df = pd.read_csv(PROC / "semeval_clean.csv")
    tr = df[df.split == "train"]
    if TRAIN_N: tr = tr.sample(min(TRAIN_N, len(tr)), random_state=SEED)
    print(f"device: {dev} | train rows: {len(tr)} | epochs: {EPOCHS}")

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(dev)
    loader = make_loader(tr["text"].tolist(), tr["label"].tolist(), True, BS)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total = EPOCHS * len(loader); warm = int(0.06 * total)
    sched = LambdaLR(opt, lambda s: s / max(warm, 1) if s < warm else max(0.0, (total - s) / max(total - warm, 1)))

    step, t0 = 0, time.time()
    for ep in range(EPOCHS):
        model.train(); run = 0.0
        for i, b in enumerate(loader):
            b = {k: v.to(dev) for k, v in b.items()}
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                loss = model(**b).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            run += loss.item(); step += 1
            if (i + 1) % 100 == 0:
                print(f"epoch {ep+1}/{EPOCHS} step {i+1}/{len(loader)} loss {run/100:.4f} "
                      f"| {(time.time()-t0)/60:.1f} min elapsed", flush=True)
                run = 0.0

    model.save_pretrained(MODELS / "distilbert"); tok.save_pretrained(MODELS / "distilbert")

    preds, rows = [], []
    sc = {}
    for split in ("val", "test"):
        d = df[df.split == split]
        sc[split] = predict_scores(model, d["text"].tolist())
        preds.append(pd.DataFrame({"id": d["id"].values, "split": split, "model_name": "distilbert",
                                   "y_true": d["label"].values, "score": sc[split], "source": d["source"].values}))
    y_val = df.loc[df.split == "val", "label"].values
    thr_fpr = trad.tune_threshold(y_val, sc["val"], TARGET_FPR)
    for split in ("val", "test"):
        y = df.loc[df.split == split, "label"].values
        rows.append({"model": "distilbert", "split": split, "threshold": "default", **trad.metrics(y, sc[split], 0.5)})
        rows.append({"model": "distilbert", "split": split, "threshold": f"FPR<={TARGET_FPR}",
                     **trad.metrics(y, sc[split], thr_fpr)})
    P = pd.concat(preds)
    P["pred_default"] = (P.score > 0.5).astype(int); P["pred_lowFPR"] = (P.score > thr_fpr).astype(int)
    P.to_csv(PRED / "distilbert_predictions.csv", index=False)
    res = pd.DataFrame(rows); res.to_csv(PRED / "distilbert_metrics.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n", res.round(4).to_string(index=False))
    try:
        print("\nTraditional models for comparison:\n",
              pd.read_csv(PRED / "traditional_metrics.csv").round(4).to_string(index=False))
    except FileNotFoundError:
        pass
