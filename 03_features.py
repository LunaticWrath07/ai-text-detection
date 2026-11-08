"""STEP 3 - Feature extraction (>=1 family each: Lexical, Syntactic, Statistical, Semantic).


"""
import sys, re, math
from collections import Counter
import numpy as np, pandas as pd, torch, spacy
from tqdm import tqdm
from config import PROC, MAX_CHARS_FEATURES, PPL_MAX_TOKENS

dev = "cuda" if torch.cuda.is_available() else "cpu"
nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])


def entropy(counts):
    tot = sum(counts.values())
    return -sum(c / tot * math.log2(c / tot) for c in counts.values()) if tot else 0.0


def lexical_statistical(t):
    words = re.findall(r"[A-Za-z']+", t.lower())
    n = max(len(words), 1)
    wc = Counter(words)
    first100 = words[:100]
    sents = [s for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]
    slen = np.array([len(s.split()) for s in sents]) if sents else np.array([0])
    bigr = Counter(zip(words, words[1:]))
    return {
        # --- Lexical ---
        "lex_n_words": n,
        "lex_avg_word_len": np.mean([len(w) for w in words]) if words else 0,
        "lex_ttr_100": len(set(first100)) / max(len(first100), 1),   # length-robust TTR
        "lex_hapax_ratio": sum(1 for c in wc.values() if c == 1) / n,
        "lex_long_word_ratio": sum(len(w) >= 7 for w in words) / n,
        # --- Statistical ---
        "stat_char_entropy": entropy(Counter(t)),
        "stat_word_entropy": entropy(wc),
        "stat_sent_len_mean": slen.mean(),
        "stat_sent_len_std": slen.std(),
        "stat_burstiness": slen.std() / (slen.mean() + 1e-9),        # humans tend to vary more
        "stat_repeat_bigram_ratio": sum(c > 1 for c in bigr.values()) / max(len(bigr), 1),
        "stat_upper_ratio": sum(ch.isupper() for ch in t) / max(len(t), 1),
        "stat_digit_ratio": sum(ch.isdigit() for ch in t) / max(len(t), 1),
        "stat_comma_per_word": t.count(",") / n,
        "stat_semicolon_per_word": t.count(";") / n,
        "stat_punct_ratio": sum(ch in ".,;:!?-()\"'" for ch in t) / max(len(t), 1),
    }


def depth(tok):
    """Distance from token to its sentence root (compare indexes: spaCy makes new Token objects)."""
    d = 0
    while tok.head.i != tok.i and d < 100:
        tok = tok.head
        d += 1
    return d


def syntactic(docs):
    rows = []
    for doc in docs:
        n = max(len(doc), 1)
        pos = Counter(t.pos_ for t in doc)
        dd = [depth(t) for t in doc]
        rows.append({
            "syn_noun_ratio": pos["NOUN"] / n, "syn_verb_ratio": pos["VERB"] / n,
            "syn_adj_ratio": pos["ADJ"] / n, "syn_adv_ratio": pos["ADV"] / n,
            "syn_pron_ratio": pos["PRON"] / n, "syn_adp_ratio": pos["ADP"] / n,
            "syn_conj_ratio": (pos["CCONJ"] + pos["SCONJ"]) / n,
            "syn_stop_ratio": sum(t.is_stop for t in doc) / n,
            "syn_avg_dep_depth": np.mean(dd) if dd else 0,
            "syn_max_dep_depth": max(dd) if dd else 0,
        })
    return pd.DataFrame(rows)


@torch.no_grad()
def perplexity_feats(texts, bs=16):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained("distilgpt2")
    tok.pad_token = tok.eos_token
    lm = AutoModelForCausalLM.from_pretrained("distilgpt2").to(dev).eval()
    out = [None] * len(texts)
    order = np.argsort([len(t) for t in texts])
    for i in tqdm(range(0, len(texts), bs), desc="perplexity"):
        idx = order[i:i + bs]
        enc = tok([texts[j] for j in idx], return_tensors="pt", padding=True,
                  truncation=True, max_length=PPL_MAX_TOKENS).to(dev)
        logits = lm(**enc).logits[:, :-1]
        tgt = enc["input_ids"][:, 1:]
        mask = enc["attention_mask"][:, 1:].bool()
        lp = torch.log_softmax(logits.float(), -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        rank1 = (logits.argmax(-1) == tgt)
        for k, j in enumerate(idx):
            v = lp[k][mask[k]].cpu().numpy()
            r = rank1[k][mask[k]].float().mean().item() if v.size else 0
            out[j] = {"sem_perplexity": float(np.exp(-v.mean())) if v.size else 0,
                      "sem_mean_logprob": float(v.mean()) if v.size else 0,
                      "sem_std_logprob": float(v.std()) if v.size else 0,   # burstiness of surprise
                      "sem_top1_token_rate": r}
    return pd.DataFrame(out)


@torch.no_grad()
def coherence_feats(texts):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("all-MiniLM-L6-v2", device=dev)
    rows = []
    for t in tqdm(texts, desc="coherence"):
        sents = [s for s in re.split(r"(?<=[.!?])\s+", t) if len(s.split()) > 2][:20]
        if len(sents) < 2:
            rows.append({"sem_adj_sim_mean": 0, "sem_adj_sim_std": 0})
            continue
        e = m.encode(sents, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
        s = (e[:-1] * e[1:]).sum(1)
        rows.append({"sem_adj_sim_mean": float(s.mean()), "sem_adj_sim_std": float(s.std())})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "semeval"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None

    df = pd.read_csv(PROC / f"{name}_clean.csv")
    if limit:
        df = df.sample(min(limit, len(df)), random_state=42).reset_index(drop=True)
    print(f"device: {dev} | rows: {len(df)}")

    texts = df["text"].str.slice(0, MAX_CHARS_FEATURES).tolist()
    ls = pd.DataFrame([lexical_statistical(t) for t in tqdm(texts, desc="lexical/statistical")])
    sy = syntactic(nlp.pipe(tqdm(texts, desc="spaCy"), batch_size=64, n_process=1))
    pp = perplexity_feats(texts)
    co = coherence_feats(texts)
    feats = pd.concat([df[["id", "split", "label", "source", "model"]].reset_index(drop=True),
                       ls, sy, pp, co], axis=1)
    feats = feats.replace([np.inf, -np.inf], np.nan).fillna(0)

    out = PROC / (f"{name}_features_smoke.csv" if limit else f"{name}_features.csv")
    feats.to_csv(out, index=False)
    print("saved", out, feats.shape)
    cols = [c for c in feats.columns if c.startswith(("lex_", "syn_", "stat_", "sem_"))]
    print(len(cols), "features")
    pd.set_option("display.width", 160)
    print(feats.groupby("label")[cols].mean().T.round(3))   # quick human(0) vs AI(1) comparison