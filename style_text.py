"""Text transformations for the word-based (TF-IDF) models. Used by 17_style_ngrams.py and predict.py.

mode "word"   : lower-cased text            (the current setup; word 1-2-grams)
mode "char"   : first 1,500 characters      (character n-grams are built by the vectoriser)
mode "masked" : function words and punctuation are kept, every other word becomes a placeholder
                (<Cap>, <num> or a word-length class) -> the model sees HOW the text is written, not WHAT it is about
mode "pos"    : the text becomes its sequence of part-of-speech tags (NOUN VERB ADP ...)
"""
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

CHAR_LIMIT = 1500
POS_LIMIT = 1500
_TOK = re.compile(r"[A-Za-z]+|\d+|[^\sA-Za-z\d]")
FUNC = set(ENGLISH_STOP_WORDS)          # 318 English function words


def mask_text(t):
    out = []
    for w in _TOK.findall(t):
        lw = w.lower()
        if lw in FUNC or not w[0].isalnum():
            out.append(lw)                                   # function word or punctuation: keep
        elif w.isdigit():
            out.append("<num>")
        elif w[0].isupper():
            out.append("<Cap>")
        else:
            out.append(f"<w{min(len(w) // 3, 4)}>")          # content word -> length class
    return " ".join(out)


def pos_texts(texts, nlp=None, n_process=1):
    if nlp is None:
        import spacy
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])
    docs = nlp.pipe([t[:POS_LIMIT] for t in texts], batch_size=256, n_process=n_process)
    return [" ".join(tok.pos_ for tok in d) for d in docs]


def prepare(mode, texts, nlp=None, n_process=1):
    if mode == "word":
        return [t.lower() for t in texts]
    if mode == "char":
        return [t[:CHAR_LIMIT] for t in texts]
    if mode == "masked":
        return [mask_text(t) for t in texts]
    if mode == "pos":
        return pos_texts(texts, nlp, n_process)
    raise ValueError(f"unknown mode {mode}")
