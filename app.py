"""AI-generated text detector - web app.
Run:  py -m streamlit run app.py     (keep app.py, app_helpers.py and predict.py in the same folder)
Accepts .txt / .pdf / .docx uploads or pasted text and runs the three trained traditional models.
"""
import io
import pandas as pd
import streamlit as st

import sys
try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    if get_script_run_ctx() is None:
        sys.exit("Run this app with:  py -m streamlit run app.py")
except ImportError:
    pass

st.set_page_config(page_title="AI text detector", page_icon="🔍", layout="wide")


@st.cache_resource(show_spinner="Loading trained models (first time only)...")
def load():
    import predict
    import app_helpers as H
    return predict, H


def read_upload(f):
    name, data = f.name.lower(), f.getvalue()
    if name.endswith(".txt"):
        return data.decode("utf-8", errors="ignore")
    if name.endswith(".pdf"):
        from pypdf import PdfReader
        return "\n\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(data)).pages)
    if name.endswith(".docx"):
        from docx import Document
        return "\n\n".join(p.text for p in Document(io.BytesIO(data)).paragraphs)
    raise ValueError("Unsupported file type")


# ------------------------------------------------------------------ header
st.title("🔍 AI-generated text detector")
st.caption("Trained models vote: TF-IDF + Logistic Regression, TF-IDF + Linear SVM, a Random Forest on "
           "writing-style features and a fine-tuned DistilBERT (when installed). Built to keep false accusations low.")

try:
    P, H = load()
except FileNotFoundError as e:
    st.error(f"A trained model or prediction file is missing: {e}. Run Step 4 (04_traditional_models.py) first.")
    st.stop()

left, right = st.columns([1, 1])
with left:
    up = st.file_uploader("Upload an essay", type=["txt", "pdf", "docx"])
with right:
    pasted = st.text_area("...or paste text here", height=130)
go = st.button("Analyze", type="primary")

if go:
    try:
        raw = read_upload(up) if up else pasted
    except Exception as e:
        st.error(f"Could not read the file: {e}")
        st.stop()
    if not raw or not raw.strip():
        st.warning("Please upload a file or paste some text.")
        st.stop()
    with st.spinner("Analyzing... (the first analysis takes longer)"):
        n_words, scores, row = P.score_text(raw)
        chunks = H.chunk_table(P, raw)
    st.session_state["res"] = dict(raw=raw, n_words=n_words, scores=scores, row=row, chunks=chunks)

res = st.session_state.get("res")
if not res:
    st.info("Upload a file or paste text, then press **Analyze**.")
    st.stop()

raw, n_words, scores, row, chunks = res["raw"], res["n_words"], res["scores"], res["row"], res["chunks"]
V = P.verdicts(scores)

# ------------------------------------------------------------------ verdicts (traditional models and DistilBERT are reported separately)
st.divider()
active = [(name, v) for name, v in V.items() if v]
cols = st.columns(len(active))
for col, (name, v) in zip(cols, active):
    with col:
        detail = f"{v['strong']} of {v['n']} model{'s' if v['n'] > 1 else ''} above the strict (1% false-positive) threshold."
        if v["level"] == "LIKELY":
            st.error(f"### 🔴 {name}: likely AI-generated\n{detail}")
        elif v["level"] == "UNCERTAIN":
            st.warning(f"### 🟡 {name}: uncertain\nMixed signals. {detail}")
        else:
            st.success(f"### 🟢 {name}: no strong evidence of AI\nThis does **not** prove the text is human-written.")
if len(active) == 2 and active[0][1]["level"] != active[1][1]["level"]:
    st.info("The two groups disagree. DistilBERT generalised better to an unseen generator in testing, but neither is proof.")
if n_words < 100:
    st.warning(f"Only {n_words} words. Scores are unreliable below about 100 words.")

n_strong = sum(scores[n] > P.LOW_FPR_THR[n] for n in scores)
n_models = len(scores)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Words", n_words, help=f"Random Forest uses {P.RF_DESC}; DistilBERT {'on' if P.HAS_DISTILBERT else 'off'}")
c2.metric("Models flagging (strict)", f"{n_strong} / {n_models}")
c3.metric("Perplexity", f"{row.get('sem_perplexity', float('nan')):.1f}", help="Lower = more predictable text (AI text tends to be lower)")
c4.metric("Burstiness", f"{row.get('stat_burstiness', float('nan')):.2f}", help="Variation in sentence length (human text tends to be higher)")

tab1, tab2 = st.tabs(["📊 Model scores", "🧩 Which parts look AI?"])

# ------------------------------------------------------------------ tab 1
with tab1:
    st.markdown("Black line = this text's score. **Green** = below the default cut-off, **yellow** = above default but "
                "below the strict cut-off, **red** = above the strict cut-off (dashed line, tuned to flag at most 1% of human text).")
    import matplotlib.pyplot as plt
    for n, s in scores.items():
        fig = H.gauge_fig(n, s, P.DEFAULT_THR[n], P.LOW_FPR_THR[n])
        st.pyplot(fig)
        plt.close(fig)
    table = pd.DataFrame([{
        "model": H.LABEL[n], "score": round(s, 3),
        "default cut-off": round(P.DEFAULT_THR[n], 3), "strict cut-off (1% FPR)": round(P.LOW_FPR_THR[n], 3),
        "decision": {"high": "AI (strong)", "mid": "AI (weak)", "low": "human-like"}[
            H.zone(s, P.DEFAULT_THR[n], P.LOW_FPR_THR[n])]} for n, s in scores.items()])
    st.dataframe(table, hide_index=True)

# ------------------------------------------------------------------ tab 2
with tab2:
    st.markdown("The text is split into ~120-word blocks and each block is scored by the two TF-IDF models "
                "(the Random Forest needs the whole text). Short blocks are greyed out. Block-level scores are noisier "
                "than the document score, so treat colours as hints, not evidence.")
    if len(chunks):
        counts = chunks["level"].value_counts()
        a, b, c = st.columns(3)
        a.metric("Strong AI signal blocks", int(counts.get("high", 0)))
        b.metric("Weak signal blocks", int(counts.get("mid", 0)))
        c.metric("No signal blocks", int(counts.get("low", 0)))
        st.markdown(H.chunks_html(chunks), unsafe_allow_html=True)
    else:
        st.info("Not enough text to split into blocks.")