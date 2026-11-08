# AI Text Detection

A machine-learning framework that classifies English academic text as
**human-written** or **AI-generated**, with a focus on keeping false alarms low
and testing honestly on a generator the models never saw in training.

## Data
SemEval-2024 Task 8, Subtask A (English, monolingual): about 124k texts from
Wikipedia, WikiHow, Reddit, arXiv and PeerRead.
- Train generators: davinci-003, ChatGPT, Cohere, Dolly-v2
- Test generator (unseen): BLOOMz
- 85/15 train/validation split; the BLOOMz texts are the test set

## Models
| Model | Input |
|---|---|
| TF-IDF + Logistic Regression | Word 1-2-grams (300,000 terms) |
| TF-IDF + Linear SVM | Word 1-2-grams (300,000 terms) |
| Random Forest | 32 style features (lexical, syntactic, statistical, semantic) |
| Fine-tuned DistilBERT | Token sequences (up to two 510-token windows) |

An extra experiment (`20_hybrid_lr.py`) combines TF-IDF with the 32 style features.

## Strict threshold
Each model has a strict cut-off set at the 99th percentile of human validation
scores, so only about 1% of human texts are flagged. It is frozen and applied to
the test set and the app.

## Results (unseen generator, BLOOMz)
| Model | ROC-AUC | AI caught (strict) |
|---|---|---|
| LogReg | 0.788 | 21.9% |
| SVM | 0.784 | 22.2% |
| Random Forest | 0.816 | 1.8% |
| Hybrid | 0.844 | 24.8% |
| DistilBERT | 0.890 | 37.5% |

Validation accuracy is 94-96%. On the unseen generator, accuracy falls to 58-77%
at the default threshold, so the tool is a screening aid, not proof of authorship.

## App
Streamlit app with `.txt`, `.pdf` and `.docx` upload. It shows two separate
verdicts (traditional models and DistilBERT), per-model scores against both
cut-offs, and a block-level view of which parts look AI.