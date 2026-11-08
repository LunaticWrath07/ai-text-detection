from pathlib import Path

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
FIG = ROOT / "outputs" / "figures"
PRED = ROOT / "outputs" / "predictions"
MODELS = ROOT / "outputs" / "models"
for p in (RAW, PROC, FIG, PRED, MODELS):
    p.mkdir(parents=True, exist_ok=True)

SEED = 42
MIN_WORDS = 20            # drop texts shorter than this
SAMPLE_N = None           # e.g. 20000 to prototype quickly; None = use everything
MAX_CHARS_FEATURES = 2000 # truncate text for spaCy / perplexity (speed)
PPL_MAX_TOKENS = 256
TARGET_FPR = 0.01         # false-positive budget used for threshold tuning
