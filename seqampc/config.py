import os
from pathlib import Path

_DATA_ROOT = Path(os.environ.get("SEQAMPC_DATA_ROOT", "./data"))

DATASETS_DIR = _DATA_ROOT / "archive"
MODELS_DIR = _DATA_ROOT / "models"
LEARNING_CURVES_DIR = MODELS_DIR / "learning_curves"
