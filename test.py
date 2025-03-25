import sys
from pathlib import Path
from src.config.path_config import MODEL_DIR

import cloudpickle
with open(MODEL_DIR / "base_models.pkl", "rb") as f:
    base_models = cloudpickle.load(f)

with open(MODEL_DIR / "meta_model.pkl", "rb") as f:
    meta_model = cloudpickle.load(f)