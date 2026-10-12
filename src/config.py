import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("SALINAS_DATA_DIR", ROOT_DIR / "data")).expanduser().resolve()
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
MODELS_DIR = ROOT_DIR / "models"

# Search a few common locations so the project works whether the dataset sits in
# the repo's data folder or alongside the repo in a parent folder.
DATA_DIR_CANDIDATES = [
    DATA_DIR,
    ROOT_DIR / "data",
    ROOT_DIR.parent / "data",
    ROOT_DIR.parent,
]

IMAGE_PATH = None
GROUND_TRUTH_PATH = None

IMAGE_KEY = "salinas"
GROUND_TRUTH_KEY = "groundtruth"
RANDOM_SEED = 1671
N_COMPONENTS = 5
PATCH_SIZE = 25
TRAIN_FRACTION_PER_CLASS = 0.05
EPOCHS = 35
BATCH_SIZE = 16
LEARNING_RATE = 0.001
CLIP_VALUE = 0.5
DROPOUT_RATE = 0.3
NORMALIZATION_DIVISOR = 255.0  # Preserves the source notebook's preprocessing.
