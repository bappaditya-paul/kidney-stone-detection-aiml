"""ML Pipeline Configuration: Paths, Hyperparameters, and Diagnostic Thresholds."""

from pathlib import Path

# ==========================================
# Paths Configuration
# ==========================================
BASE_DIR = Path(__file__).resolve().parent

# Datasets directory (gitignored)
DATASETS_DIR = BASE_DIR / "datasets"
RAW_DATA_DIR = DATASETS_DIR / "raw_dataset"
CLEAN_DATA_DIR = DATASETS_DIR / "raw_dataset_clean"
DATA_DIR = DATASETS_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

# Fallback to root dataset directories if symlink resolution is needed
if not RAW_DATA_DIR.exists() and (BASE_DIR.parent / "raw_dataset").exists():
    RAW_DATA_DIR = BASE_DIR.parent / "raw_dataset"
if not CLEAN_DATA_DIR.exists() and (BASE_DIR.parent / "raw_dataset_clean").exists():
    CLEAN_DATA_DIR = BASE_DIR.parent / "raw_dataset_clean"
if not DATA_DIR.exists() and (BASE_DIR.parent / "data").exists():
    DATA_DIR = BASE_DIR.parent / "data"
    TRAIN_DIR = DATA_DIR / "train"
    VAL_DIR = DATA_DIR / "val"
    TEST_DIR = DATA_DIR / "test"

# Model artifacts
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
BEST_MODEL_PATH = MODELS_DIR / "best_model.h5"
FINAL_MODEL_PATH = MODELS_DIR / "final_model.h5"

# ==========================================
# Hyperparameters
# ==========================================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-4
SEED = 42

# Train / Val / Test split ratios
TEST_SPLIT = 0.2
VAL_SPLIT = 0.2

# ==========================================
# Classification Labels & Thresholds
# ==========================================
CLASS_NAMES = ['no_stone', 'stone']
DISPLAY_CLASS_NAMES = ['No Stone', 'Stone']
THRESHOLD = 0.5

# Supported image extensions
VALID_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')

# ==========================================
# Ultrasound Modality Guard Thresholds
# ==========================================
MAX_MEAN_SATURATION = 0.18
MAX_HIGH_SAT_RATIO = 0.12
MAX_CHANNEL_DIVERGENCE = 16.0
MIN_CONTRAST_STD = 15.0
MIN_SPECKLE_VARIANCE = 8.0
MAX_MEAN_INTENSITY = 235.0
MIN_MEAN_INTENSITY = 8.0
