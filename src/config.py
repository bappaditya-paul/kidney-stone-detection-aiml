"""Centralized configuration for Kidney Stone Detection ML Pipeline."""

from pathlib import Path

# ==========================================
# Paths Configuration
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset paths (stored locally, gitignored)
RAW_DATA_DIR = BASE_DIR / "raw_dataset"
CLEAN_DATA_DIR = BASE_DIR / "raw_dataset_clean"
DATA_DIR = BASE_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

# Artifacts and Logs
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
BEST_MODEL_PATH = MODEL_DIR / "best_model.h5"
FINAL_MODEL_PATH = MODEL_DIR / "final_model.h5"
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# Model Hyperparameters
# ==========================================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-4
SEED = 42

# Train/Val/Test split ratios
TEST_SPLIT = 0.2
VAL_SPLIT = 0.2

# ==========================================
# Classification Labels & Thresholds
# ==========================================
CLASS_NAMES = ['no_stone', 'stone']
DISPLAY_CLASS_NAMES = ['No Stone', 'Stone']
THRESHOLD = 0.5  # Sigmoid decision boundary

# Valid image extensions (case-insensitive)
VALID_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')

# ==========================================
# Ultrasound Modality Guard Thresholds
# ==========================================
# B-mode medical ultrasound scans are monochromatic acoustic images
MAX_MEAN_SATURATION = 0.18
MAX_HIGH_SAT_RATIO = 0.12
MAX_CHANNEL_DIVERGENCE = 16.0
MIN_CONTRAST_STD = 15.0
MIN_SPECKLE_VARIANCE = 8.0
MAX_MEAN_INTENSITY = 235.0
MIN_MEAN_INTENSITY = 8.0
