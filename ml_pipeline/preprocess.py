"""ML Pipeline: Data Preprocessing and Train/Val/Test Partitioning."""

import os
import sys
import shutil
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Ensure ml_pipeline is on sys.path
PIPELINE_ROOT = Path(__file__).resolve().parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

import config


def setup_directories(clear_existing=True):
    """Set up directory structure for train/val/test splits."""
    print("\n📂 Setting up directory structure...")
    
    if clear_existing and config.DATA_DIR.exists():
        shutil.rmtree(config.DATA_DIR)
        print(f"  - Cleared existing split directory: {config.DATA_DIR}")
    
    directories = [
        config.TRAIN_DIR / "stone", config.TRAIN_DIR / "no_stone",
        config.VAL_DIR / "stone", config.VAL_DIR / "no_stone",
        config.TEST_DIR / "stone", config.TEST_DIR / "no_stone"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    print("✓ Directory structure ready")


def verify_source_dataset():
    """Verify that source images exist in raw or clean dataset directories."""
    print("🔍 Verifying dataset structure...")
    source_dir = config.CLEAN_DATA_DIR if config.CLEAN_DATA_DIR.exists() else config.RAW_DATA_DIR
    
    if not source_dir.exists():
        raise FileNotFoundError(f"Dataset source folder not found at: {source_dir}")
    
    stone_dir = source_dir / "stone"
    no_stone_dir = source_dir / "no_stone"
    
    if not (stone_dir.exists() and no_stone_dir.exists()):
        raise FileNotFoundError(f"Source directory {source_dir} must contain 'stone' and 'no_stone' folders.")
    
    return source_dir


def split_dataset(source_dir):
    """Partition dataset into train/val/test splits."""
    print(f"\n📊 Splitting dataset from: {source_dir}")
    
    for class_name in config.CLASS_NAMES:
        class_folder = source_dir / class_name
        all_files = [
            f for f in class_folder.iterdir()
            if f.is_file() and f.suffix.lower() in config.VALID_IMAGE_EXTENSIONS
        ]
        
        train_val, test_files = train_test_split(
            all_files, test_size=config.TEST_SPLIT, random_state=config.SEED
        )
        train_files, val_files = train_test_split(
            train_val, test_size=config.VAL_SPLIT, random_state=config.SEED
        )
        
        for f in train_files:
            shutil.copy2(f, config.TRAIN_DIR / class_name / f.name)
        for f in val_files:
            shutil.copy2(f, config.VAL_DIR / class_name / f.name)
        for f in test_files:
            shutil.copy2(f, config.TEST_DIR / class_name / f.name)
        
        print(f"✓ Processed {class_name}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")


if __name__ == "__main__":
    src_dir = verify_source_dataset()
    setup_directories()
    split_dataset(src_dir)
