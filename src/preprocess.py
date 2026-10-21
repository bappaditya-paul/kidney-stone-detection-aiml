import os
import sys
import shutil
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config


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
        print(f"  - Created directory: {directory}")
    
    print("✓ Directory structure ready")


def verify_source_dataset():
    """Verify that source images exist in raw or clean dataset directories."""
    print("🔍 Verifying dataset structure...")
    
    source_dir = config.CLEAN_DATA_DIR if config.CLEAN_DATA_DIR.exists() else config.RAW_DATA_DIR
    
    if not source_dir.exists():
        raise FileNotFoundError(
            f"❌ Error: Dataset source folder not found at: {source_dir}\n"
            "Please ensure you have a 'stone' and 'no_stone' folder."
        )
    
    class_dirs = config.CLASS_NAMES
    for cls in class_dirs:
        cls_dir = source_dir / cls
        if not cls_dir.exists():
            raise FileNotFoundError(f"❌ Missing class folder: {cls_dir}")
        
        img_count = len([
            f for f in cls_dir.iterdir()
            if f.is_file() and f.suffix.lower() in config.VALID_IMAGE_EXTENSIONS
        ])
        
        if img_count == 0:
            raise FileNotFoundError(
                f"❌ No valid images found in {cls_dir}/\n"
                f"Supported formats: {', '.join(config.VALID_IMAGE_EXTENSIONS)}"
            )
        
        print(f"✓ Found {img_count} images in '{cls}' class")
        
    return source_dir


def split_dataset(source_dir):
    """Split the dataset into train, validation, and test sets."""
    print("\n🔄 Splitting dataset into train, validation, and test sets...")
    
    for class_name in config.CLASS_NAMES:
        class_dir = source_dir / class_name
        image_files = [
            f for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in config.VALID_IMAGE_EXTENSIONS
        ]
        
        # Split into train and (val + test)
        train_files, val_test_files = train_test_split(
            image_files,
            test_size=config.TEST_SPLIT + config.VAL_SPLIT,
            random_state=config.SEED
        )
        
        # Split val_test into val and test
        val_files, test_files = train_test_split(
            val_test_files,
            test_size=config.TEST_SPLIT / (config.TEST_SPLIT + config.VAL_SPLIT),
            random_state=config.SEED
        )
        
        # Copy files to destination directories
        for file in train_files:
            shutil.copy2(file, config.TRAIN_DIR / class_name / file.name)
        for file in val_files:
            shutil.copy2(file, config.VAL_DIR / class_name / file.name)
        for file in test_files:
            shutil.copy2(file, config.TEST_DIR / class_name / file.name)
            
        print(f"  ✓ {class_name.upper()}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")


def create_generators():
    """Create data generators with medical augmentation."""
    print("\n🔄 Creating data generators...")
    
    # Training augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )
    
    # Validation and test generators (only rescaling)
    eval_datagen = ImageDataGenerator(rescale=1./255)
    
    train_ds = train_datagen.flow_from_directory(
        config.TRAIN_DIR,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode="binary",
        shuffle=True,
        seed=config.SEED
    )
    
    val_ds = eval_datagen.flow_from_directory(
        config.VAL_DIR,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode="binary",
        shuffle=False
    )
    
    test_ds = eval_datagen.flow_from_directory(
        config.TEST_DIR,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode="binary",
        shuffle=False
    )
    
    return train_ds, val_ds, test_ds


def main():
    print("\n" + "="*60)
    print("🚀 KIDNEY STONE DETECTION - DATA PREPROCESSING")
    print("="*60)
    
    try:
        source_dir = verify_source_dataset()
        setup_directories()
        split_dataset(source_dir)
        train_ds, val_ds, test_ds = create_generators()
        
        print("\n" + "="*60)
        print("🎉 PREPROCESSING COMPLETE!")
        print("="*60)
        print(f"  - Training samples:   {train_ds.samples:>6}")
        print(f"  - Validation samples: {val_ds.samples:>6}")
        print(f"  - Test samples:       {test_ds.samples:>6}")
        print(f"\n💡 Next step: Run 'python src/train_cnn.py' to train your model.")
        return 0
    except Exception as e:
        print(f"\n❌ Preprocessing error: {str(e)}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())