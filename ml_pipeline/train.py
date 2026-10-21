"""ML Pipeline: MobileNetV2 Transfer Learning Training Script."""

import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Ensure ml_pipeline is on sys.path
PIPELINE_ROOT = Path(__file__).resolve().parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

import config


def create_data_generators():
    """Create data generators with augmentation for training and testing."""
    print("📊 Creating dataset generators...")
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    eval_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        config.TRAIN_DIR,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='binary',
        shuffle=True,
        seed=config.SEED
    )
    
    val_generator = eval_datagen.flow_from_directory(
        config.VAL_DIR,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )
    
    test_generator = eval_datagen.flow_from_directory(
        config.TEST_DIR,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator


def build_model():
    """Build MobileNetV2 transfer learning model."""
    print("🏗️ Building MobileNetV2 architecture...")
    
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(*config.IMG_SIZE, 3)
    )
    
    # Freeze base model weights
    base_model.trainable = False
    
    inputs = Input(shape=(*config.IMG_SIZE, 3), name='input_layer')
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs, name="KidneyStoneDetector_MobileNetV2")
    
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_model():
    """Execute the training loop with callbacks."""
    print("\n" + "="*60)
    print("🚀 STARTING CNN TRAINING")
    print("="*60 + "\n")
    
    train_gen, val_gen, test_gen = create_data_generators()
    model = build_model()
    
    callbacks = [
        ModelCheckpoint(
            filepath=str(config.BEST_MODEL_PATH),
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    print("\n📈 Training model...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save(config.FINAL_MODEL_PATH)
    print(f"\n💾 Saved final model to: {config.FINAL_MODEL_PATH}")
    print(f"🏆 Best checkpoint saved to: {config.BEST_MODEL_PATH}")
    
    return model, history


if __name__ == "__main__":
    train_model()
