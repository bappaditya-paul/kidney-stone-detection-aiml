"""Kidney Stone Detection: CNN Model Training with Transfer Learning."""

import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config


def create_data_generators():
    """Create data generators for training, validation, and testing."""
    print("📊 Creating data generators...")
    
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
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        config.TRAIN_DIR,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='binary',
        shuffle=True,
        seed=config.SEED
    )
    
    val_generator = test_datagen.flow_from_directory(
        config.VAL_DIR,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )
    
    test_generator = test_datagen.flow_from_directory(
        config.TEST_DIR,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator


def build_model(backbone="mobilenet_v2"):
    """Build and compile the transfer learning model."""
    print(f"🔨 Building model with backbone: {backbone}...")
    
    inputs = Input(shape=(*config.IMG_SIZE, 3))
    
    if backbone == "efficientnet":
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_tensor=inputs)
    else:
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=inputs)
    
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def plot_training_history(history):
    """Plot and save training and validation accuracy/loss."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    output_path = config.MODEL_DIR / 'training_history.png'
    plt.savefig(output_path)
    plt.close()
    print(f"📊 Training curves saved to: {output_path}")


def train(epochs=config.EPOCHS, backbone="mobilenet_v2"):
    """Execute model training pipeline."""
    print("🚀 Starting model training...")
    
    train_generator, val_generator, test_generator = create_data_generators()
    model = build_model(backbone=backbone)
    
    callbacks = [
        ModelCheckpoint(
            filepath=config.BEST_MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
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
    
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    model.save(config.FINAL_MODEL_PATH)
    plot_training_history(history)
    
    print(f"\n✅ Training complete! Best model saved at: {config.BEST_MODEL_PATH}")


if __name__ == "__main__":
    train()