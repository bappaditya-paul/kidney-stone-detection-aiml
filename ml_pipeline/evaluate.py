"""ML Pipeline: Comprehensive Model Evaluation Script (ROC-AUC, Confusion Matrix, Classification Report)."""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)

# Ensure ml_pipeline is on sys.path
PIPELINE_ROOT = Path(__file__).resolve().parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

import config


def load_test_generator():
    """Load test dataset without augmentation."""
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_gen = test_datagen.flow_from_directory(
        config.TEST_DIR,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )
    return test_gen


def evaluate_model():
    """Run full evaluation suite on test dataset."""
    print("\n" + "="*60)
    print("🔍 ML PIPELINE - MODEL EVALUATION SUITE")
    print("="*60 + "\n")
    
    if not config.BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {config.BEST_MODEL_PATH}")
    
    print(f"📦 Loading model from: {config.BEST_MODEL_PATH}")
    model = load_model(config.BEST_MODEL_PATH)
    test_gen = load_test_generator()
    
    y_true = test_gen.classes
    print("🔮 Generating predictions...")
    y_probs = model.predict(test_gen, verbose=1).flatten()
    y_pred = (y_probs > config.THRESHOLD).astype(int)
    
    # Calculate Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_true, y_probs)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES)
    plt.title('Kidney Stone Detection - Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('Ground Truth')
    plt.tight_layout()
    cm_path = config.MODELS_DIR / 'confusion_matrix_eval.png'
    plt.savefig(cm_path)
    plt.close()
    
    # ROC Curve Plot
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='#0d9488', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    roc_path = config.MODELS_DIR / 'roc_curve.png'
    plt.savefig(roc_path)
    plt.close()
    
    # Save Report
    report = classification_report(y_true, y_pred, target_names=config.CLASS_NAMES)
    report_content = f"""==================================================
KIDNEY STONE DETECTION - MODEL EVALUATION REPORT
==================================================

Accuracy:  {accuracy:.4f}
Precision: {precision:.4f}
Recall:    {recall:.4f}
F1-Score:  {f1:.4f}
ROC-AUC:   {roc_auc:.4f}

--------------------------------------------------
Classification Report:
{report}
==================================================
"""
    report_path = config.MODELS_DIR / 'evaluation_report.txt'
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    print("\n" + "="*60)
    print("✅ EVALUATION RESULTS:")
    print("="*60)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"Artifacts saved in: {config.MODELS_DIR}")
    print(f" - Report: {report_path.name}")
    print(f" - Confusion Matrix: {cm_path.name}")
    print(f" - ROC Curve: {roc_path.name}")
    print("="*60)


if __name__ == "__main__":
    evaluate_model()
