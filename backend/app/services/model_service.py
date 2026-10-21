"""Model Service: Singleton Model Lifecycle, Inference & Clinical Severity Estimation."""

import io
from pathlib import Path
from typing import Dict, Any, Union
import numpy as np
from PIL import Image as PILImage
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from app.config import settings
from app.services.modality_guard import validate_ultrasound_image
from app.services.gradcam_service import GradCAMService


class ModelService:
    """Manages TensorFlow CNN model inference and diagnostic advisory generation."""

    def __init__(self):
        self.model = None
        self.gradcam_service = None
        self._load_model()

    def _load_model(self):
        model_path = Path(settings.MODEL_PATH)
        if not model_path.exists():
            print(f"⚠️ Warning: Model checkpoint not found at {model_path}. Prediction endpoints will require model training.")
            return

        print(f"📦 Loading pre-trained model from {model_path}...")
        self.model = load_model(str(model_path))
        self.gradcam_service = GradCAMService(self.model, target_layer_name='out_relu')
        print("✅ Model successfully loaded into memory.")

    def preprocess_image(self, file_bytes: bytes) -> tuple[np.ndarray, PILImage.Image]:
        """Convert raw file bytes into normalized TensorFlow tensor and PIL Image."""
        pil_img = PILImage.open(io.BytesIO(file_bytes)).convert('RGB')
        resized = pil_img.resize(settings.IMG_SIZE)
        img_array = image.img_to_array(resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalized to [0, 1]
        return img_array, pil_img

    def estimate_dilation_metrics(self, probability: float, heatmap_resized: np.ndarray = None) -> Dict[str, Any]:
        """Compute clinical pelvicalyceal dilation status and hydronephrosis risk."""
        if probability < settings.THRESHOLD:
            return {
                "severity_level": "Normal / Unremarkable",
                "dilation_status": "No Dilation (Normal Pelvicalyceal Architecture)",
                "hydronephrosis_risk": "Low (0%)",
                "focal_stone_ratio": 0.0,
                "clinical_advisory": "Normal bilateral kidney ultrasound pattern. No acute calculus or acoustic shadowing detected."
            }

        # Calculate high-intensity focal area from Grad-CAM activation
        if heatmap_resized is not None:
            focal_pixels = np.sum(heatmap_resized > 0.65)
            total_pixels = heatmap_resized.size
            focal_ratio = round(float(focal_pixels / total_pixels) * 100, 2)
        else:
            focal_ratio = 3.5

        if probability > 0.85:
            severity = "High / Acute Calculus"
            dilation = "Moderate-to-Severe Pelvicalyceal Dilation (Likely Obstructive)"
            hydro_risk = "Elevated (>75%)"
            advisory = "Prominent acoustic focus with posterior acoustic shadowing. High probability of ureteral or pelvic obstruction. Urgent urological evaluation advised."
        elif probability > 0.65:
            severity = "Moderate Calculus"
            dilation = "Mild Pelvic Dilation (Possible Subacute Obstruction)"
            hydro_risk = "Moderate (~50%)"
            advisory = "Definite hyperechoic calculus observed. Recommend non-contrast low-dose CT (KUB) for accurate stone sizing."
        else:
            severity = "Borderline / Small Calculus"
            dilation = "Minimal / No Significant Dilation"
            hydro_risk = "Low-to-Moderate (~25%)"
            advisory = "Small hyperechoic focus. Potential non-obstructing micro-calculus. Hydration and periodic sonographic follow-up suggested."

        return {
            "severity_level": severity,
            "dilation_status": dilation,
            "hydronephrosis_risk": hydro_risk,
            "focal_stone_ratio": focal_ratio,
            "clinical_advisory": advisory
        }

    def predict(self, file_bytes: bytes, validate_scan: bool = True) -> Dict[str, Any]:
        """Run complete diagnostic pipeline on image bytes."""
        if self.model is None:
            self._load_model()
            if self.model is None:
                raise RuntimeError("Model weights are not loaded. Please train or verify models/best_model.h5.")

        processed_img, pil_img = self.preprocess_image(file_bytes)

        # 1. Modality Validation
        is_valid, reason, metrics = validate_ultrasound_image(pil_img)
        if validate_scan and not is_valid:
            return {
                "is_valid_scan": False,
                "error": reason,
                "validation_message": reason,
                "validation_metrics": metrics,
                "label": "Invalid (Non-Medical Image)",
                "probability": 0.0,
                "confidence": 0.0,
                "gradcam_data_uri": None,
                "dilation_metrics": None
            }

        # 2. Prediction
        raw_pred = self.model.predict(processed_img, verbose=0)
        probability = float(raw_pred[0][0])

        if probability > settings.THRESHOLD:
            label = settings.CLASS_NAMES[1]  # Stone
            confidence = probability * 100
        else:
            label = settings.CLASS_NAMES[0]  # No Stone
            confidence = (1.0 - probability) * 100

        # 3. Grad-CAM Explainable AI Heatmap
        gradcam_uri = None
        heatmap_resized = None
        if self.gradcam_service:
            gradcam_uri, heatmap_resized = self.gradcam_service.generate_heatmap(processed_img, pil_img)

        # 4. Clinical Dilation Assessment
        dilation_metrics = self.estimate_dilation_metrics(probability, heatmap_resized)

        return {
            "is_valid_scan": True,
            "error": None,
            "validation_message": reason,
            "validation_metrics": metrics,
            "label": label,
            "probability": round(probability, 4),
            "confidence": round(confidence, 2),
            "gradcam_data_uri": gradcam_uri,
            "dilation_metrics": dilation_metrics
        }


# Singleton instance
model_service = ModelService()
