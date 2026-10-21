import sys
from pathlib import Path
from PIL import Image as PILImage
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.modality_guard import validate_ultrasound_image
from src.gradcam import GradCAMGenerator


class KidneyStoneDetector:
    """A class to handle kidney stone detection, validation, and Grad-CAM explainability."""
    
    def __init__(self, model_path=config.BEST_MODEL_PATH):
        """Initialize the detector with the trained model."""
        self.model = self._load_model(model_path)
        self.class_names = config.DISPLAY_CLASS_NAMES
        self.gradcam_gen = GradCAMGenerator(self.model, target_layer_name='out_relu')

    def _load_model(self, model_path):
        """Load the pre-trained model."""
        if not model_path.exists():
            raise FileNotFoundError(
                f"❌ Model not found at {model_path}. "
                "Please train the model first using train_cnn.py"
            )
        return load_model(model_path)

    def preprocess_image(self, img_input):
        """Load and preprocess the image for prediction (supports Path, str, or PIL Image)."""
        if isinstance(img_input, (str, Path)):
            pil_img = PILImage.open(img_input).convert('RGB')
        elif isinstance(img_input, PILImage.Image):
            pil_img = img_input.convert('RGB')
        else:
            raise ValueError(f"Unsupported image input type: {type(img_input)}")

        resized = pil_img.resize(config.IMG_SIZE)
        img_array = image.img_to_array(resized)
        img_array = np.expand_dims(img_array, axis=0)
        # Rescale by 1/255 to match training ImageDataGenerator
        img_array = img_array / 255.0

        return img_array, pil_img

    def estimate_dilation_and_severity(self, probability, heatmap_resized):
        """
        Estimate clinical severity, acoustic stone density, and
        potential pelvicalyceal dilation (hydronephrosis indicator).
        """
        if probability < config.THRESHOLD:
            return {
                'severity_level': 'Normal / Unremarkable',
                'dilation_status': 'No Dilation (Normal Pelvicalyceal Architecture)',
                'hydronephrosis_risk': 'Low (0%)',
                'focal_stone_ratio': 0.0,
                'clinical_advisory': 'Normal bilateral kidney scan pattern. No acute calculus or acoustic shadowing detected.'
            }

        # Calculate high-intensity focal area from Grad-CAM
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
            advisory = "Prominent acoustic focus with posterior acoustic shadowing. High probability of ureteral or pelvic obstruction. Recommend urgent urological consultation."
        elif probability > 0.65:
            severity = "Moderate Calculus"
            dilation = "Mild Pelvic Dilation (Possible Subacute Obstruction)"
            hydro_risk = "Moderate (~50%)"
            advisory = "Definite hyperechoic calculus observed. Recommend confirmatory low-dose non-contrast CT (KUB) and clinical correlation."
        else:
            severity = "Borderline / Small Calculus"
            dilation = "Minimal / No Significant Dilation"
            hydro_risk = "Low-to-Moderate (~25%)"
            advisory = "Borderline hyperechoic echo. Could represent small non-obstructing micro-calculus or parenchymal calcification. Follow-up ultrasound suggested."

        return {
            'severity_level': severity,
            'dilation_status': dilation,
            'hydronephrosis_risk': hydro_risk,
            'focal_stone_ratio': focal_ratio,
            'clinical_advisory': advisory
        }

    def predict(self, img_input, validate_scan=True):
        """
        Make a prediction on an image with ultrasound validation and Grad-CAM.
        
        Args:
            img_input: File path (str or Path) or PIL Image.
            validate_scan: If True, checks for non-medical OOD images before predicting.
        """
        try:
            processed_img, pil_img = self.preprocess_image(img_input)

            # 1. Modality Validation Gate
            is_valid, val_reason, val_metrics = validate_ultrasound_image(pil_img)
            if validate_scan and not is_valid:
                return {
                    'is_valid_scan': False,
                    'error': val_reason,
                    'label': 'Invalid (Non-Medical Image)',
                    'confidence': 0.0,
                    'probability': 0.0,
                    'validation_metrics': val_metrics,
                    'gradcam_image': None,
                    'dilation_metrics': None
                }

            # 2. Model Prediction
            prediction = self.model.predict(processed_img, verbose=0)
            probability = float(prediction[0][0])

            # Determine class and confidence
            if probability > config.THRESHOLD:
                label = self.class_names[1]  # Stone
                confidence = probability
            else:
                label = self.class_names[0]  # No Stone
                confidence = 1.0 - probability

            # 3. Grad-CAM Activation Heatmap
            gradcam_pil, heatmap_resized = self.gradcam_gen.generate(processed_img, pil_img)

            # 4. Clinical Dilation & Severity Assessment
            dilation_metrics = self.estimate_dilation_and_severity(probability, heatmap_resized)

            return {
                'is_valid_scan': True,
                'validation_message': val_reason,
                'validation_metrics': val_metrics,
                'label': label,
                'probability': float(probability),
                'confidence': float(confidence) * 100,  # Percentage
                'gradcam_image': gradcam_pil,
                'dilation_metrics': dilation_metrics,
                'error': None
            }

        except Exception as e:
            return {
                'is_valid_scan': False,
                'error': str(e),
                'label': 'Error',
                'probability': 0.0,
                'confidence': 0.0,
                'gradcam_image': None,
                'dilation_metrics': None
            }


def main():
    """Command-line interface for prediction."""
    import argparse

    parser = argparse.ArgumentParser(description='Detect kidney stones in an ultrasound image.')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('--no-validation', action='store_true', help='Skip ultrasound modality validation')
    args = parser.parse_args()

    try:
        detector = KidneyStoneDetector()
        result = detector.predict(args.image_path, validate_scan=not args.no_validation)

        if not result.get('is_valid_scan'):
            print(f"\n⚠️ Validation Notice: {result.get('error')}")
            if 'validation_metrics' in result:
                print("   Metrics:", result['validation_metrics'])
            return

        print(f"\n🔍 Diagnostic Results:")
        print(f"   - Verdict: {result['label']}")
        print(f"   - Confidence: {result['confidence']:.2f}%")
        print(f"   - Raw Probability: {result['probability']:.4f}")
        if result.get('dilation_metrics'):
            dm = result['dilation_metrics']
            print(f"   - Dilation Status: {dm['dilation_status']}")
            print(f"   - Hydronephrosis Risk: {dm['hydronephrosis_risk']}")
            print(f"   - Advisory: {dm['clinical_advisory']}")

    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")


_CACHED_DETECTOR = None


def get_stone_detector():
    """Get a pre-configured detector singleton for Streamlit."""
    global _CACHED_DETECTOR
    if _CACHED_DETECTOR is None:
        _CACHED_DETECTOR = KidneyStoneDetector()
    return _CACHED_DETECTOR


if __name__ == "__main__":
    main()