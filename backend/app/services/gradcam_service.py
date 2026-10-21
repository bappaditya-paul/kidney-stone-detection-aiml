"""Grad-CAM Service for Explainable AI (XAI) Activation Visualizations."""

import io
import base64
from typing import Optional, Tuple
import cv2
import numpy as np
from PIL import Image as PILImage
import tensorflow as tf


class GradCAMService:
    """Computes Grad-CAM attention heatmaps and base64 overlays."""

    def __init__(self, model, target_layer_name: str = 'out_relu'):
        self.model = model
        self.target_layer_name = target_layer_name
        self.grad_model = self._build_grad_model()

    def _build_grad_model(self) -> Optional[tf.keras.models.Model]:
        try:
            target_layer = self.model.get_layer(self.target_layer_name)
            return tf.keras.models.Model(
                inputs=self.model.inputs,
                outputs=[target_layer.output, self.model.output]
            )
        except Exception:
            return None

    def generate_heatmap(self, processed_img: np.ndarray, original_pil: PILImage.Image) -> Tuple[Optional[str], Optional[np.ndarray]]:
        """
        Generate Grad-CAM heatmap overlay.
        
        Returns:
            (base64_data_uri, heatmap_resized_array)
        """
        if self.grad_model is None:
            return None, None

        try:
            with tf.GradientTape() as tape:
                conv_outputs, predictions = self.grad_model(processed_img)
                loss = predictions[:, 0]

            grads = tape.gradient(loss, conv_outputs)
            if grads is None:
                return None, None

            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            conv_outputs = conv_outputs[0]
            heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + tf.keras.backend.epsilon())
            heatmap = heatmap.numpy()

            # Resize heatmap to original image dimensions
            orig_w, orig_h = original_pil.size
            heatmap_resized = cv2.resize(heatmap, (orig_w, orig_h))
            heatmap_uint8 = np.uint8(255 * heatmap_resized)

            # Apply COLORMAP_JET
            colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            colored_heatmap_rgb = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

            # Blend with original ultrasound image
            orig_rgb = np.array(original_pil.convert('RGB'))
            blended = cv2.addWeighted(orig_rgb, 0.65, colored_heatmap_rgb, 0.35, 0)

            # Convert to PIL Image & Base64 Data URI
            overlay_pil = PILImage.fromarray(blended)
            buffered = io.BytesIO()
            overlay_pil.save(buffered, format="PNG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            data_uri = f"data:image/png;base64,{img_b64}"

            return data_uri, heatmap_resized

        except Exception as e:
            print(f"Warning: Grad-CAM generation failed: {e}")
            return None, None
