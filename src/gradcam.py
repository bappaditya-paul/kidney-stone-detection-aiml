"""Grad-CAM Explainable AI (XAI) for focal calculus localization."""

import cv2
import numpy as np
from PIL import Image as PILImage
import tensorflow as tf


class GradCAMGenerator:
    """Extracts and overlays Grad-CAM activation maps for convolutional models."""

    def __init__(self, model, target_layer_name='out_relu'):
        self.model = model
        self.target_layer_name = target_layer_name
        self.grad_model = self._build_grad_model()

    def _build_grad_model(self):
        try:
            target_layer = self.model.get_layer(self.target_layer_name)
            return tf.keras.models.Model(
                inputs=self.model.inputs,
                outputs=[target_layer.output, self.model.output]
            )
        except Exception:
            return None

    def generate(self, img_array, orig_pil_img: PILImage.Image):
        """
        Compute Grad-CAM activation map overlaid onto the original ultrasound scan.

        Args:
            img_array: Preprocessed numpy array of shape (1, 224, 224, 3) normalized in [0, 1].
            orig_pil_img: Original PIL Image.

        Returns:
            (overlay_pil: PILImage, heatmap_resized: np.ndarray)
        """
        if self.grad_model is None:
            return None, None

        try:
            with tf.GradientTape() as tape:
                conv_outputs, predictions = self.grad_model(img_array)
                loss = predictions[:, 0]

            grads = tape.gradient(loss, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            conv_outputs = conv_outputs[0]
            heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
            heatmap_np = heatmap.numpy()

            # Resize to match original image dimensions
            orig_w, orig_h = orig_pil_img.size
            heatmap_resized = cv2.resize(heatmap_np, (orig_w, orig_h))
            heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

            orig_arr = np.array(orig_pil_img)
            overlay_arr = cv2.addWeighted(orig_arr, 0.65, heatmap_colored, 0.35, 0)
            overlay_pil = PILImage.fromarray(overlay_arr)

            return overlay_pil, heatmap_resized
        except Exception:
            return None, None
