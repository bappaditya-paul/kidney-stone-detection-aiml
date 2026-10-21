"""Ultrasound Modality Guard: Out-Of-Distribution (OOD) Validation Service."""

from typing import Tuple, Dict, Any
import cv2
import numpy as np
from PIL import Image as PILImage
from app.config import settings


def validate_ultrasound_image(img_pil: PILImage.Image) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Validate whether the input image exhibits the physical and visual characteristics
    of a genuine B-mode medical ultrasound scan.
    
    Prevents Out-Of-Distribution false positives (e.g., animals, scenery, random photos).
    
    Returns:
        (is_valid: bool, status_message: str, metrics: dict)
    """
    img_rgb = np.array(img_pil.convert('RGB'))
    
    # 1. Color saturation check (B-mode ultrasound is acoustic/grayscale)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1] / 255.0
    mean_sat = float(np.mean(sat))
    high_sat_ratio = float(np.mean(sat > 0.30))
    
    # 2. Channel similarity check (Ultrasound scans have near-identical R, G, B channels)
    r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
    channel_diff = float(np.mean(
        np.abs(r.astype(int) - g.astype(int)) +
        np.abs(g.astype(int) - b.astype(int)) +
        np.abs(b.astype(int) - r.astype(int))
    ))
    
    # 3. Dynamic range and contrast check
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    std_dev = float(np.std(gray))
    mean_intensity = float(np.mean(gray))
    
    # 4. Acoustic speckle noise variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    speckle_var = float(laplacian.var())
    
    metrics = {
        'mean_saturation': round(mean_sat, 4),
        'high_saturation_ratio': round(high_sat_ratio, 4),
        'channel_divergence': round(channel_diff, 2),
        'contrast_std': round(std_dev, 2),
        'mean_intensity': round(mean_intensity, 2),
        'speckle_variance': round(speckle_var, 2)
    }
    
    # Validation Rules
    if mean_sat > settings.MAX_MEAN_SATURATION or high_sat_ratio > settings.MAX_HIGH_SAT_RATIO or channel_diff > settings.MAX_CHANNEL_DIVERGENCE:
        return (
            False,
            "Non-medical image detected (high chromaticity/color saturation). Kidney ultrasound scans are grayscale B-mode acoustic scans. Please upload an authentic B-mode kidney ultrasound image.",
            metrics
        )
    
    if std_dev < settings.MIN_CONTRAST_STD or speckle_var < settings.MIN_SPECKLE_VARIANCE:
        return (
            False,
            "Image lacks ultrasound acoustic speckle pattern or sufficient dynamic contrast. Please upload a clear medical scan.",
            metrics
        )
        
    if mean_intensity > settings.MAX_MEAN_INTENSITY or mean_intensity < settings.MIN_MEAN_INTENSITY:
        return (
            False,
            "Image is overexposed or underexposed (blank/uniform image detected).",
            metrics
        )
    
    return (True, "Valid B-mode medical ultrasound scan detected.", metrics)
