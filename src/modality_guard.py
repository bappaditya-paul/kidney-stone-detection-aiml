"""Ultrasound Modality Guard: Out-Of-Distribution (OOD) Validation."""

import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image as PILImage

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config


def validate_ultrasound_image(img_pil: PILImage.Image):
    """
    Validate whether the input image exhibits the physical and visual characteristics
    of a genuine B-mode medical ultrasound scan.
    
    Prevents Out-Of-Distribution (OOD) false positives (e.g. users uploading animals,
    selfies, landscapes, cars, text documents, or flat graphics).
    
    Returns:
        (is_valid: bool, status_message: str, metrics: dict)
    """
    img_rgb = np.array(img_pil.convert('RGB'))
    
    # 1. Color saturation check (B-mode ultrasound is acoustic / grayscale)
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
    
    # 4. Acoustic texture / speckle measure (Laplacian variance)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    
    metrics = {
        'mean_saturation': round(mean_sat, 4),
        'high_saturation_ratio': round(high_sat_ratio, 4),
        'channel_divergence': round(channel_diff, 2),
        'contrast_std': round(std_dev, 2),
        'mean_intensity': round(mean_intensity, 2),
        'speckle_variance': round(laplacian_var, 2)
    }
    
    # Rejection criteria for non-ultrasound images
    if (
        mean_sat > config.MAX_MEAN_SATURATION
        or high_sat_ratio > config.MAX_HIGH_SAT_RATIO
        or channel_diff > config.MAX_CHANNEL_DIVERGENCE
    ):
        return (
            False,
            "Non-medical image detected (high chromaticity/color saturation). "
            "Kidney ultrasound scans are grayscale B-mode acoustic scans. "
            "Please upload an authentic B-mode kidney ultrasound image.",
            metrics
        )
        
    if std_dev < config.MIN_CONTRAST_STD:
        return (
            False,
            "Image lacks sufficient dynamic contrast (blank, solid color, or washed out). "
            "Please upload a clear, diagnostic-grade kidney ultrasound scan.",
            metrics
        )
        
    if laplacian_var < config.MIN_SPECKLE_VARIANCE:
        return (
            False,
            "Image lacks ultrasound acoustic speckle patterns and anatomical edges. "
            "Please upload an authentic ultrasound image.",
            metrics
        )
        
    if mean_intensity > config.MAX_MEAN_INTENSITY or mean_intensity < config.MIN_MEAN_INTENSITY:
        return (
            False,
            "Image is severely overexposed or underexposed. "
            "Cannot identify renal anatomical structures.",
            metrics
        )
        
    return (True, "Verified B-mode Ultrasound Scan", metrics)
