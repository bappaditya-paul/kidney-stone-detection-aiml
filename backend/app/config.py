"""Backend Configuration & Environment Settings."""

import os
from pathlib import Path
from typing import List

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Resolve model path flexibly across Docker container and local runtime
DEFAULT_MODEL_PATH = Path(os.getenv("MODEL_PATH", str(BASE_DIR / "ml_pipeline" / "models" / "best_model.h5")))
if not DEFAULT_MODEL_PATH.exists():
    fallback_path = BASE_DIR / "models" / "best_model.h5"
    if fallback_path.exists():
        DEFAULT_MODEL_PATH = fallback_path

class Settings:
    PROJECT_NAME: str = "NephroScan AI Clinical API"
    VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api/v1"
    
    # Model parameters
    MODEL_PATH: Path = DEFAULT_MODEL_PATH
    IMG_SIZE: tuple = (224, 224)
    THRESHOLD: float = 0.5
    CLASS_NAMES: list = ["No Stone", "Stone"]
    
    # CORS Origins (allow frontend dev server and standard ports)
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "*"
    ]
    
    # Modality Guard Thresholds (Out-of-Distribution validation)
    MAX_MEAN_SATURATION: float = 0.18
    MAX_HIGH_SAT_RATIO: float = 0.12
    MAX_CHANNEL_DIVERGENCE: float = 16.0
    MIN_CONTRAST_STD: float = 15.0
    MIN_SPECKLE_VARIANCE: float = 8.0
    MAX_MEAN_INTENSITY: float = 235.0
    MIN_MEAN_INTENSITY: float = 8.0

settings = Settings()
