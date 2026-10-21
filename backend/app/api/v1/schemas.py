"""Pydantic Request & Response Schemas."""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class ValidationMetrics(BaseModel):
    mean_saturation: float
    high_saturation_ratio: float
    channel_divergence: float
    contrast_std: float
    mean_intensity: float
    speckle_variance: float


class ValidationResponse(BaseModel):
    is_valid_ultrasound: bool
    message: str
    metrics: Dict[str, Any]


class DilationMetrics(BaseModel):
    severity_level: str
    dilation_status: str
    hydronephrosis_risk: str
    focal_stone_ratio: float
    clinical_advisory: str


class PredictionResponse(BaseModel):
    is_valid_scan: bool
    label: str
    probability: float
    confidence: float
    dilation_metrics: Optional[DilationMetrics] = None
    gradcam_data_uri: Optional[str] = None
    validation_message: Optional[str] = None
    validation_metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str = "healthy"
    model_loaded: bool
    version: str
