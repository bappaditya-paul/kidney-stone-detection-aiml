"""API v1 Route Endpoints: /predict, /validate, /health."""

import io
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from PIL import Image as PILImage

from app.api.v1.schemas import PredictionResponse, ValidationResponse, HealthResponse
from app.services.model_service import model_service
from app.services.modality_guard import validate_ultrasound_image
from app.config import settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Verify backend API readiness and model status."""
    return HealthResponse(
        status="healthy",
        model_loaded=model_service.model is not None,
        version=settings.VERSION
    )


@router.post("/validate", response_model=ValidationResponse)
async def validate_scan(file: UploadFile = File(...)):
    """Pre-inference validation: Check if uploaded image is authentic B-mode kidney ultrasound."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be a valid image (JPEG, PNG)."
        )

    try:
        contents = await file.read()
        pil_img = PILImage.open(io.BytesIO(contents)).convert('RGB')
        is_valid, message, metrics = validate_ultrasound_image(pil_img)
        return ValidationResponse(
            is_valid_ultrasound=is_valid,
            message=message,
            metrics=metrics
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to process image: {str(e)}"
        )


@router.post("/predict", response_model=PredictionResponse)
async def predict_scan(
    file: UploadFile = File(...),
    validate_scan: bool = Form(True)
):
    """
    Predict kidney stone presence from ultrasound image.
    
    Includes:
    - Ultrasound Modality Guard (prevents OOD false positives)
    - MobileNetV2 inference & probability
    - Explainable AI Grad-CAM heatmap visualization (Base64 data URI)
    - Clinical Pelvicalyceal Dilation & Severity Index
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be a valid image."
        )

    try:
        contents = await file.read()
        result = model_service.predict(contents, validate_scan=validate_scan)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )
