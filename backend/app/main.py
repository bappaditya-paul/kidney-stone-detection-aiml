"""NephroScan AI FastAPI Main Entrypoint."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.config import settings
from app.api.v1.endpoints import router as api_v1_router
from app.services.model_service import model_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup/shutdown."""
    print("=" * 60)
    print(f"🚀 Starting {settings.PROJECT_NAME} (v{settings.VERSION})")
    print(f"📁 Model Location: {settings.MODEL_PATH}")
    print("=" * 60)
    yield
    print("🛑 Shutting down NephroScan AI Backend API.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Production-grade AI microservice for kidney stone detection, Grad-CAM visualization, and ultrasound modality validation.",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_origin_regex=settings.CORS_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include v1 endpoints
app.include_router(api_v1_router, prefix=settings.API_V1_PREFIX, tags=["Diagnostic Endpoints"])


@app.get("/", include_in_schema=False)
def root():
    """Redirect root to OpenAPI interactive documentation."""
    return RedirectResponse(url="/docs")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
