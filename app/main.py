from fastapi import FastAPI

from app.config import settings
from app.api.routes.ingest import router as ingest_router


app = FastAPI(
    title=settings.app_name,
    description="Customer Support RAG Application",
    version="1.0.0",
)


app.include_router(ingest_router)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "application": settings.app_name,
    }