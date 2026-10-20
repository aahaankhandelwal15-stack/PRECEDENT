"""FastAPI application for Precedent V1."""

from __future__ import annotations

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import ingest_router, retrieval_router, trace_router
from app.core.database import init_db
from app.core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    init_db()
    yield
    # Shutdown (cleanup if needed)


app = FastAPI(
    title="Precedent V1",
    description="Internal Policy Assurance Engine — Retrieval-Augmented Policy QA with Auditability",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    import traceback
    traceback.print_exc()
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "category": "internal",
                "retryable": False,
                "safe_message": "An internal server error occurred",
                "diagnostic_code": f"ERR-INTERNAL-{type(exc).__name__}",
            },
            "trace_id": None,
        },
    )


# Include routers
app.include_router(ingest_router, prefix="/api/v1")
app.include_router(retrieval_router, prefix="/api/v1")
app.include_router(trace_router, prefix="/api/v1")


# Health check endpoint
@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "precedent",
        "version": "0.1.0",
    }


# Root endpoint
@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API info."""
    return {
        "service": "Precedent V1",
        "description": "Internal Policy Assurance Engine",
        "version": "0.1.0",
        "docs_url": "/docs",
        "endpoints": {
            "ingest": "/api/v1/ingest",
            "retrieve": "/api/v1/retrieval/retrieve",
            "query": "/api/v1/retrieval/query",
            "trace": "/api/v1/trace/{trace_id}",
            "health": "/health",
        },
    }