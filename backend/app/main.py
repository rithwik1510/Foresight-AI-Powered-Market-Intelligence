"""
FastAPI Application Entry Point
Stock & Mutual Fund Analyzer API
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.config import settings
from app.api.v1.router import api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager"""
    # Startup
    print(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")

    # Initialize database tables
    from app.models.base import init_db, close_db
    try:
        await init_db()
        print("Database tables initialized")
    except Exception as e:
        print(f"Warning: Database initialization failed: {e}")

    yield

    # Shutdown
    print("Shutting down...")
    try:
        await close_db()
        print("Database connections closed")
    except Exception as e:
        print(f"Warning: Database cleanup failed: {e}")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI-powered Stock & Mutual Fund Analyzer for Indian Markets (NSE/BSE)",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api/v1")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}
