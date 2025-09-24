import logging

from app.api.api_v1.api import api_router
from app.api.openapi.api import router as openapi_router
from app.core.config import settings
from app.core.minio import init_minio
from app.startup.migarate import DatabaseMigrator
from fastapi import FastAPI
from app.db.session import SessionLocal
from app.startup.seed_data import seed_knowledge_base

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
)

# Include routers
app.include_router(api_router, prefix=settings.API_V1_STR)
app.include_router(openapi_router, prefix="/openapi")


@app.on_event("startup")
async def startup_event():
    # Initialize MinIO
    init_minio()
    
    # Run database migrations
    migrator = DatabaseMigrator(settings.get_database_url)
    migrator.run_migrations()

       # Tạo session DB
    db = SessionLocal()
    try:
        # Gọi hàm seed data tạo user, knowledge base, upload document
        await seed_knowledge_base(db)
    except Exception as e:
        print(f"Seed data failed: {e}")
    finally:
        db.close()
    


@app.get("/")
def root():
    return {"message": "Welcome to RAG Web UI API"}


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "version": settings.VERSION,
    }
