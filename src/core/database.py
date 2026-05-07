from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from src.core.config import get_settings

settings = get_settings()

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# 1. Async Setup (for FastAPI)
async_engine = create_async_engine(
    settings.database.url,
    pool_size=settings.database.pool_size,
    max_overflow=settings.database.max_overflow,
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# 2. Sync Setup (for Celery Workers)
# Convert asyncpg URL to standard postgresql if needed
sync_url = settings.database.url.replace("postgresql+asyncpg://", "postgresql://")
sync_engine = create_engine(sync_url)
SessionLocal = sessionmaker(
    bind=sync_engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)

class Base(DeclarativeBase):
    pass

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
