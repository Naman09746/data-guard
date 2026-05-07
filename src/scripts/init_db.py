import asyncio
from src.core.database import async_engine, Base
from src.core.models import Scan, EDAReport, Insight, Alert

async def init_db():
    async with async_engine.begin() as conn:
        # Import models here to ensure they are registered with Base.metadata
        await conn.run_sync(Base.metadata.create_all)
    print("Database tables created successfully.")

if __name__ == "__main__":
    asyncio.run(init_db())
