"""Async SQLAlchemy session construction for PostgreSQL-backed canonical state."""

import os

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


def database_url() -> str:
    """Read the required async PostgreSQL URL without exposing credentials in errors."""
    value = os.environ.get("DATABASE_URL", "").strip()
    if not value:
        raise RuntimeError("Set DATABASE_URL to use PostgreSQL persistence.")
    return value


def create_session_factory(
    url: str | None = None,
) -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    """Create an engine and session factory; callers own engine disposal on shutdown."""
    engine = create_async_engine(url or database_url(), pool_pre_ping=True)
    return engine, async_sessionmaker(engine, expire_on_commit=False)
