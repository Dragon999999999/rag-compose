"""Run canonical PostgreSQL migrations using environment-based configuration."""

import asyncio
import os
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from dotenv import load_dotenv
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from llm_grounding.infrastructure.database.models import Base

PROJECT_ROOT = Path(__file__).resolve().parents[5]
load_dotenv(PROJECT_ROOT / ".env")

config = context.config
if config.config_file_name:
    fileConfig(config.config_file_name)
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Generate SQL without connecting when DATABASE_URL is supplied."""
    context.configure(
        url=os.environ.get("DATABASE_URL", config.get_main_option("sqlalchemy.url")),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Apply migrations over Alembic's synchronous bridge connection."""
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """Connect using asyncpg and execute all migration revisions."""
    settings = config.get_section(config.config_ini_section, {})
    settings["sqlalchemy.url"] = os.environ.get("DATABASE_URL") or settings.get(
        "sqlalchemy.url", ""
    )
    connectable = async_engine_from_config(
        settings, prefix="sqlalchemy.", poolclass=pool.NullPool
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
