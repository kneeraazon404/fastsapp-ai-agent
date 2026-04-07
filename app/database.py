"""
Database engine and session factory.

The engine is built lazily from settings so that importing this module
during tests does not attempt a database connection unless a session is
actually requested.
"""
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.engine import URL, Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings

_engine: Engine | None = None
_SessionLocal: sessionmaker | None = None


def _get_engine() -> Engine:
    global _engine
    if _engine is None:
        s = get_settings()
        url = URL.create(
            drivername="postgresql",
            username=s.db_user,
            password=s.db_password,
            host=s.db_host,
            database=s.db_name,
            port=s.db_port,
        )
        _engine = create_engine(url, pool_pre_ping=True)
    return _engine


def _get_session_factory() -> sessionmaker:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=_get_engine(),
        )
    return _SessionLocal


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a database session per request."""
    factory = _get_session_factory()
    db: Session = factory()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables if they do not already exist. Called at app startup."""
    from app.models import Base  # local import avoids circular dependency

    Base.metadata.create_all(bind=_get_engine())
