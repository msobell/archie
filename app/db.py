import logging
import os
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
MEMORY_DB_PATH = os.path.join(PROJECT_ROOT, ".memory.db")

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


def _make_engine():
    engine = create_engine(f"sqlite:///{MEMORY_DB_PATH}", connect_args={"check_same_thread": False})

    @event.listens_for(engine, "connect")
    def _on_connect(dbapi_conn, _):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()

        try:
            import sqlite_vec
            dbapi_conn.enable_load_extension(True)
            sqlite_vec.load(dbapi_conn)
            dbapi_conn.enable_load_extension(False)
        except ImportError:
            logger.warning("sqlite-vec not installed — vector search disabled.")
        except Exception as e:
            logger.error(f"Failed to load sqlite-vec: {e}")

    return engine


engine = _make_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def create_tables():
    import app.models.memory  # noqa: F401
    Base.metadata.create_all(engine)
