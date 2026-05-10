import logging
from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def ensure_virtual_tables(session: Session):
    session.execute(text("""
        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
            content,
            id UNINDEXED,
            tokenize="trigram"
        )
    """))

    session.execute(text("""
        CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING vec0(
            id INTEGER PRIMARY KEY,
            embedding float[384]
        )
    """))

    session.commit()
    logger.debug("Memory virtual tables ensured.")
