from __future__ import annotations
import json
import logging
from typing import List, Optional
from sqlalchemy import text
from app.db import SessionLocal, create_tables
from app.memory.db_setup import ensure_virtual_tables
from app.memory.embeddings import get_embedding
from app.memory.search import hybrid_search
from app.models.memory import KnowledgeEdge, Memory

logger = logging.getLogger(__name__)

_initialized = False


def _get_session():
    global _initialized
    if not _initialized:
        create_tables()
        _initialized = True
    session = SessionLocal()
    ensure_virtual_tables(session)
    return session


def save_memory(content: str, entities: List[str], metadata: Optional[dict] = None) -> str:
    """
    Save a note to persistent memory and link it to named entities (homeowners, addresses, request types, etc.).
    Returns the saved memory ID.
    """
    session = _get_session()
    try:
        memory = Memory(content=content, metadata_json=json.dumps(metadata) if metadata else None)
        session.add(memory)
        session.flush()

        session.execute(
            text("INSERT INTO memories_fts(content, id) VALUES(:c, :id)"),
            {"c": content, "id": memory.id},
        )

        try:
            import sqlite_vec
            blob = sqlite_vec.serialize_float32(get_embedding(content))
            session.execute(
                text("INSERT INTO memories_vec(id, embedding) VALUES(:id, :emb)"),
                {"id": memory.id, "emb": blob},
            )
        except Exception as e:
            logger.warning(f"Vector index skipped: {e}")

        for name in entities:
            entity = session.query(Memory).filter(Memory.content == name.lower()).first()
            if not entity:
                entity = Memory(content=name.lower(), metadata_json=json.dumps({"type": "entity"}))
                session.add(entity)
                session.flush()
                session.execute(
                    text("INSERT INTO memories_fts(content, id) VALUES(:c, :id)"),
                    {"c": name.lower(), "id": entity.id},
                )
            session.add(KnowledgeEdge(
                source_id=memory.id,
                target_id=entity.id,
                relationship_type="MENTIONS",
            ))

        session.commit()
        return f"Saved memory #{memory.id}, linked to {len(entities)} entities."
    except Exception as e:
        session.rollback()
        return f"Error saving memory: {e}"
    finally:
        session.close()


def query_memory(query: str, n_results: int = 5) -> str:
    """
    Search saved memories using hybrid keyword + semantic search.
    Useful for looking up prior decisions, homeowner history, or recurring request patterns.
    """
    session = _get_session()
    try:
        results = hybrid_search(session, query, n_results)
        if not results:
            return "No matching memories found."
        lines = ["### Memory Results:"]
        for i, m in enumerate(results):
            lines.append(f"{i+1}. [#{m.id}] {m.content}")
            entities = [e.target.content for e in m.out_edges if e.relationship_type == "MENTIONS"]
            if entities:
                lines.append(f"   Entities: {', '.join(entities)}")
        return "\n".join(lines)
    except Exception as e:
        return f"Error querying memory: {e}"
    finally:
        session.close()


def get_related_entities(entity_name: str) -> str:
    """
    Look up all memories and entities linked to a name (e.g. a homeowner or address).
    Walks the knowledge graph to surface prior requests and decisions.
    """
    session = _get_session()
    try:
        entity = session.query(Memory).filter(Memory.content == entity_name.lower()).first()
        if not entity:
            results = hybrid_search(session, entity_name, n_results=1)
            if not results:
                return f"'{entity_name}' not found in memory."
            entity = results[0]
            lines = [f"Closest match: '{entity.content}'"]
        else:
            lines = [f"Related to '{entity_name}':"]

        mentions = [e.source.content for e in entity.in_edges if e.relationship_type == "MENTIONS"]
        if mentions:
            lines.append("\n**Memories mentioning this:**")
            lines.extend(f"- {m}" for m in mentions)

        related = [e.target.content for e in entity.out_edges if e.relationship_type == "MENTIONS"]
        if related:
            lines.append("\n**Also linked to:**")
            lines.extend(f"- {r}" for r in related)

        return "\n".join(lines) if len(lines) > 1 else f"No relations found for '{entity_name}'."
    except Exception as e:
        return f"Error: {e}"
    finally:
        session.close()
