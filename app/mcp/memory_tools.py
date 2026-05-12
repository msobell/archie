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
    Use this AFTER a review or conversation to persist anything worth recalling in a future session — a final decision, a board note, a homeowner's prior history, a recurring request pattern, or a clarification the user wants remembered. Memory persists across Claude Desktop sessions.

    When to use:
    - The user says "remember that...", "save this", "make a note", or finalizes a decision worth tracking.
    - After `draft_review_email` produces a decision the user accepts — save the outcome linked to the homeowner.

    Do NOT use for:
    - Ephemeral within-conversation state.
    - Content already present in the HOA documents (those are indexed separately).

    Arguments:
    - `content`: the note as a complete sentence or short paragraph.
    - `entities`: list of names to index this memory under — homeowner names, addresses, request types (e.g. ["Chad Bloor", "123 Canyon Dr", "basketball court"]). At least one is strongly recommended so the memory is later findable via `get_related_entities`.
    - `metadata`: optional dict for structured fields (decision, date, etc.).

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
    Use this BEFORE reviewing a request to surface relevant prior context — past decisions on similar requests, prior interactions with the same homeowner, or recurring patterns the board has flagged. Searches via hybrid keyword + semantic ranking.

    When to use:
    - At the start of a new review, to check whether this homeowner or this kind of request has come up before.
    - When the user asks "have we seen this before?", "what did we decide last time?", or "any history on X?".
    - For free-text searches across saved notes.

    Prefer `get_related_entities` instead when:
    - You already know an exact homeowner name or address and want the full graph of memories linked to them.

    Arguments:
    - `query`: free-text search string.
    - `n_results`: max results to return (default 5).
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
    Use this when you already have a specific homeowner name, address, or request-type label and want the complete history linked to it. Walks the knowledge graph and returns every memory mentioning the entity plus every other entity those memories link to.

    When to use:
    - The user names a specific homeowner or address ("what do we have on Chad Bloor?", "history for 123 Canyon Dr").
    - As a follow-up to `query_memory` once you've identified the right entity.

    Prefer `query_memory` instead when:
    - The search is conceptual or topical rather than tied to a specific named entity.

    Argument: `entity_name` — exact homeowner name, address, or label as previously saved. Case-insensitive. If no exact match exists, falls back to the closest memory.
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
