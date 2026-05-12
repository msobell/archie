import os
import sys
from typing import Optional
from mcp.server.fastmcp import FastMCP
from app.reviewer import ARCHIEReviewer, REVIEW_MODEL
from app.ingest import ingest_documents
from app.mcp.memory_tools import get_related_entities, query_memory, save_memory

# Calculate absolute paths based on this file's location
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")
CHROMA_DB_DIR = os.path.join(PROJECT_ROOT, ".chroma_db")

# Ensure environment is loaded
if not os.getenv("ANTHROPIC_API_KEY"):
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(PROJECT_ROOT, ".envrc"))
    except ImportError:
        pass

# Create FastMCP server
mcp = FastMCP("Archie")

# Lazy-loaded reviewer to handle missing DB gracefully
_reviewer: Optional[ARCHIEReviewer] = None

def get_reviewer():
    global _reviewer
    if _reviewer is None:
        if not os.path.exists(CHROMA_DB_DIR):
            raise ValueError(f"Vector database not found at {CHROMA_DB_DIR}. Please run 'ingest_docs' first.")
        _reviewer = ARCHIEReviewer(persist_dir=CHROMA_DB_DIR)
    return _reviewer

@mcp.tool()
def review_request(request: str) -> str:
    """
    Use this when a homeowner has submitted an architectural / design / improvement request and you need a decision (Approval, Denial, or Conditional Approval) with rule citations — but you are NOT yet drafting a reply to the homeowner.

    Returns: decision, reasoning, HITL flag, and a list of citations (source, article, section, verbatim quote) drawn from the indexed HOA documents.

    When to use:
    - The user asks "should this be approved?", "what do the guidelines say about X?", "review this request", or pastes a request and wants an analysis.
    - As a precursor when the user wants to discuss the decision before any email is written.

    Do NOT use when:
    - The user explicitly wants an email drafted — call `draft_review_email` instead (it runs the review internally; calling both is redundant).
    - The user is asking a general lookup question with no specific request — answer from `query_memory` or by reading docs directly.

    Argument: `request` is the homeowner's request in natural language.
    """
    try:
        reviewer = get_reviewer()
        result = reviewer.review_request(request)

        output = [
            f"### DECISION: {result['decision']}",
            f"**HITL Required:** {result['hitl_required']}",
            f"\n**Reasoning:**\n{result['reasoning']}",
        ]

        failure_notes = result.get("failure_notes", [])
        if failure_notes:
            output.append("\n**Review Notes (manual attention required):**")
            output.extend(f"- {n}" for n in failure_notes)

        output.append("\n**Citations:**")
        for cite in result['citations']:
            source = cite.get('source', 'Unknown')
            article = cite.get('article', '').replace('Unknown', '').strip()
            section = cite.get('section', '').strip()
            quote = cite.get('quote', '').strip()
            line = f"- {source}: {article} {section}".strip().replace("  ", " ")
            if quote:
                line += f'\n  "{quote}"'
            output.append(line)

        return "\n".join(output)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def draft_review_email(request: str, homeowner_name: Optional[str] = None) -> str:
    """
    Use this when the user wants a board-ready email reply to a homeowner about their architectural request. This is the one-shot tool: it runs the full review internally AND drafts the email — do NOT call `review_request` first.

    Returns: the decision header followed by a formatted email (Subject line, greeting, numbered citation items, closing).

    When to use:
    - The user says "draft an email", "write a reply", "respond to [homeowner]", or supplies a homeowner name alongside a request.
    - Any time the deliverable is text to send back to a homeowner.

    Do NOT use when:
    - The user only wants a decision or rule analysis with no email — call `review_request` instead.
    - The homeowner's name is unknown — ask for it first; it is required for the salutation.

    Arguments: `request` is the homeowner's request; `homeowner_name` is who the email is addressed to (e.g. "Chad and Jaclyn").
    """
    try:
        reviewer = get_reviewer()
        result = reviewer.review_request(request)

        # Caller-supplied name wins; otherwise use the parser's extraction; otherwise generic.
        name = homeowner_name or result.get("homeowner_name") or "Homeowner"

        email = reviewer.draft_email(result, name)
        decision = result.get('decision', 'Unknown')

        sections = [f"### Decision: {decision}"]
        failure_notes = result.get("failure_notes", [])
        if failure_notes:
            sections.append("### Review Notes (manual attention required):\n" + "\n".join(f"- {n}" for n in failure_notes))
        sections.append(f"### Draft Email (addressed to: {name}):\n\n{email}")
        return "\n\n".join(sections)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def ingest_docs() -> str:
    """
    Use this ONLY when the HOA's source documents have changed and the vector index needs to be rebuilt — e.g. the user says "I added a new bylaw", "re-index the docs", "I updated the design guidelines", or reviews start returning stale citations.

    Re-processes every .txt file in `docs/` and rewrites the ChromaDB index in place. Safe to re-run, but unnecessary on a normal review and slow (seconds to minutes depending on corpus size).

    Do NOT use as a troubleshooting step for "no citations found" unless the user confirms the docs were changed — call `server_status` first to check whether the DB exists at all.
    """
    try:
        ingest_documents(DOCS_DIR, CHROMA_DB_DIR)
        global _reviewer
        _reviewer = None # Reset reviewer to pick up new DB
        return f"Successfully ingested documents from {DOCS_DIR} into {CHROMA_DB_DIR}."
    except Exception as e:
        return f"Ingestion failed: {str(e)}"

@mcp.tool()
def server_status() -> str:
    """
    Use this for diagnostics — when a tool call has failed, when the user asks "is Archie set up?", or before the first review in a fresh environment to confirm the index and API key are in place.

    Returns: project root path, whether the vector DB directory exists, whether ANTHROPIC_API_KEY is configured, the embedding model, and the review model.

    Do NOT use as part of a normal review flow — it does not perform any review work.
    """
    db_exists = os.path.exists(CHROMA_DB_DIR)
    api_key_exists = bool(os.getenv("ANTHROPIC_API_KEY"))

    return (
        f"Project Root: {PROJECT_ROOT}\n"
        f"Database Initialized: {db_exists} ({CHROMA_DB_DIR})\n"
        f"API Key Configured: {api_key_exists}\n"
        f"Embedding Model: all-MiniLM-L6-v2 (local)\n"
        f"Review Model: {REVIEW_MODEL}"
    )

mcp.tool()(save_memory)
mcp.tool()(query_memory)
mcp.tool()(get_related_entities)

if __name__ == "__main__":
    mcp.run(transport='stdio')
