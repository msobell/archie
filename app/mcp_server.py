import os
import sys
from typing import Optional
from mcp.server.fastmcp import FastMCP
from app.reviewer import ARCHIEReviewer
from app.ingest import ingest_documents

# Calculate absolute paths based on this file's location
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")
CHROMA_DB_DIR = os.path.join(PROJECT_ROOT, ".chroma_db")

# Ensure environment is loaded
if not os.getenv("GEMINI_API_KEY"):
    try:
        from dotenv import load_dotenv
        # Look for .envrc in the project root
        load_dotenv(os.path.join(PROJECT_ROOT, ".envrc"))
    except ImportError:
        pass

# LangChain requires GOOGLE_API_KEY
if os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

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
    Analyzes a resident's architectural request against community guidelines.
    Provides a decision (Approval, Denial, or Conditional) with reasoning and citations.
    """
    try:
        reviewer = get_reviewer()
        result = reviewer.review_request(request)
        
        output = [
            f"### DECISION: {result['decision']}",
            f"**HITL Required:** {result['hitl_required']}",
            f"\n**Reasoning:**\n{result['reasoning']}",
            "\n**Citations:**"
        ]
        
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
def draft_review_email(request: str, homeowner_name: str) -> str:
    """
    Reviews a resident's architectural request and drafts a decision email addressed to the homeowner.
    Returns the full email body ready for board review before sending.
    """
    try:
        reviewer = get_reviewer()
        result = reviewer.review_request(request)
        email = reviewer.draft_email(result, homeowner_name)
        decision = result.get('decision', 'Unknown')
        return f"### Decision: {decision}\n\n### Draft Email:\n\n{email}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def ingest_docs() -> str:
    """
    Re-processes all text files in the docs/ folder and updates the vector database.
    Use this if the HOA rules have been updated.
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
    """Returns the status of the vector database and configuration."""
    db_exists = os.path.exists(CHROMA_DB_DIR)
    api_key_exists = bool(os.getenv("GEMINI_API_KEY"))
    
    return (
        f"Project Root: {PROJECT_ROOT}\n"
        f"Database Initialized: {db_exists} ({CHROMA_DB_DIR})\n"
        f"API Key Configured: {api_key_exists}\n"
        f"Embedding Model: gemini-embedding-001\n"
        f"Review Model: gemini-3-flash-preview"
    )

if __name__ == "__main__":
    mcp.run(transport='stdio')
