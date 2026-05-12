# archie

AI-powered HOA architectural review assistant. Reviews resident design requests against community bylaws and guidelines, returning a decision with citations and a draft response email.

## Features

- **Automatic OCR**: Scanned PDFs are converted to text using Claude Vision (with Tesseract fallback) before indexing.
- **RAG-based review**: Chunks and indexes HOA documents into a local ChromaDB vector store, then retrieves relevant rules at query time.
- **Multi-agent review pipeline**: Parser → Retriever → Evaluator → Synthesizer, so each item in a request gets its own focused retrieval and citation pass.
- **Decision + citations**: Returns `Approval`, `Denial`, or `Conditional Approval` with verbatim quotes from the matched guideline sections.
- **Draft email**: Generates a board-ready response email for the homeowner with each rule quoted verbatim.
- **Memory layer**: Persists notes, prior decisions, and homeowner history across sessions with hybrid keyword + semantic search.
- **MCP Server**: Exposes all tools for use with Claude Desktop or other MCP clients.

## How the review works

A single request often bundles several distinct improvements ("basketball court, walkway, drainage changes"). Reviewing them in one pass tends to collapse into generic Covenants citations. Archie splits the work across four stages:

1. **Parser** (Haiku) — extracts the homeowner's distinct improvement items, the global regulatory concerns those items raise (drainage, easements, impervious surface, neighbor approval, etc.), and the homeowner's name if it appears in the request.
2. **Retriever** — for each item and concern, runs a source-filtered vector search against the Design Guidelines. Falls back to the Covenants only when no Design Guidelines section covers the topic.
3. **Evaluator** (Sonnet, one call per topic) — selects the most specific applicable rule from the retrieved chunks, copies a verbatim quote, writes a concrete "ask" describing what the homeowner must provide, and assigns a per-topic outcome (`compliant`, `violates`, `needs_info`, or `not_addressed`).
4. **Synthesizer** — dedupes citations across topics, rolls up the per-topic outcomes into an overall decision, and emits `failure_notes` for any topic that returned no retrieval or no specific rule. Always flags HITL.

The email renderer is a pure formatter over the synthesizer's output — one numbered item per citation, rule text quoted verbatim.

## Setup

Requires Python 3.10+. This project uses `direnv` for environment management.

```bash
git clone <repo-url>
cd archie
direnv allow
pip install -e .
```

Copy `.envrc.example` to `.envrc` and fill in your values:

```bash
cp .envrc.example .envrc
```

```bash
export ANTHROPIC_API_KEY=your-anthropic-api-key
```

Tesseract must be installed separately if you want the OCR fallback:

```bash
brew install tesseract   # macOS
```

## Workflow

### 1. Add source documents

Drop scanned HOA document PDFs into `docs/source/`:

```
docs/source/
  bylaws.pdf
  design-guidelines.pdf
```

### 2. Ingest

OCRs any new PDFs using Claude Vision, writes text files to `docs/`, then indexes them into the local vector database:

```bash
archie ingest
```

Re-running `ingest` is safe — already-converted PDFs are skipped.

### 3. Review a request

```bash
archie review "I want to paint my front door red and add a wood pergola in the backyard."
```

Output includes a decision, reasoning, and cited excerpts from the guidelines.

### 4. Draft a response email

```bash
archie review "I want to add a wood pergola in the backyard." --draft-email "Bob"
```

Prints a short, board-ready email to the homeowner after the review output.

## MCP Server

Archie can be used as an MCP tool server with Claude Desktop or any MCP-compatible client:

```bash
python -m app.mcp_server
```

### Review tools
| Tool | Description |
|---|---|
| `review_request` | Analyze a request against community guidelines and return a decision with citations |
| `draft_review_email` | Review a request and produce a board-ready email to the homeowner. `homeowner_name` is optional — the Parser will extract it from the request when possible |
| `ingest_docs` | Re-index all documents in `docs/` into the vector database |
| `server_status` | Check database and API key configuration |

### Memory tools
| Tool | Description |
|---|---|
| `save_memory` | Save a note and link it to named entities (homeowners, addresses, request types) |
| `query_memory` | Hybrid keyword + semantic search across saved memories |
| `get_related_entities` | Walk the knowledge graph for all memories linked to a name or address |

Memory is stored in `.memory.db` (SQLite) and persists across sessions. Vector search uses the same `all-MiniLM-L6-v2` model as the document index.
