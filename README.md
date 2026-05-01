# archie

AI-powered HOA architectural review assistant. Reviews resident design requests against community bylaws and guidelines, returning a decision with citations and a draft response email.

## Features

- **Automatic OCR**: Scanned PDFs are converted to text using Claude Vision (with Tesseract fallback) before indexing.
- **RAG-based review**: Chunks and indexes HOA documents into a local ChromaDB vector store, then retrieves relevant rules at query time.
- **Decision + citations**: Returns `Approval`, `Denial`, or `Conditional Approval` with verbatim quotes from the matched guideline sections.
- **Draft email**: Generates a short, board-ready response email for the homeowner.
- **MCP Server**: Exposes all tools for use with Claude Desktop or other MCP clients.

## Setup

Requires Python 3.10+. This project uses `direnv` for environment management.

```bash
git clone <repo-url>
cd archie
direnv allow
pip install -e .
```

Create a `.envrc` file with your Anthropic API key:

```bash
export ANTHROPIC_API_KEY=your-anthropic-key
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
archie review "I want to add a wood pergola in the backyard." --draft-email "Brandon"
```

Prints a short, board-ready email to the homeowner after the review output.

## MCP Server

Archie can be used as an MCP tool server with Claude Desktop or any MCP-compatible client:

```bash
python -m app.mcp_server
```

Tools exposed: `review_request`, `draft_review_email`, `ingest_docs`, `server_status`.
