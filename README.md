# archie

**archie** is an automated HOA design review assistant. It reviews resident design requests against community bylaws and architectural guidelines to provide preliminary approval or denial recommendations.

## Features

- **Automatic OCR**: Scanned PDFs are converted to text using Claude Vision before indexing.
- **Guideline Analysis**: Parses and indexes HOA bylaws and architectural guidelines using RAG (ChromaDB).
- **Request Review**: Evaluates resident submissions for compliance and returns a decision with citations.
- **MCP Server**: Exposes review and ingestion tools for use with Claude Desktop or other MCP clients.

## Setup

This project uses `direnv` for environment management.

```bash
git clone <repo-url>
cd archie
direnv allow
pip install -e .
```

Create a `.envrc` file with your API keys:

```bash
export GEMINI_API_KEY=your-gemini-key
export ANTHROPIC_API_KEY=your-anthropic-key
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

OCRs any new PDFs in `docs/source/` using Claude Vision, writes text files to `docs/`, then indexes them into the vector database:

```bash
archie ingest
```

Re-running `ingest` is safe — already-converted PDFs are skipped.

### 3. Review a request

```bash
archie review "I want to paint my front door red and add a wood pergola in the backyard."
```

Output includes a decision (`Approval`, `Denial`, or `Conditional Approval`), reasoning, and exact citations.

## MCP Server

Archie can be used as an MCP tool server:

```bash
python -m app.mcp_server
```

Tools exposed: `review_request`, `ingest_docs`, `server_status`.
