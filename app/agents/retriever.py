"""Retriever: deterministic source-filtered vector search. No LLM."""
from typing import Any, Dict, List
from langchain_core.documents import Document

DESIGN_GUIDELINES_SOURCE = "Rules_Regulations_Policies_Guidelines.txt"
COVENANTS_SOURCE = "Covenants.txt"


def retrieve(vectorstore, topic: str, k: int = 8) -> Dict[str, Any]:
    """Search Design Guidelines for `topic`. Fall back to Covenants only if Design Guidelines returns nothing."""
    primary: List[Document] = vectorstore.similarity_search(
        topic, k=k, filter={"source": DESIGN_GUIDELINES_SOURCE}
    )
    fallback: List[Document] = []
    if not primary:
        fallback = vectorstore.similarity_search(
            topic, k=k, filter={"source": COVENANTS_SOURCE}
        )
    return {"topic": topic, "primary_chunks": primary, "fallback_chunks": fallback}


def format_chunks_for_prompt(chunks: List[Document]) -> str:
    return "\n\n".join(
        f"SOURCE: {d.metadata.get('source', '')} | {d.metadata.get('article', '')} | {d.metadata.get('section', '')}\nCONTENT: {d.page_content}"
        for d in chunks
    )
