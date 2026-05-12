import os
import re
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from app.utils import sanitize_path

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

class HOAStructureSplitter:
    """
    Splits HOA documents by Article and Section headers while preserving context.
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Pattern to match "ARTICLE I", "Section 1.1", or just "2.1 ", etc.
        self.header_pattern = re.compile(r'^(ARTICLE\s+[IVXLC]+|Section\s+\d+(\.\d+)*|\d+(\.\d+)+\s+)', re.IGNORECASE | re.MULTILINE)

    def _make_chunk(self, text: str, source_name: str, article: str, section: str) -> Document:
        return Document(
            page_content=text.strip(),
            metadata={"source": source_name, "article": article, "section": section},
        )

    def split_text(self, text: str, source_name: str) -> List[Document]:
        chunks = []
        current_article = "Unknown"
        current_section = "General"
        section_header_line = ""

        lines = text.split('\n')
        current_chunk_lines = []

        def flush(carry_header: bool = False):
            content = "\n".join(current_chunk_lines).strip()
            if content:
                chunks.append(self._make_chunk(content, source_name, current_article, current_section))
            current_chunk_lines.clear()
            if carry_header and section_header_line:
                # Begin the next chunk with the section header so it's never orphaned
                current_chunk_lines.append(section_header_line)

        for line in lines:
            header_match = self.header_pattern.match(line.strip())
            if header_match:
                flush()
                header_text = header_match.group(0).upper()
                if "ARTICLE" in header_text:
                    current_article = line.strip()
                    current_section = "Header"
                    section_header_line = ""
                else:
                    current_section = line.strip()
                    section_header_line = line
                current_chunk_lines.append(line)
            else:
                current_chunk_lines.append(line)
                if sum(len(l) for l in current_chunk_lines) > self.chunk_size:
                    # Keep last `chunk_overlap` chars worth of lines as overlap
                    flush(carry_header=True)
                    overlap_lines = []
                    overlap_len = 0
                    for l in reversed(current_chunk_lines):
                        if overlap_len + len(l) > self.chunk_overlap:
                            break
                        overlap_lines.insert(0, l)
                        overlap_len += len(l)
                    current_chunk_lines.extend(overlap_lines)

        flush()
        return chunks

def ingest_documents(docs_dir: str, persist_dir: str):
    """
    Reads .txt files from docs_dir, chunks them, and stores in ChromaDB.
    """
    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir)
        
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    print(f"Using embedding model: {EMBEDDING_MODEL}")
    splitter = HOAStructureSplitter()
    all_documents = []
    
    for filename in os.listdir(docs_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(docs_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                docs = splitter.split_text(text, filename)
                all_documents.extend(docs)
                
    # Filter out empty documents
    all_documents = [doc for doc in all_documents if doc.page_content.strip()]
    
    if not all_documents:
        print("No documents found to ingest.")
        return

    print(f"Ingesting {len(all_documents)} chunks into ChromaDB at {persist_dir}...")
    Chroma.from_documents(
        documents=all_documents,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    print("Ingestion complete.")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    ingest_documents("docs", ".chroma_db")
