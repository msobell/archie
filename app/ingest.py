import os
import re
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
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
        self.header_pattern = re.compile(r'^(ARTICLE\s+[IVXLC]+|Section\s+\d+(\.\d+)*|\d+\.\d+\s+)', re.IGNORECASE | re.MULTILINE)

    def split_text(self, text: str, source_name: str) -> List[Document]:
        chunks = []
        current_article = "Unknown"
        current_section = "General"
        
        # Split by lines to process headers
        lines = text.split('\n')
        current_chunk_text = ""
        
        for line in lines:
            header_match = self.header_pattern.match(line.strip())
            if header_match:
                # If we have a header, save the previous chunk if it's not empty
                if current_chunk_text.strip():
                    chunks.append(Document(
                        page_content=current_chunk_text.strip(),
                        metadata={
                            "source": source_name,
                            "article": current_article,
                            "section": current_section
                        }
                    ))
                
                # Update context
                header_text = header_match.group(0).upper()
                if "ARTICLE" in header_text:
                    current_article = line.strip()
                    current_section = "Header"
                else:
                    current_section = line.strip()
                
                current_chunk_text = line + "\n"
            else:
                current_chunk_text += line + "\n"
                
                # Fallback: if chunk gets too big, split it
                if len(current_chunk_text) > self.chunk_size:
                    chunks.append(Document(
                        page_content=current_chunk_text.strip(),
                        metadata={
                            "source": source_name,
                            "article": current_article,
                            "section": current_section
                        }
                    ))
                    current_chunk_text = "" # Simplified overlap for now

        # Add the final chunk
        if current_chunk_text.strip():
            chunks.append(Document(
                page_content=current_chunk_text.strip(),
                metadata={
                    "source": source_name,
                    "article": current_article,
                    "section": current_section
                }
            ))
            
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
