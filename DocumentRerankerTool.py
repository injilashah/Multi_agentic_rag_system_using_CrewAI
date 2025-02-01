
from crewai.tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict
import os

from typing import Type

from markitdown import MarkItDown

from chonkie import SemanticChunker

from qdrant_client import QdrantClient



class DocumentRerankerToolInput(BaseModel):
    """Input schema for DocumentSearchTool."""
    query: str = Field(..., description="Query to search the document.")


class DocumentRerankerTool(BaseTool):
    name: str = "DocumentRerankerTool"
    description: str = "Rerank document chunks based on their scores and return the sorted documents."
    args_schema: Type[BaseModel] = DocumentRerankerToolInput

    
    
    model_config = ConfigDict(extra="allow")
    def __init__(self, file_path: str):
        """Initialize the searcher with a PDF file path and set up the Qdrant collection."""
        super().__init__()
        self.file_path = file_path
        self.client = QdrantClient(":memory:")  # For small experiments
        self._process_document()

    def _extract_text(self) -> str:
        """Extract raw text from PDF using MarkItDown."""
        
        md = MarkItDown()
        result = md.convert(self.file_path)
        return result.text_content

    def _create_chunks(self, raw_text: str) -> list:
        """Create semantic chunks from raw text."""
        chunker = SemanticChunker(
            embedding_model="minishlab/potion-base-8M",
            threshold=0.5,
            chunk_size=512,
            min_sentences=1
        )
        return chunker.chunk(raw_text)

    def _process_document(self):
        """Process the document and add chunks to Qdrant collection."""
        raw_text = self._extract_text()
        chunks = self._create_chunks(raw_text)
        
        docs = [chunk.text for chunk in chunks]
        metadata = [{"source": os.path.basename(self.file_path)} for _ in range(len(chunks))]
        ids = list(range(len(chunks)))

        self.client.add(
            collection_name="demo_collection",
            documents=docs,
            metadata=metadata,
            ids=ids
        )

    def _run(self, query: str) -> list:
        """Search the document with a query string."""
        relevant_chunks = self.client.query(
            collection_name="demo_collection",
            query_text=query
        )
        
        
        chunks_with_scores = [(chunk.document, chunk.score) for chunk in relevant_chunks]
    
    # Sort the chunks by score in descending order
        sorted_chunks = sorted(chunks_with_scores, key=lambda x: x[1], reverse=True)
    
    # Extract the sorted documents
        sorted_docs = [chunk[0] for chunk in sorted_chunks]
    
        separator = "\n___\n"
        return separator.join(sorted_docs[0:4])

# Test the implementation
def test_document_searcher():
    # Test file path
    pdf_path = r"I:/Assignments/CREW_AI/2501.01623v1.pdf"
    
    # Create instance
    searcher = DocumentRerankerTool(file_path=pdf_path)
    
    # Test search
    result = searcher._run("Estimates of Revascularization Effects on Quality of Life")
    print("Search Results:", result)

if __name__ == "__main__":
    test_document_searcher()