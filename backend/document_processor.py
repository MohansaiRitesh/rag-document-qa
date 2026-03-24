"""
Document Processing Module

This module handles:
1. Loading documents (PDF, DOCX, TXT)
2. Splitting documents into chunks
3. Extracting metadata

Key Concepts:
- Chunking: Breaking documents into smaller pieces for better retrieval
- Overlap: Keeping some text between chunks to maintain context
- Metadata: Storing source information for citation
"""

from typing import List, Dict
from pathlib import Path
import PyPDF2
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document as LangchainDocument


class DocumentProcessor:
    """
    Handles all document processing operations
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the processor with chunking parameters
        
        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # RecursiveCharacterTextSplitter intelligently splits on paragraphs,
        # then sentences, then words - preserving context
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]  # Priority order
        )
    
    def load_pdf(self, file_path: str) -> str:
        """
        Extract text from PDF file
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Extracted text as string
        """
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    # Add page number for citation
                    text += f"\n[Page {page_num + 1}]\n{page_text}"
            return text
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")
    
    def load_docx(self, file_path: str) -> str:
        """
        Extract text from DOCX file
        
        Args:
            file_path: Path to DOCX file
            
        Returns:
            Extracted text as string
        """
        try:
            doc = Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        except Exception as e:
            raise Exception(f"Error reading DOCX: {str(e)}")
    
    def load_txt(self, file_path: str) -> str:
        """
        Read text file
        
        Args:
            file_path: Path to TXT file
            
        Returns:
            File content as string
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            raise Exception(f"Error reading TXT: {str(e)}")
    
    def load_document(self, file_path: str) -> str:
        """
        Load document based on file extension
        
        Args:
            file_path: Path to document
            
        Returns:
            Extracted text
        """
        path = Path(file_path)
        extension = path.suffix.lower()
        
        loaders = {
            '.pdf': self.load_pdf,
            '.docx': self.load_docx,
            '.txt': self.load_txt
        }
        
        if extension not in loaders:
            raise ValueError(f"Unsupported file type: {extension}")
        
        return loaders[extension](file_path)
    
    def create_chunks(self, text: str, metadata: Dict = None) -> List[LangchainDocument]:
        """
        Split text into chunks with metadata
        
        Args:
            text: Document text to split
            metadata: Document metadata (source, title, etc.)
            
        Returns:
            List of LangchainDocument objects with chunks
        """
        if metadata is None:
            metadata = {}
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Create LangchainDocument objects with metadata
        documents = []
        for i, chunk in enumerate(chunks):
            doc = LangchainDocument(
                page_content=chunk,
                metadata={
                    **metadata,
                    "chunk_id": i,
                    "total_chunks": len(chunks)
                }
            )
            documents.append(doc)
        
        return documents
    
    def process_document(self, file_path: str) -> List[LangchainDocument]:
        """
        Complete pipeline: Load → Chunk → Add Metadata
        
        Args:
            file_path: Path to document
            
        Returns:
            List of chunked documents with metadata
        """
        # Load document
        text = self.load_document(file_path)
        
        # Extract filename for metadata
        filename = Path(file_path).name
        
        # Create metadata
        metadata = {
            "source": filename,
            "file_path": file_path
        }
        
        # Create chunks
        chunks = self.create_chunks(text, metadata)
        
        print(f"✅ Processed {filename}: {len(chunks)} chunks created")
        return chunks


# Example usage (for testing)
if __name__ == "__main__":
    processor = DocumentProcessor()
    
    # Test with a sample text
    sample_text = "This is a sample document. " * 100
    chunks = processor.create_chunks(sample_text, {"source": "test.txt"})
    
    print(f"Created {len(chunks)} chunks")
    print(f"First chunk preview: {chunks[0].page_content[:100]}...")
