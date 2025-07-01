# LlamaParse integration
# src/parsers/llama_parser.py

"""
LlamaParse integration for high-quality PDF parsing with table and figure extraction.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import asyncio
import nest_asyncio

# Apply nest_asyncio to handle nested event loops
nest_asyncio.apply()

try:
    from llama_parse import LlamaParse
except ImportError as e:
    raise ImportError(
        "LlamaParse not found. Please install: pip install llama-parse"
    ) from e

from config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class ParsedDocument:
    """Container for parsed document data."""
    content: str
    metadata: Dict[str, Any]
    pages: List[Dict[str, Any]]
    file_path: str

class LlamaParserWrapper:
    """
    Wrapper for LlamaParse with enhanced functionality for chunking pipeline.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        result_type: str = "markdown",
        verbose: bool = True,
        partition_pages: Optional[int] = None,
    ):
        """
        Initialize LlamaParse wrapper.
        
        Args:
            api_key: LlamaParse API key (defaults to settings)
            result_type: Output format ("markdown" or "text")
            verbose: Enable verbose logging
            partition_pages: Split large documents into smaller jobs
        """
        self.api_key = api_key or settings.LLAMA_CLOUD_API_KEY
        self.result_type = result_type
        self.verbose = verbose
        self.partition_pages = partition_pages or settings.LLAMAPARSE_PARTITION_PAGES
        
        if not self.api_key:
            raise ValueError("LLAMA_CLOUD_API_KEY is required")
        
        # Initialize the parser
        self.parser = LlamaParse(
            api_key=self.api_key,
            result_type=self.result_type,
            verbose=self.verbose,
            partition_pages=self.partition_pages
        )
        
        logger.info(f"LlamaParse initialized with result_type={result_type}")
    
    def parse_document(self, file_path: str) -> ParsedDocument:
        """
        Parse a PDF document using LlamaParse.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            ParsedDocument with extracted content and metadata
        """
        try:
            logger.info(f"Starting to parse document: {file_path}")
            
            # Parse the document
            documents = self.parser.load_data(file_path)
            
            if not documents:
                raise ValueError(f"No documents parsed from {file_path}")
            
            # Combine all document content
            combined_content = ""
            pages = []
            metadata = {}
            
            for i, doc in enumerate(documents):
                # Extract content
                combined_content += doc.text + "\n\n"
                
                # Extract page information
                page_info = {
                    "page_number": i + 1,
                    "content": doc.text,
                    "metadata": doc.metadata or {}
                }
                pages.append(page_info)
                
                # Merge metadata
                if doc.metadata:
                    metadata.update(doc.metadata)
            
            # Add parsing metadata
            metadata.update({
                "total_pages": len(documents),
                "parser": "LlamaParse",
                "result_type": self.result_type,
                "file_path": file_path
            })
            
            parsed_doc = ParsedDocument(
                content=combined_content.strip(),
                metadata=metadata,
                pages=pages,
                file_path=file_path
            )
            
            logger.info(
                f"Successfully parsed {file_path}: "
                f"{len(documents)} pages, {len(combined_content)} characters"
            )
            
            return parsed_doc
            
        except Exception as e:
            logger.error(f"Error parsing document {file_path}: {str(e)}")
            raise
    
    async def parse_document_async(self, file_path: str) -> ParsedDocument:
        """
        Asynchronously parse a PDF document using LlamaParse.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            ParsedDocument with extracted content and metadata
        """
        try:
            logger.info(f"Starting async parse of document: {file_path}")
            
            # Parse the document asynchronously
            documents = await self.parser.aload_data(file_path)
            
            if not documents:
                raise ValueError(f"No documents parsed from {file_path}")
            
            # Process documents same as sync version
            combined_content = ""
            pages = []
            metadata = {}
            
            for i, doc in enumerate(documents):
                combined_content += doc.text + "\n\n"
                
                page_info = {
                    "page_number": i + 1,
                    "content": doc.text,
                    "metadata": doc.metadata or {}
                }
                pages.append(page_info)
                
                if doc.metadata:
                    metadata.update(doc.metadata)
            
            metadata.update({
                "total_pages": len(documents),
                "parser": "LlamaParse",
                "result_type": self.result_type,
                "file_path": file_path
            })
            
            parsed_doc = ParsedDocument(
                content=combined_content.strip(),
                metadata=metadata,
                pages=pages,
                file_path=file_path
            )
            
            logger.info(
                f"Successfully parsed {file_path} async: "
                f"{len(documents)} pages, {len(combined_content)} characters"
            )
            
            return parsed_doc
            
        except Exception as e:
            logger.error(f"Error parsing document {file_path} async: {str(e)}")
            raise
    
    def parse_multiple_documents(self, file_paths: List[str]) -> List[ParsedDocument]:
        """
        Parse multiple PDF documents.
        
        Args:
            file_paths: List of PDF file paths
            
        Returns:
            List of ParsedDocument objects
        """
        try:
            logger.info(f"Parsing {len(file_paths)} documents")
            
            # Parse all documents
            documents = self.parser.load_data(file_paths)
            
            # Group documents by source file
            parsed_docs = []
            docs_per_file = len(documents) // len(file_paths)
            
            for i, file_path in enumerate(file_paths):
                start_idx = i * docs_per_file
                end_idx = start_idx + docs_per_file if i < len(file_paths) - 1 else len(documents)
                file_docs = documents[start_idx:end_idx]
                
                combined_content = ""
                pages = []
                metadata = {}
                
                for j, doc in enumerate(file_docs):
                    combined_content += doc.text + "\n\n"
                    
                    page_info = {
                        "page_number": j + 1,
                        "content": doc.text,
                        "metadata": doc.metadata or {}
                    }
                    pages.append(page_info)
                    
                    if doc.metadata:
                        metadata.update(doc.metadata)
                
                metadata.update({
                    "total_pages": len(file_docs),
                    "parser": "LlamaParse",
                    "result_type": self.result_type,
                    "file_path": file_path
                })
                
                parsed_doc = ParsedDocument(
                    content=combined_content.strip(),
                    metadata=metadata,
                    pages=pages,
                    file_path=file_path
                )
                
                parsed_docs.append(parsed_doc)
            
            logger.info(f"Successfully parsed {len(file_paths)} documents")
            return parsed_docs
            
        except Exception as e:
            logger.error(f"Error parsing multiple documents: {str(e)}")
            raise
    
    def get_page_batches(self, parsed_doc: ParsedDocument, batch_size: int = 4) -> List[List[Dict[str, Any]]]:
        """
        Split document pages into batches for processing.
        
        Args:
            parsed_doc: Parsed document
            batch_size: Number of pages per batch
            
        Returns:
            List of page batches
        """
        pages = parsed_doc.pages
        batches = []
        
        for i in range(0, len(pages), batch_size):
            batch = pages[i:i + batch_size]
            batches.append(batch)
        
        logger.info(f"Split {len(pages)} pages into {len(batches)} batches")
        return batches
    
    def get_batch_content(self, batch: List[Dict[str, Any]]) -> str:
        """
        Combine content from a batch of pages.
        
        Args:
            batch: List of page dictionaries
            
        Returns:
            Combined content string
        """
        content_parts = []
        
        for page in batch:
            page_num = page.get("page_number", "Unknown")
            page_content = page.get("content", "")
            
            content_parts.append(f"=== PAGE {page_num} ===\n{page_content}\n")
        
        return "\n".join(content_parts)

# Convenience function for quick parsing
def parse_pdf(file_path: str, **kwargs) -> ParsedDocument:
    """
    Quick function to parse a PDF with default settings.
    
    Args:
        file_path: Path to PDF file
        **kwargs: Additional arguments for LlamaParserWrapper
        
    Returns:
        ParsedDocument
    """
    parser = LlamaParserWrapper(**kwargs)
    return parser.parse_document(file_path)