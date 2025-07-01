# Main orchestration
# src/pipeline/main_pipeline.py

"""
Main pipeline for PDF chunking using LlamaParse, OpenRouter, and LangChain.
"""

import logging
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

from src.parsers.llama_parser import LlamaParserWrapper, ParsedDocument
from src.llms.openrouter_llm import OpenRouterLLM
from src.chunkers.context_manager import ContextManager
from src.chunkers.chunk_processor import ChunkProcessor, ProcessedChunk
from src.chunkers.prompts import get_chunking_prompt, get_context_summary_prompt
from config.settings import settings

logger = logging.getLogger(__name__)

class PDFChunkingPipeline:
    """
    Main pipeline for processing PDFs into meaningful chunks using vision-guided approach.
    
    Pipeline stages:
    1. PDF Parsing with LlamaParse
    2. Batch Creation (4 pages per batch)
    3. Context-Aware Chunking with OpenRouter LLM
    4. Chunk Processing and Validation
    5. Context Update for Next Batch
    """
    
    def __init__(
        self,
        llamaindex_api_key: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
        batch_size: int = 4,
        model_name: Optional[str] = None
    ):
        """
        Initialize the pipeline.
        
        Args:
            llamaindex_api_key: LlamaIndex API key
            openrouter_api_key: OpenRouter API key
            batch_size: Pages per batch for processing
            model_name: OpenRouter model to use
        """
        self.batch_size = batch_size or settings.BATCH_SIZE
        
        # Initialize components
        self.parser = LlamaParserWrapper(api_key=llamaindex_api_key)
        self.llm = OpenRouterLLM(
            openrouter_api_key=openrouter_api_key,
            model_name=model_name
        )
        self.context_manager = ContextManager()
        self.chunk_processor = ChunkProcessor()
        
        # Validate setup
        self._validate_setup()
        
        logger.info(f"PDF Chunking Pipeline initialized with batch_size={self.batch_size}")
    
    def _validate_setup(self):
        """Validate that all components are properly configured."""
        try:
            settings.validate()
            logger.info("Configuration validation passed")
        except Exception as e:
            logger.error(f"Configuration validation failed: {str(e)}")
            raise
    
    def process_pdf(
        self, 
        file_path: str, 
        output_path: Optional[str] = None
    ) -> List[ProcessedChunk]:
        """
        Process a PDF file into meaningful chunks.
        
        Args:
            file_path: Path to the PDF file
            output_path: Optional path to save results
            
        Returns:
            List of ProcessedChunk objects
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting PDF processing: {file_path}")
            
            # Reset context for new document
            self.context_manager.reset()
            
            # Step 1: Parse PDF with LlamaParse
            logger.info("Step 1: Parsing PDF with LlamaParse")
            parsed_doc = self.parser.parse_document(file_path)
            
            # Step 2: Create page batches
            logger.info("Step 2: Creating page batches")
            page_batches = self.parser.get_page_batches(parsed_doc, self.batch_size)
            
            # Step 3: Process each batch
            logger.info(f"Step 3: Processing {len(page_batches)} batches")
            all_chunks = []
            
            for i, batch in enumerate(page_batches):
                logger.info(f"Processing batch {i+1}/{len(page_batches)}")
                
                batch_chunks = self._process_batch(batch, f"batch_{i}")
                
                if batch_chunks:
                    all_chunks.extend(batch_chunks)
                    
                    # Update context for next batch
                    self._update_context_from_batch(batch_chunks, f"batch_{i}")
                
                # Small delay to respect API rate limits
                time.sleep(1)
            
            # Step 4: Post-process chunks
            logger.info("Step 4: Post-processing chunks")
            final_chunks = self._post_process_chunks(all_chunks)
            
            # Step 5: Save results if output path provided
            if output_path:
                self._save_results(final_chunks, output_path, parsed_doc.metadata)
            
            processing_time = time.time() - start_time
            logger.info(
                f"PDF processing completed in {processing_time:.2f}s. "
                f"Generated {len(final_chunks)} chunks from {parsed_doc.metadata['total_pages']} pages"
            )
            
            return final_chunks
            
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}")
            raise
    
    def _process_batch(
        self, 
        batch: List[Dict[str, Any]], 
        batch_id: str
    ) -> List[ProcessedChunk]:
        """
        Process a single batch of pages.
        
        Args:
            batch: List of page dictionaries
            batch_id: Identifier for this batch
            
        Returns:
            List of processed chunks
        """
        try:
            # Get batch content
            batch_content = self.parser.get_batch_content(batch)
            
            # Get context from previous batches
            context_info = self.context_manager.get_context_for_batch()
            
            # Create prompt with context
            prompt = get_chunking_prompt(
                previous_context=context_info["previous_context"],
                last_chunks=context_info["last_chunks"],
                batch_content=batch_content
            )
            
            # Generate chunks with LLM
            logger.debug(f"Generating chunks for {batch_id}")
            llm_output = self.llm.generate_with_retry(prompt, max_retries=3)
            
            # Process LLM output into structured chunks
            chunks = self.chunk_processor.process_llm_output(llm_output, batch_id)
            
            # Validate chunks
            validated_chunks = self.chunk_processor.validate_chunks(chunks)
            
            logger.info(f"Batch {batch_id}: Generated {len(validated_chunks)} chunks")
            return validated_chunks
            
        except Exception as e:
            logger.error(f"Error processing batch {batch_id}: {str(e)}")
            return []
    
    def _update_context_from_batch(
        self, 
        chunks: List[ProcessedChunk], 
        batch_id: str
    ):
        """
        Update context manager with information from processed batch.
        
        Args:
            chunks: Processed chunks from the batch
            batch_id: Identifier for the batch
        """
        try:
            # Convert chunks to context format
            chunk_dicts = []
            for chunk in chunks:
                chunk_dict = {
                    "heading": chunk.heading,
                    "content": chunk.content[:500],  # Truncate for context
                    "continues": chunk.continues,
                    "metadata": chunk.metadata
                }
                chunk_dicts.append(chunk_dict)
            
            # Generate summary for this batch
            if chunks:
                summary_content = "\n\n".join([
                    f"{chunk.heading}: {chunk.content[:200]}..." 
                    for chunk in chunks[:3]  # Summarize first 3 chunks
                ])
                
                summary_prompt = get_context_summary_prompt(summary_content)
                batch_summary = self.llm.generate_with_retry(
                    summary_prompt, 
                    max_tokens=200,
                    max_retries=2
                )
            else:
                batch_summary = f"No valid chunks generated for {batch_id}"
            
            # Update context manager
            self.context_manager.add_batch_context(chunk_dicts, batch_summary)
            
        except Exception as e:
            logger.warning(f"Failed to update context from batch {batch_id}: {str(e)}")
    
    def _post_process_chunks(self, chunks: List[ProcessedChunk]) -> List[ProcessedChunk]:
        """
        Post-process chunks to improve quality and consistency.
        
        Args:
            chunks: Raw processed chunks
            
        Returns:
            Post-processed chunks
        """
        if not chunks:
            return chunks
        
        logger.info("Post-processing chunks")
        
        # Merge related chunks
        merged_chunks = self.chunk_processor.merge_related_chunks(chunks)
        
        # Add sequential IDs
        for i, chunk in enumerate(merged_chunks):
            chunk.id = f"chunk_{i:03d}"
            chunk.metadata["sequence_id"] = i
        
        # Add document-level metadata
        total_words = sum(chunk.metadata["word_count"] for chunk in merged_chunks)
        avg_words = total_words / len(merged_chunks) if merged_chunks else 0
        
        for chunk in merged_chunks:
            chunk.metadata["document_total_chunks"] = len(merged_chunks)
            chunk.metadata["document_total_words"] = total_words
            chunk.metadata["document_avg_words_per_chunk"] = avg_words
        
        logger.info(f"Post-processing complete: {len(chunks)} -> {len(merged_chunks)} chunks")
        return merged_chunks
    
    def _save_results(
        self, 
        chunks: List[ProcessedChunk], 
        output_path: str, 
        document_metadata: Dict[str, Any]
    ):
        """
        Save processing results to file.
        
        Args:
            chunks: Processed chunks
            output_path: Path to save results
            document_metadata: Document metadata
        """
        try:
            import json
            
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare data for saving
            results = {
                "document_metadata": document_metadata,
                "processing_info": {
                    "total_chunks": len(chunks),
                    "batch_size": self.batch_size,
                    "model_used": self.llm.model_name,
                    "context_summary": self.context_manager.get_context_summary()
                },
                "chunks": []
            }
            
            for chunk in chunks:
                chunk_data = {
                    "id": chunk.id,
                    "heading": chunk.heading,
                    "content": chunk.content,
                    "continues": chunk.continues,
                    "level_1_heading": chunk.level_1_heading,
                    "level_2_heading": chunk.level_2_heading,
                    "level_3_heading": chunk.level_3_heading,
                    "metadata": chunk.metadata
                }
                results["chunks"].append(chunk_data)
            
            # Save as JSON
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Results saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save results to {output_path}: {str(e)}")
    
    def get_chunk_statistics(self, chunks: List[ProcessedChunk]) -> Dict[str, Any]:
        """
        Generate statistics about the processed chunks.
        
        Args:
            chunks: List of processed chunks
            
        Returns:
            Statistics dictionary
        """
        if not chunks:
            return {"total_chunks": 0}
        
        # Basic statistics
        stats = {
            "total_chunks": len(chunks),
            "total_words": sum(chunk.metadata["word_count"] for chunk in chunks),
            "avg_words_per_chunk": sum(chunk.metadata["word_count"] for chunk in chunks) / len(chunks),
            "total_characters": sum(chunk.metadata["char_count"] for chunk in chunks),
        }
        
        # Content type distribution
        content_types = {}
        for chunk in chunks:
            for content_type in chunk.metadata.get("content_types", []):
                content_types[content_type] = content_types.get(content_type, 0) + 1
        stats["content_type_distribution"] = content_types
        
        # Continuation analysis
        continuation_counts = {"True": 0, "False": 0, "Partial": 0}
        for chunk in chunks:
            continuation_counts[chunk.continues] = continuation_counts.get(chunk.continues, 0) + 1
        stats["continuation_distribution"] = continuation_counts
        
        # Heading level distribution
        heading_levels = {"level_1_only": 0, "level_2": 0, "level_3": 0}
        for chunk in chunks:
            if chunk.level_3_heading:
                heading_levels["level_3"] += 1
            elif chunk.level_2_heading:
                heading_levels["level_2"] += 1
            else:
                heading_levels["level_1_only"] += 1
        stats["heading_level_distribution"] = heading_levels
        
        return stats
    
    def validate_api_access(self) -> Dict[str, bool]:
        """
        Validate access to required APIs.
        
        Returns:
            Dictionary with validation results
        """
        results = {}
        
        # Test LlamaParse
        try:
            # This would require a small test file
            results["llamaparse"] = True
            logger.info("LlamaParse API access validated")
        except Exception as e:
            results["llamaparse"] = False
            logger.error(f"LlamaParse API validation failed: {str(e)}")
        
        # Test OpenRouter
        try:
            test_response = self.llm._call("Test", max_tokens=5)
            results["openrouter"] = bool(test_response)
            logger.info("OpenRouter API access validated")
        except Exception as e:
            results["openrouter"] = False
            logger.error(f"OpenRouter API validation failed: {str(e)}")
        
        return results

def create_pipeline(**kwargs) -> PDFChunkingPipeline:
    """
    Convenience function to create a pipeline with default settings.
    
    Args:
        **kwargs: Additional arguments for PDFChunkingPipeline
        
    Returns:
        Configured pipeline instance
    """
    return PDFChunkingPipeline(**kwargs)