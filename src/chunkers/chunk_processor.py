# LLM output processing
# src/chunkers/chunk_processor.py

"""
Chunk processor for parsing LLM output and structuring chunks according to research specifications.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ProcessedChunk:
    """Container for a processed chunk with all metadata."""
    id: str
    heading: str
    content: str
    continues: str
    level_1_heading: str
    level_2_heading: str
    level_3_heading: str
    metadata: Dict[str, Any]
    raw_text: str

class ChunkProcessor:
    """
    Processes LLM output to extract and structure chunks according to the research paper format.
    
    Expected format:
    [CONTINUES]True|False|Partial[/CONTINUES]
    [HEAD]main_heading > section_heading > chunk_heading[/HEAD]
    chunk_content
    """
    
    def __init__(self):
        """Initialize chunk processor."""
        # Regex patterns for parsing LLM output
        self.continues_pattern = re.compile(r'\[CONTINUES\](True|False|Partial)\[/CONTINUES\]')
        self.head_pattern = re.compile(r'\[HEAD\](.*?)\[/HEAD\]')
        
        # Patterns for identifying content types
        self.step_patterns = [
            re.compile(r'^\s*\d+\.\s', re.MULTILINE),
            re.compile(r'step\s+\d+', re.IGNORECASE),
            re.compile(r'(first|second|third|next|then|finally)', re.IGNORECASE)
        ]
        
        self.table_patterns = [
            re.compile(r'\|.*\|', re.MULTILINE),
            re.compile(r'^[-\s|]+$', re.MULTILINE)
        ]
        
        logger.info("Chunk processor initialized")
    
    def process_llm_output(self, llm_output: str, batch_id: str = "") -> List[ProcessedChunk]:
        """
        Process LLM output to extract structured chunks.
        
        Args:
            llm_output: Raw output from LLM
            batch_id: Identifier for the batch being processed
            
        Returns:
            List of ProcessedChunk objects
        """
        try:
            chunks = self._split_into_raw_chunks(llm_output)
            processed_chunks = []
            
            for i, raw_chunk in enumerate(chunks):
                try:
                    processed = self._process_single_chunk(raw_chunk, f"{batch_id}_chunk_{i}")
                    if processed:
                        processed_chunks.append(processed)
                except Exception as e:
                    logger.warning(f"Failed to process chunk {i} in batch {batch_id}: {str(e)}")
                    continue
            
            logger.info(f"Processed {len(processed_chunks)} chunks from batch {batch_id}")
            return processed_chunks
            
        except Exception as e:
            logger.error(f"Error processing LLM output for batch {batch_id}: {str(e)}")
            return []
    
    def _split_into_raw_chunks(self, llm_output: str) -> List[str]:
        """
        Split LLM output into individual chunk strings.
        
        Args:
            llm_output: Raw LLM output
            
        Returns:
            List of raw chunk strings
        """
        # Split on CONTINUES flag as chunk boundary
        chunks = re.split(r'(?=\[CONTINUES\])', llm_output)
        
        # Filter out empty chunks and clean up
        cleaned_chunks = []
        for chunk in chunks:
            chunk = chunk.strip()
            if chunk and '[CONTINUES]' in chunk:
                cleaned_chunks.append(chunk)
        
        return cleaned_chunks
    
    def _process_single_chunk(self, raw_chunk: str, chunk_id: str) -> Optional[ProcessedChunk]:
        """
        Process a single raw chunk into a structured chunk.
        
        Args:
            raw_chunk: Raw chunk text
            chunk_id: Unique identifier for the chunk
            
        Returns:
            ProcessedChunk object or None if processing fails
        """
        # Extract continues flag
        continues_match = self.continues_pattern.search(raw_chunk)
        continues = continues_match.group(1) if continues_match else "False"
        
        # Extract heading
        head_match = self.head_pattern.search(raw_chunk)
        if not head_match:
            logger.warning(f"No heading found in chunk {chunk_id}")
            return None
        
        heading = head_match.group(1).strip()
        
        # Parse heading hierarchy
        level_1, level_2, level_3 = self._parse_heading_hierarchy(heading)
        
        # Extract content (everything after the HEAD tag)
        content_start = head_match.end()
        content = raw_chunk[content_start:].strip()
        
        # Clean up content
        content = self._clean_content(content)
        
        # Skip if content is too short
        if len(content.split('\n')) < 3 and len(content.split()) < 10:
            logger.debug(f"Skipping short chunk {chunk_id}")
            return None
        
        # Extract metadata
        metadata = self._extract_metadata(content, continues)
        
        return ProcessedChunk(
            id=chunk_id,
            heading=heading,
            content=content,
            continues=continues,
            level_1_heading=level_1,
            level_2_heading=level_2,
            level_3_heading=level_3,
            metadata=metadata,
            raw_text=raw_chunk
        )
    
    def _parse_heading_hierarchy(self, heading: str) -> Tuple[str, str, str]:
        """
        Parse heading into three-level hierarchy.
        
        Args:
            heading: Full heading string
            
        Returns:
            Tuple of (level_1, level_2, level_3)
        """
        if " > " in heading:
            parts = [part.strip() for part in heading.split(" > ")]
            
            level_1 = parts[0] if len(parts) > 0 else ""
            level_2 = parts[1] if len(parts) > 1 else ""
            level_3 = parts[2] if len(parts) > 2 else ""
            
            # If there are more than 3 parts, combine the rest into level 3
            if len(parts) > 3:
                level_3 = " > ".join(parts[2:])
                
        else:
            level_1 = heading
            level_2 = ""
            level_3 = ""
        
        return level_1, level_2, level_3
    
    def _clean_content(self, content: str) -> str:
        """
        Clean and normalize chunk content.
        
        Args:
            content: Raw content
            
        Returns:
            Cleaned content
        """
        # Remove extra whitespace
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        content = re.sub(r'[ \t]+', ' ', content)
        
        # Remove page numbers and footers
        content = re.sub(r'^\s*Page \d+.*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'^\s*\d+\s*$', '', content, flags=re.MULTILINE)
        
        # Clean up bullet points and lists
        content = re.sub(r'^\s*[•·‣⁃]\s*', '• ', content, flags=re.MULTILINE)
        
        return content.strip()
    
    def _extract_metadata(self, content: str, continues: str) -> Dict[str, Any]:
        """
        Extract metadata from chunk content.
        
        Args:
            content: Chunk content
            continues: Continuation flag
            
        Returns:
            Metadata dictionary
        """
        metadata = {
            "continues": continues,
            "word_count": len(content.split()),
            "char_count": len(content),
            "line_count": len(content.split('\n')),
        }
        
        # Detect content type
        content_types = []
        
        # Check for steps/procedures
        if any(pattern.search(content) for pattern in self.step_patterns):
            content_types.append("procedure")
        
        # Check for tables
        if any(pattern.search(content) for pattern in self.table_patterns):
            content_types.append("table")
        
        # Check for lists
        if re.search(r'^\s*[•\-\*]\s', content, re.MULTILINE):
            content_types.append("list")
        
        # Check for code
        if re.search(r'```|`.*`', content):
            content_types.append("code")
        
        # Check for figures/images
        if re.search(r'(figure|image|diagram|chart|graph)', content, re.IGNORECASE):
            content_types.append("figure")
        
        metadata["content_types"] = content_types
        
        # Extract key phrases
        key_phrases = self._extract_key_phrases(content)
        metadata["key_phrases"] = key_phrases
        
        return metadata
    
    def _extract_key_phrases(self, content: str, max_phrases: int = 5) -> List[str]:
        """
        Extract key phrases from content.
        
        Args:
            content: Chunk content
            max_phrases: Maximum number of phrases to extract
            
        Returns:
            List of key phrases
        """
        # Simple keyword extraction - could be enhanced with NLP libraries
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
        
        # Count frequency
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get most frequent phrases
        sorted_phrases = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        return [phrase for phrase, _ in sorted_phrases[:max_phrases]]
    
    def validate_chunks(self, chunks: List[ProcessedChunk]) -> List[ProcessedChunk]:
        """
        Validate and filter chunks based on quality criteria.
        
        Args:
            chunks: List of chunks to validate
            
        Returns:
            List of validated chunks
        """
        validated_chunks = []
        
        for chunk in chunks:
            validation_issues = []
            
            # Check minimum content length
            if chunk.metadata["word_count"] < 5:
                validation_issues.append("Content too short")
            
            # Check for meaningful content
            if not re.search(r'[a-zA-Z]', chunk.content):
                validation_issues.append("No alphabetic content")
            
            # Check heading structure
            if not chunk.level_1_heading:
                validation_issues.append("Missing level 1 heading")
            
            # Log issues but don't filter out (allow for debugging)
            if validation_issues:
                logger.debug(f"Chunk {chunk.id} validation issues: {validation_issues}")
            
            validated_chunks.append(chunk)
        
        logger.info(f"Validated {len(validated_chunks)} chunks")
        return validated_chunks
    
    def merge_related_chunks(self, chunks: List[ProcessedChunk]) -> List[ProcessedChunk]:
        """
        Merge chunks that should be combined based on continuation flags and content analysis.
        
        Args:
            chunks: List of chunks to potentially merge
            
        Returns:
            List of chunks with related chunks merged
        """
        if not chunks:
            return chunks
        
        merged_chunks = []
        current_chunk = chunks[0]
        
        for next_chunk in chunks[1:]:
            should_merge = self._should_merge_chunks(current_chunk, next_chunk)
            
            if should_merge:
                # Merge chunks
                current_chunk = self._merge_two_chunks(current_chunk, next_chunk)
                logger.debug(f"Merged chunks {current_chunk.id} and {next_chunk.id}")
            else:
                # Add current chunk to results and start new one
                merged_chunks.append(current_chunk)
                current_chunk = next_chunk
        
        # Add the last chunk
        merged_chunks.append(current_chunk)
        
        logger.info(f"Merged chunks: {len(chunks)} -> {len(merged_chunks)}")
        return merged_chunks
    
    def _should_merge_chunks(self, chunk1: ProcessedChunk, chunk2: ProcessedChunk) -> bool:
        """
        Determine if two chunks should be merged.
        
        Args:
            chunk1: First chunk
            chunk2: Second chunk
            
        Returns:
            True if chunks should be merged
        """
        # Check continuation flag
        if chunk2.continues == "True":
            return True
        
        # Check if same heading
        if (chunk1.heading == chunk2.heading and 
            chunk1.heading != "" and 
            chunk2.heading != ""):
            return True
        
        # Check for step continuation
        if ("procedure" in chunk1.metadata.get("content_types", []) and 
            "procedure" in chunk2.metadata.get("content_types", [])):
            return True
        
        # Check for table continuation
        if ("table" in chunk1.metadata.get("content_types", []) and 
            "table" in chunk2.metadata.get("content_types", [])):
            return True
        
        return False
    
    def _merge_two_chunks(self, chunk1: ProcessedChunk, chunk2: ProcessedChunk) -> ProcessedChunk:
        """
        Merge two chunks into one.
        
        Args:
            chunk1: First chunk
            chunk2: Second chunk
            
        Returns:
            Merged chunk
        """
        # Combine content
        merged_content = chunk1.content + "\n\n" + chunk2.content
        
        # Use first chunk's heading if second is similar, otherwise combine
        if chunk2.heading == chunk1.heading or chunk2.level_3_heading == "":
            merged_heading = chunk1.heading
        else:
            merged_heading = chunk1.heading
        
        # Merge metadata
        merged_metadata = chunk1.metadata.copy()
        merged_metadata["word_count"] += chunk2.metadata["word_count"]
        merged_metadata["char_count"] += chunk2.metadata["char_count"]
        merged_metadata["line_count"] += chunk2.metadata["line_count"]
        
        # Combine content types
        content_types = list(set(
            merged_metadata.get("content_types", []) + 
            chunk2.metadata.get("content_types", [])
        ))
        merged_metadata["content_types"] = content_types
        
        # Combine key phrases
        key_phrases = list(set(
            merged_metadata.get("key_phrases", []) + 
            chunk2.metadata.get("key_phrases", [])
        ))
        merged_metadata["key_phrases"] = key_phrases[:10]  # Limit to 10
        
        # Mark as merged
        merged_metadata["merged_from"] = [chunk1.id, chunk2.id]
        
        return ProcessedChunk(
            id=f"{chunk1.id}_merged",
            heading=merged_heading,
            content=merged_content,
            continues=chunk1.continues,
            level_1_heading=chunk1.level_1_heading,
            level_2_heading=chunk1.level_2_heading,
            level_3_heading=chunk1.level_3_heading,
            metadata=merged_metadata,
            raw_text=chunk1.raw_text + "\n\n" + chunk2.raw_text
        )