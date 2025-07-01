# Cross-batch context
# src/chunkers/context_manager.py

"""
Context manager for maintaining semantic continuity across document batches.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ChunkContext:
    """Container for chunk context information."""
    summary: str
    last_chunks: List[Dict[str, Any]]
    heading_hierarchy: Dict[str, str]
    document_title: str
    current_section: str

class ContextManager:
    """
    Manages context preservation across document processing batches.
    
    Based on the research paper's approach to maintaining semantic coherence
    and structural integrity across page boundaries.
    """
    
    def __init__(self, max_last_chunks: int = 3):
        """
        Initialize context manager.
        
        Args:
            max_last_chunks: Maximum number of last chunks to preserve
        """
        self.max_last_chunks = max_last_chunks
        self.context_history: List[ChunkContext] = []
        self.global_heading_hierarchy = {
            "level_1": "",
            "level_2": "",
            "level_3": ""
        }
        self.document_title = ""
        self.reset()
    
    def reset(self):
        """Reset context for new document."""
        self.context_history = []
        self.global_heading_hierarchy = {
            "level_1": "",
            "level_2": "",
            "level_3": ""
        }
        self.document_title = ""
        logger.info("Context manager reset for new document")
    
    def extract_heading_hierarchy(self, chunk_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Extract heading hierarchy from chunk data.
        
        Args:
            chunk_data: Chunk with heading information
            
        Returns:
            Dictionary with heading levels
        """
        heading_text = chunk_data.get("heading", "")
        
        if " > " in heading_text:
            parts = [part.strip() for part in heading_text.split(" > ")]
            
            hierarchy = {
                "level_1": parts[0] if len(parts) > 0 else "",
                "level_2": parts[1] if len(parts) > 1 else "",
                "level_3": parts[2] if len(parts) > 2 else ""
            }
        else:
            hierarchy = {
                "level_1": heading_text,
                "level_2": "",
                "level_3": ""
            }
        
        return hierarchy
    
    def update_global_hierarchy(self, hierarchy: Dict[str, str]):
        """
        Update global heading hierarchy with new information.
        
        Args:
            hierarchy: New heading hierarchy
        """
        for level, heading in hierarchy.items():
            if heading and heading != self.global_heading_hierarchy.get(level, ""):
                self.global_heading_hierarchy[level] = heading
                
                # If updating a higher level, clear lower levels
                if level == "level_1":
                    self.global_heading_hierarchy["level_2"] = ""
                    self.global_heading_hierarchy["level_3"] = ""
                elif level == "level_2":
                    self.global_heading_hierarchy["level_3"] = ""
        
        # Set document title from level 1 if not set
        if not self.document_title and self.global_heading_hierarchy["level_1"]:
            self.document_title = self.global_heading_hierarchy["level_1"]
    
    def add_batch_context(
        self,
        chunks: List[Dict[str, Any]],
        batch_summary: str = ""
    ):
        """
        Add context from a processed batch.
        
        Args:
            chunks: List of chunks from the batch
            batch_summary: Summary of the batch content
        """
        if not chunks:
            logger.warning("No chunks provided for context update")
            return
        
        # Update heading hierarchy from chunks
        for chunk in chunks:
            hierarchy = self.extract_heading_hierarchy(chunk)
            self.update_global_hierarchy(hierarchy)
        
        # Keep only the last few chunks for context
        last_chunks = chunks[-self.max_last_chunks:] if len(chunks) > self.max_last_chunks else chunks
        
        # Determine current section from the last chunk
        current_section = ""
        if last_chunks:
            last_hierarchy = self.extract_heading_hierarchy(last_chunks[-1])
            current_section = last_hierarchy.get("level_2", "")
        
        # Create context object
        context = ChunkContext(
            summary=batch_summary,
            last_chunks=last_chunks,
            heading_hierarchy=self.global_heading_hierarchy.copy(),
            document_title=self.document_title,
            current_section=current_section
        )
        
        self.context_history.append(context)
        
        logger.info(
            f"Added context from batch with {len(chunks)} chunks. "
            f"Current section: {current_section}"
        )
    
    def get_context_for_batch(self) -> Dict[str, Any]:
        """
        Get context information for processing the next batch.
        
        Returns:
            Dictionary with context information
        """
        if not self.context_history:
            return {
                "previous_context": "",
                "last_chunks": "",
                "heading_hierarchy": self.global_heading_hierarchy,
                "document_title": self.document_title
            }
        
        latest_context = self.context_history[-1]
        
        # Build previous context summary
        context_parts = []
        if self.document_title:
            context_parts.append(f"Document: {self.document_title}")
        
        if latest_context.current_section:
            context_parts.append(f"Current Section: {latest_context.current_section}")
        
        if latest_context.summary:
            context_parts.append(f"Previous Content: {latest_context.summary}")
        
        previous_context = "\n".join(context_parts)
        
        # Format last chunks
        last_chunks_text = self._format_chunks_for_context(latest_context.last_chunks)
        
        return {
            "previous_context": previous_context,
            "last_chunks": last_chunks_text,
            "heading_hierarchy": latest_context.heading_hierarchy,
            "document_title": latest_context.document_title
        }
    
    def _format_chunks_for_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Format chunks for inclusion in context.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Formatted chunk text
        """
        if not chunks:
            return ""
        
        formatted_chunks = []
        
        for i, chunk in enumerate(chunks):
            chunk_text = f"CHUNK {i+1}:\n"
            
            # Add heading if present
            heading = chunk.get("heading", "")
            if heading:
                chunk_text += f"[HEAD]{heading}[/HEAD]\n"
            
            # Add content
            content = chunk.get("content", "")
            if content:
                # Truncate very long content
                if len(content) > 300:
                    content = content[:300] + "..."
                chunk_text += content
            
            # Add continuation flag if present
            continues = chunk.get("continues", "")
            if continues:
                chunk_text += f"\n[CONTINUES]{continues}[/CONTINUES]"
            
            formatted_chunks.append(chunk_text)
        
        return "\n\n".join(formatted_chunks)
    
    def get_context_summary(self) -> str:
        """
        Get a summary of all context accumulated so far.
        
        Returns:
            Context summary string
        """
        if not self.context_history:
            return "No context available."
        
        summary_parts = []
        
        # Document overview
        if self.document_title:
            summary_parts.append(f"Document: {self.document_title}")
        
        # Current hierarchy
        hierarchy_parts = []
        for level, heading in self.global_heading_hierarchy.items():
            if heading:
                hierarchy_parts.append(f"{level.replace('_', ' ').title()}: {heading}")
        
        if hierarchy_parts:
            summary_parts.append("Structure: " + " > ".join(hierarchy_parts))
        
        # Recent sections
        recent_sections = []
        for context in self.context_history[-3:]:  # Last 3 contexts
            if context.current_section and context.current_section not in recent_sections:
                recent_sections.append(context.current_section)
        
        if recent_sections:
            summary_parts.append(f"Recent Sections: {', '.join(recent_sections)}")
        
        # Latest summary
        if self.context_history[-1].summary:
            summary_parts.append(f"Latest: {self.context_history[-1].summary}")
        
        return "\n".join(summary_parts)
    
    def detect_context_continuity(self, new_chunk: Dict[str, Any]) -> bool:
        """
        Detect if a new chunk continues from previous context.
        
        Args:
            new_chunk: New chunk to analyze
            
        Returns:
            True if chunk appears to continue from previous context
        """
        if not self.context_history:
            return False
        
        latest_context = self.context_history[-1]
        
        # Check if chunk is explicitly marked as continuing
        continues_flag = new_chunk.get("continues", "").lower()
        if continues_flag == "true":
            return True
        
        # Check heading hierarchy continuity
        new_hierarchy = self.extract_heading_hierarchy(new_chunk)
        
        # If same document and section, likely continuous
        if (new_hierarchy.get("level_1") == latest_context.heading_hierarchy.get("level_1") and
            new_hierarchy.get("level_2") == latest_context.heading_hierarchy.get("level_2")):
            return True
        
        return False
    
    def should_merge_with_previous(self, new_chunk: Dict[str, Any]) -> bool:
        """
        Determine if new chunk should be merged with previous chunk.
        
        Args:
            new_chunk: New chunk to analyze
            
        Returns:
            True if should merge
        """
        if not self.context_history or not self.context_history[-1].last_chunks:
            return False
        
        last_chunk = self.context_history[-1].last_chunks[-1]
        
        # Check for explicit continuation
        if new_chunk.get("continues", "").lower() == "true":
            return True
        
        # Check for same heading
        if (new_chunk.get("heading", "") == last_chunk.get("heading", "") and
            new_chunk.get("heading", "") != ""):
            return True
        
        # Check for step continuation in procedures
        last_content = last_chunk.get("content", "").lower()
        new_content = new_chunk.get("content", "").lower()
        
        step_indicators = ["step", "instruction", "procedure", "first", "next", "then", "finally"]
        
        if any(indicator in last_content for indicator in step_indicators):
            if any(indicator in new_content for indicator in step_indicators):
                return True
        
        return False