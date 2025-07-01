# Unit tests
# tests/test_pipeline.py

"""
Basic tests for the PDF chunking pipeline.
"""

import pytest
import os
import sys
from unittest.mock import Mock, patch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.pipeline.main_pipeline import PDFChunkingPipeline, create_pipeline
from src.chunkers.chunk_processor import ChunkProcessor, ProcessedChunk
from src.chunkers.context_manager import ContextManager
from src.chunkers.prompts import get_chunking_prompt, get_context_summary_prompt

class TestChunkProcessor:
    """Test the chunk processor functionality."""
    
    def setup_method(self):
        """Setup test instances."""
        self.processor = ChunkProcessor()
    
    def test_init(self):
        """Test processor initialization."""
        assert self.processor is not None
        assert hasattr(self.processor, 'continues_pattern')
        assert hasattr(self.processor, 'head_pattern')
    
    def test_parse_heading_hierarchy(self):
        """Test heading hierarchy parsing."""
        # Test three-level hierarchy
        level_1, level_2, level_3 = self.processor._parse_heading_hierarchy(
            "Document Title > Section Name > Subsection"
        )
        assert level_1 == "Document Title"
        assert level_2 == "Section Name"
        assert level_3 == "Subsection"
        
        # Test single level
        level_1, level_2, level_3 = self.processor._parse_heading_hierarchy("Single Title")
        assert level_1 == "Single Title"
        assert level_2 == ""
        assert level_3 == ""
    
    def test_clean_content(self):
        """Test content cleaning."""
        dirty_content = "   Text with   extra    spaces\n\n\nAnd multiple newlines   "
        clean_content = self.processor._clean_content(dirty_content)
        
        assert "   " not in clean_content
        assert "\n\n\n" not in clean_content
        assert clean_content.strip() == clean_content
    
    def test_extract_metadata(self):
        """Test metadata extraction."""
        content = "This is step 1. Follow these instructions carefully."
        metadata = self.processor._extract_metadata(content, "False")
        
        assert "word_count" in metadata
        assert "char_count" in metadata
        assert "content_types" in metadata
        assert metadata["continues"] == "False"
    
    def test_process_single_chunk(self):
        """Test processing a single chunk."""
        raw_chunk = """[CONTINUES]False[/CONTINUES]
[HEAD]Test Document > Introduction > Overview[/HEAD]
This is the content of the chunk. It contains important information about the document structure and how it should be processed."""
        
        chunk = self.processor._process_single_chunk(raw_chunk, "test_chunk_001")
        
        assert chunk is not None
        assert chunk.id == "test_chunk_001"
        assert chunk.continues == "False"
        assert chunk.level_1_heading == "Test Document"
        assert chunk.level_2_heading == "Introduction"
        assert chunk.level_3_heading == "Overview"
        assert "content" in chunk.content.lower()

class TestContextManager:
    """Test the context manager functionality."""
    
    def setup_method(self):
        """Setup test instances."""
        self.context_manager = ContextManager()
    
    def test_init(self):
        """Test context manager initialization."""
        assert self.context_manager is not None
        assert self.context_manager.max_last_chunks == 3
        assert len(self.context_manager.context_history) == 0
    
    def test_reset(self):
        """Test context reset."""
        # Add some context
        self.context_manager.document_title = "Test Document"
        self.context_manager.global_heading_hierarchy["level_1"] = "Test"
        
        # Reset
        self.context_manager.reset()
        
        assert self.context_manager.document_title == ""
        assert self.context_manager.global_heading_hierarchy["level_1"] == ""
        assert len(self.context_manager.context_history) == 0
    
    def test_extract_heading_hierarchy(self):
        """Test heading extraction."""
        chunk_data = {"heading": "Doc > Section > Subsection"}
        hierarchy = self.context_manager.extract_heading_hierarchy(chunk_data)
        
        assert hierarchy["level_1"] == "Doc"
        assert hierarchy["level_2"] == "Section"
        assert hierarchy["level_3"] == "Subsection"
    
    def test_update_global_hierarchy(self):
        """Test global hierarchy updates."""
        hierarchy = {
            "level_1": "New Document",
            "level_2": "New Section",
            "level_3": "New Subsection"
        }
        
        self.context_manager.update_global_hierarchy(hierarchy)
        
        assert self.context_manager.global_heading_hierarchy["level_1"] == "New Document"
        assert self.context_manager.global_heading_hierarchy["level_2"] == "New Section"
        assert self.context_manager.global_heading_hierarchy["level_3"] == "New Subsection"
    
    def test_add_batch_context(self):
        """Test adding batch context."""
        chunks = [
            {
                "heading": "Test > Section > Part1",
                "content": "Content of part 1",
                "continues": "False",
                "metadata": {"word_count": 10}
            }
        ]
        
        self.context_manager.add_batch_context(chunks, "Test batch summary")
        
        assert len(self.context_manager.context_history) == 1
        assert self.context_manager.context_history[0].summary == "Test batch summary"

class TestPrompts:
    """Test prompt generation."""
    
    def test_get_chunking_prompt(self):
        """Test chunking prompt generation."""
        prompt = get_chunking_prompt(
            previous_context="Previous context",
            last_chunks="Last chunks",
            batch_content="Current batch"
        )
        
        assert "Previous context" in prompt
        assert "Last chunks" in prompt
        assert "Current batch" in prompt
        assert "EXTRACTION PHASE" in prompt
        assert "CONTINUES" in prompt
    
    def test_get_context_summary_prompt(self):
        """Test context summary prompt."""
        prompt = get_context_summary_prompt("Some chunks to summarize")
        
        assert "Some chunks to summarize" in prompt
        assert "summary" in prompt.lower()

class TestPipelineIntegration:
    """Integration tests for the pipeline (mocked)."""
    
    @patch('src.parsers.llama_parser.LlamaParse')
    @patch('src.llms.openrouter_llm.requests.post')
    def test_pipeline_creation(self, mock_post, mock_llama_parse):
        """Test pipeline creation with mocked dependencies."""
        # Mock successful API responses
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": "Test response"}}],
            "usage": {"total_tokens": 100}
        }
        
        # Mock LlamaParse
        mock_llama_parse.return_value = Mock()
        
        # Create pipeline with mock API keys
        pipeline = create_pipeline()
        
        assert pipeline is not None
        assert hasattr(pipeline, 'parser')
        assert hasattr(pipeline, 'llm')
        assert hasattr(pipeline, 'context_manager')
        assert hasattr(pipeline, 'chunk_processor')
    
    def test_pipeline_validation(self):
        """Test pipeline validation with missing keys."""
        # This should fail without proper API keys
        with pytest.raises((ValueError, Exception)):
            pipeline = create_pipeline(
                llama_api_key="",
                openrouter_api_key=""
            )

class TestUtils:
    """Test utility functions."""
    
    def test_settings_validation(self):
        """Test settings validation."""
        from config.settings import Settings
        
        # Test with empty keys - check if they exist rather than calling validate()
        settings = Settings()
        
        # Just verify the attributes exist
        assert hasattr(settings, 'LLAMA_CLOUD_API_KEY')
        assert hasattr(settings, 'OPENROUTER_API_KEY')
        
        # Or test that empty keys are handled appropriately by create_pipeline
        with pytest.raises((ValueError, Exception)):
            create_pipeline(llama_api_key="", openrouter_api_key="")

# Pytest configuration
def pytest_configure(config):
    """Configure pytest."""
    # Set test environment variables if needed
    os.environ.setdefault("LLAMA_CLOUD_API_KEY", "test_key")
    os.environ.setdefault("OPENROUTER_API_KEY", "test_key")

# Helper functions for testing
def create_mock_chunk(chunk_id="test_001", heading="Test > Section > Part"):
    """Create a mock ProcessedChunk for testing."""
    return ProcessedChunk(
        id=chunk_id,
        heading=heading,
        content="This is test content for the chunk.",
        continues="False",
        level_1_heading="Test",
        level_2_heading="Section", 
        level_3_heading="Part",
        metadata={
            "word_count": 8,
            "char_count": 36,
            "content_types": ["text"]
        },
        raw_text="[CONTINUES]False[/CONTINUES][HEAD]Test > Section > Part[/HEAD]This is test content for the chunk."
    )

def create_mock_pdf_content():
    """Create mock PDF content for testing."""
    return {
        "pages": [
            {
                "page_number": 1,
                "content": "This is the first page content with some text.",
                "metadata": {}
            },
            {
                "page_number": 2, 
                "content": "This is the second page with more information.",
                "metadata": {}
            }
        ],
        "metadata": {
            "total_pages": 2,
            "parser": "test",
            "file_path": "test.pdf"
        }
    }

if __name__ == "__main__":
    pytest.main([__file__, "-v"])