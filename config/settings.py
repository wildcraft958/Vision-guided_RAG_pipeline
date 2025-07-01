# Configuration management
# config/settings.py

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    """Configuration settings for the PDF chunking pipeline."""
    
    # API Keys
    LLAMAINDEX_API_KEY: str = os.getenv("LLAMAINDEX_API_KEY", "")
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    LLAMA_CLOUD_API_KEY = LLAMAINDEX_API_KEY  # Alias for compatibility
    # OpenRouter Configuration
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_MODEL: str = "deepseek/deepseek-r1:free"
    
    # LlamaParse Configuration  
    LLAMAPARSE_RESULT_TYPE: str = "markdown"
    LLAMAPARSE_VERBOSE: bool = True
    LLAMAPARSE_PARTITION_PAGES: int = 100
    
    # Chunking Configuration
    BATCH_SIZE: int = 2  # Pages per batch
    MAX_CHUNK_SIZE: int = 10000  # Maximum tokens per chunk
    OVERLAP_SIZE: int = 200  # Overlap between chunks
    
    # Model Configuration
    TEMPERATURE: float = 0.1
    MAX_TOKENS: int = 20000
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that required API keys are present."""
        missing_keys = []
        
        if not cls.LLAMAINDEX_API_KEY:
            missing_keys.append("LLAMAINDEX_API_KEY")
        
        if not cls.OPENROUTER_API_KEY:
            missing_keys.append("OPENROUTER_API_KEY")
        
        if missing_keys:
            raise ValueError(f"Missing required API keys: {', '.join(missing_keys)}")
        
        return True

# Global settings instance
settings = Settings()