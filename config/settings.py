import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # OpenAI Configuration
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4-1106-preview")
    openai_temperature: float = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))
    
    # Data Paths
    data_dir: Path = Path(os.getenv("DATA_DIR", "./data/raw"))
    train_dir: Path = Path(os.getenv("TRAIN_DIR", "./data/raw/train"))
    test_dir: Path = Path(os.getenv("TEST_DIR", "./data/raw/test/gold"))
    output_dir: Path = Path(os.getenv("OUTPUT_DIR", "./data/results"))
    
    # Fine-tuning Settings
    finetune_model: str = os.getenv("FINETUNE_MODEL", "gpt-3.5-turbo-1106")
    finetune_epochs: int = int(os.getenv("FINETUNE_EPOCHS", "3"))
    finetune_batch_size: str = os.getenv("FINETUNE_BATCH_SIZE", "auto")
    
    # Experiment Settings
    max_samples: Optional[int] = None if os.getenv("MAX_SAMPLES") == "None" else int(os.getenv("MAX_SAMPLES", "10"))
    n_shots: int = int(os.getenv("N_SHOTS", "3"))
    batch_size: int = int(os.getenv("BATCH_SIZE", "10"))
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_file: Path = Path(os.getenv("LOG_FILE", "./logs/scidtb.log"))
    
    # Project Root
    project_root: Path = Path(__file__).parent.parent
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create directories if they don't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def validate_api_key(self) -> bool:
        """Validate that API key is set."""
        return bool(self.openai_api_key and self.openai_api_key != "")

# Global settings instance
settings = Settings()