"""
Training script for fine-tuning models on SciDTB data.

This script:
1. Loads training data
2. Prepares fine-tuning dataset in OpenAI format
3. Uploads data and creates fine-tuning job
4. Optionally waits for completion
"""

import argparse
import json
from pathlib import Path
from typing import Optional
import sys
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import SciDTBLoader
from src.training.data_prep import FineTuningDataPreparator
from src.training.finetune import FineTuningPipeline
from src.utils.logger import get_logger, setup_logging
from config.settings import settings

logger = get_logger(__name__)


def prepare_finetune_data(
    data_dir: Path,
    output_dir: Path,
    train_split: str = 'train',
    val_split: Optional[str] = None,
    validate: bool = True
) -> tuple[Path, Optional[Path]]:
    """
    Prepare fine-tuning datasets.
    
    Args:
        data_dir: Root data directory
        output_dir: Output directory for prepared data
        train_split: Training split name
        val_split: Optional validation split name
        validate: Whether to validate prepared data
        
    Returns:
        Tuple of (training_file, validation_file)
    """
    logger.info("="*60)
    logger.info("Preparing Fine-tuning Data")
    logger.info("="*60)
    
    # Load data
    loader = SciDTBLoader(data_dir)
    
    logger.info(f"\nLoading {train_split} split...")
    train_trees = loader.load_split(train_split)
    
    if not train_trees:
        raise ValueError(f"No training data found in {train_split}")
    
    logger.info(f"✓ Loaded {len(train_trees)} training trees")
    
    # Extract texts
    train_texts = [SciDTBLoader.extract_text(tree) for tree in train_trees]
    
    # Prepare training data
    preparator = FineTuningDataPreparator(
        model_type=settings.finetune_model,
        include_format_instructions=True
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_file = output_dir / f"finetune_train_{timestamp}.jsonl"
    
    logger.info(f"\nPreparing training data: {len(train_trees)} examples")
    preparator.prepare_dataset(
        trees=train_trees,
        texts=train_texts,
        output_file=train_file,
        validate=validate
    )
    
    # Validate if requested
    if validate:
        logger.info("\nValidating training data...")
        report = preparator.validate_dataset(train_file)
        
        if report['invalid_examples'] > 0:
            logger.warning(
                f"Found {report['invalid_examples']} invalid examples "
                f"out of {report['total_examples']}"
            )
        else:
            logger.info("✓ All training examples are valid")
    
    # Prepare validation data if specified
    val_file = None
    if val_split:
        logger.info(f"\nLoading {val_split} split...")
        val_trees = loader.load_split(val_split)
        
        if val_trees:
            logger.info(f"✓ Loaded {len(val_trees)} validation trees")
            val_texts = [SciDTBLoader.extract_text(tree) for tree in val_trees]
            
            val_file = output_dir / f"finetune_val_{timestamp}.jsonl"
            
            logger.info(f"Preparing validation data: {len(val_trees)} examples")
            preparator.prepare_dataset(
                trees=val_trees,
                texts=val_texts,
                output_file=val_file,
                validate=validate
            )
            
            if validate:
                logger.info("\nValidating validation data...")
                val_report = preparator.validate_dataset(val_file)
                
                if val_report['invalid_examples'] > 0:
                    logger.warning(
                        f"Found {val_report['invalid_examples']} invalid examples "
                        f"out of {val_report['total_examples']}"
                    )
                else:
                    logger.info("✓ All validation examples are valid")
    
    logger.info("\n" + "="*60)
    logger.info("Data Preparation Complete")
    logger.info("="*60)
    logger.info(f"Training file:   {train_file}")
    if val_file:
        logger.info(f"Validation file: {val_file}")
    logger.info("="*60 + "\n")
    
    return train_file, val_file


def start_finetuning(
    training_file: Path,
    validation_file: Optional[Path],
    api_key: str,
    model: str,
    n_epochs: int,
    suffix: str,
    output_dir: Path,
    wait_for_completion: bool = False
) -> dict:
    """
    Start fine-tuning job.
    
    Args:
        training_file: Path to training JSONL file
        validation_file: Optional validation file
        api_key: OpenAI API key
        model: Base model name
        n_epochs: Number of training epochs
        suffix: Model name suffix
        output_dir: Output directory for job info
        wait_for_completion: Whether to wait for completion
        
    Returns:
        Dictionary with job information
    """
    logger.info("="*60)
    logger.info("Starting Fine-tuning Job")
    logger.info("="*60)
    logger.info(f"Base model:      {model}")
    logger.info(f"Training file:   {training_file}")
    logger.info(f"Validation file: {validation_file or 'None'}")
    logger.info(f"Epochs:          {n_epochs}")
    logger.info(f"Suffix:          {suffix}")
    logger.info("="*60 + "\n")
    
    # Create pipeline
    pipeline = FineTuningPipeline(api_key=api_key, output_dir=output_dir)
    
    # Run pipeline
    result = pipeline.run_full_pipeline(
        training_file=training_file,
        model=model,
        n_epochs=n_epochs,
        suffix=suffix,
        validation_file=validation_file,
        wait_for_completion=wait_for_completion
    )
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune models on SciDTB data"
    )
    
    # Data arguments
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('./data/raw'),
        help='Root directory containing raw SciDTB data'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('./data/processed'),
        help='Output directory for prepared data and job info'
    )
    parser.add_argument(
        '--train-split',
        type=str,
        default='train',
        help='Training split name'
    )
    parser.add_argument(
        '--val-split',
        type=str,
        default=None,
        help='Validation split name (optional)'
    )
    
    # Fine-tuning arguments
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help=f'Base model (default: {settings.finetune_model})'
    )
    parser.add_argument(
        '--n-epochs',
        type=int,
        default=None,
        help=f'Number of epochs (default: {settings.finetune_epochs})'
    )
    parser.add_argument(
        '--suffix',
        type=str,
        default='scidtb',
        help='Model name suffix'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='OpenAI API key (default: from .env)'
    )
    
    # Pipeline control
    parser.add_argument(
        '--prepare-only',
        action='store_true',
        help='Only prepare data, do not start fine-tuning'
    )
    parser.add_argument(
        '--upload-and-train',
        action='store_true',
        help='Upload data and start fine-tuning'
    )
    parser.add_argument(
        '--wait',
        action='store_true',
        help='Wait for fine-tuning to complete'
    )
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Skip data validation'
    )
    
    # Logging
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    
    # Get configuration
    api_key = args.api_key or settings.openai_api_key
    model = args.model or settings.finetune_model
    n_epochs = args.n_epochs or settings.finetune_epochs
    
    # Validate API key
    if not args.prepare_only and not api_key:
        logger.error("OpenAI API key not found. Set OPENAI_API_KEY in .env or use --api-key")
        return 1
    
    try:
        # Prepare fine-tuning data
        train_file, val_file = prepare_finetune_data(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            train_split=args.train_split,
            val_split=args.val_split,
            validate=not args.no_validate
        )
        
        # If only preparing data, stop here
        if args.prepare_only:
            logger.info("Data preparation complete. Use --upload-and-train to start fine-tuning.")
            return 0
        
        # Start fine-tuning if requested
        if args.upload_and_train:
            result = start_finetuning(
                training_file=train_file,
                validation_file=val_file,
                api_key=api_key,
                model=model,
                n_epochs=n_epochs,
                suffix=args.suffix,
                output_dir=args.output_dir,
                wait_for_completion=args.wait
            )
            
            # Display result
            print(f"\n{'='*60}")
            print("Fine-tuning Job Created".center(60))
            print(f"{'='*60}")
            print(f"Job ID:          {result['job_id']}")
            if result.get('model_id'):
                print(f"Model ID:        {result['model_id']}")
            else:
                print(f"Status:          In progress")
                print(f"\nMonitor at: https://platform.openai.com/finetune/{result['job_id']}")
            print(f"{'='*60}\n")
        
        else:
            logger.info("\nData prepared. Use --upload-and-train to start fine-tuning.")
            logger.info(f"Training file: {train_file}")
            if val_file:
                logger.info(f"Validation file: {val_file}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())