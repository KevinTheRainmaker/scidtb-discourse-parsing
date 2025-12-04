"""
Zero-shot discourse parsing experiment.

This script:
1. Loads test data
2. Parses using zero-shot approach
3. Evaluates against gold standard
4. Saves results and predictions
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional
import sys
from datetime import datetime
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parsers import ZeroShotParser
from src.data.loader import SciDTBLoader
from src.models import DiscourseTreeModel
from src.evaluation.metrics import DiscourseEvaluator, RelationEvaluator
from src.utils.logger import get_logger, setup_logging
from config.settings import settings

logger = get_logger(__name__)


def parse_test_set(
    parser: ZeroShotParser,
    gold_trees: List[DiscourseTreeModel],
    max_samples: Optional[int] = None
) -> List[Optional[DiscourseTreeModel]]:
    """
    Parse test set using zero-shot parser.
    
    Args:
        parser: ZeroShotParser instance
        gold_trees: Gold standard trees
        max_samples: Maximum number of samples to parse
        
    Returns:
        List of predicted trees (None for failed parses)
    """
    if max_samples:
        gold_trees = gold_trees[:max_samples]
    
    logger.info(f"Parsing {len(gold_trees)} test samples with zero-shot approach...")
    
    predictions = []
    
    for i, gold_tree in enumerate(tqdm(gold_trees, desc="Parsing")):
        # Extract text
        text = SciDTBLoader.extract_text(gold_tree)
        
        # Parse
        pred_tree = parser.parse(text)
        predictions.append(pred_tree)
        
        # Log progress
        if (i + 1) % 10 == 0:
            logger.info(
                f"Processed {i + 1}/{len(gold_trees)} - "
                f"Success rate: {parser.get_statistics()['success_rate']:.2f}%"
            )
    
    return predictions


def save_predictions(
    predictions: List[Optional[DiscourseTreeModel]],
    output_file: Path
):
    """
    Save predictions to JSON file.
    
    Args:
        predictions: List of predicted trees
        output_file: Output file path
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to serializable format
    data = []
    for pred in predictions:
        if pred is not None:
            data.append(pred.to_dict())
        else:
            data.append(None)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ Predictions saved to {output_file}")


def run_experiment(
    data_dir: Path,
    output_dir: Path,
    test_split: str = 'test/gold',
    max_samples: Optional[int] = None,
    model: str = "gpt-4-1106-preview",
    temperature: float = 0.0,
    save_predictions_flag: bool = True
) -> dict:
    """
    Run complete zero-shot experiment.
    
    Args:
        data_dir: Root data directory
        output_dir: Output directory for results
        test_split: Test split name
        max_samples: Maximum samples to process
        model: Model name
        temperature: Sampling temperature
        save_predictions_flag: Whether to save predictions
        
    Returns:
        Dictionary with experiment results
    """
    logger.info("="*70)
    logger.info("Zero-shot Discourse Parsing Experiment")
    logger.info("="*70)
    logger.info(f"Model: {model}")
    logger.info(f"Temperature: {temperature}")
    logger.info(f"Test split: {test_split}")
    logger.info(f"Max samples: {max_samples or 'All'}")
    logger.info("="*70)
    
    # Load test data
    logger.info("\n[1] Loading test data...")
    loader = SciDTBLoader(data_dir)
    gold_trees = loader.load_split(test_split)
    
    if not gold_trees:
        raise ValueError(f"No test data found in {test_split}")
    
    logger.info(f"✓ Loaded {len(gold_trees)} gold trees")
    
    if max_samples:
        gold_trees = gold_trees[:max_samples]
        logger.info(f"Limited to {len(gold_trees)} samples")
    
    # Initialize parser
    logger.info("\n[2] Initializing zero-shot parser...")
    parser = ZeroShotParser(
        api_key=settings.openai_api_key,
        model=model,
        temperature=temperature,
        max_retries=3
    )
    logger.info("✓ Parser initialized")
    
    # Parse test set
    logger.info("\n[3] Parsing test set...")
    predictions = parse_test_set(parser, gold_trees)
    
    # Filter out failed parses
    valid_indices = [i for i, pred in enumerate(predictions) if pred is not None]
    valid_predictions = [predictions[i] for i in valid_indices]
    valid_gold = [gold_trees[i] for i in valid_indices]
    
    logger.info(
        f"✓ Parsing complete: {len(valid_predictions)}/{len(predictions)} successful"
    )
    
    # Print parser statistics
    parser.print_statistics()
    
    # Evaluate
    logger.info("\n[4] Evaluating predictions...")
    evaluator = DiscourseEvaluator(include_root=False)
    
    if valid_predictions:
        metrics = evaluator.evaluate_batch(
            gold_trees=valid_gold,
            pred_trees=valid_predictions,
            verbose=True
        )
        
        # Print results
        metrics.print_summary(title="Zero-shot Results")
        
        # Relation-level evaluation
        logger.info("Computing relation-level metrics...")
        relation_metrics = RelationEvaluator.evaluate_relations(
            valid_gold,
            valid_predictions
        )
        RelationEvaluator.print_relation_report(relation_metrics)
        
    else:
        logger.error("No valid predictions to evaluate")
        metrics = None
        relation_metrics = {}
    
    # Save results
    logger.info("\n[5] Saving results...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save predictions
    if save_predictions_flag:
        pred_file = output_dir / f"zero_shot_predictions_{timestamp}.json"
        save_predictions(predictions, pred_file)
    
    # Save evaluation results
    if metrics:
        results = {
            "experiment": "zero_shot",
            "model": model,
            "temperature": temperature,
            "test_split": test_split,
            "timestamp": timestamp,
            "summary": metrics.to_dict(),
            "relation_metrics": relation_metrics,
            "parser_stats": parser.get_statistics()
        }
        
        results_file = output_dir / f"zero_shot_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Results saved to {results_file}")
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"{'Zero-shot Experiment Complete':^70}")
    print(f"{'='*70}")
    print(f"Total samples:       {len(gold_trees)}")
    print(f"Successful parses:   {len(valid_predictions)}")
    print(f"Failed parses:       {len(gold_trees) - len(valid_predictions)}")
    if metrics:
        print(f"UAS:                 {metrics.uas*100:.2f}%")
        print(f"LAS:                 {metrics.las*100:.2f}%")
        print(f"F1:                  {metrics.f1*100:.2f}%")
    print(f"{'='*70}\n")
    
    return {
        "metrics": metrics.to_dict() if metrics else None,
        "parser_stats": parser.get_statistics(),
        "num_predictions": len(valid_predictions)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run zero-shot discourse parsing experiment"
    )
    
    # Data arguments
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('./data/raw'),
        help='Root directory containing SciDTB data'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('./data/results/zero_shot'),
        help='Output directory for results'
    )
    parser.add_argument(
        '--test-split',
        type=str,
        default='test/gold',
        help='Test split name'
    )
    
    # Experiment arguments
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to process'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4-1106-preview',
        help='Model name'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.0,
        help='Sampling temperature'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='OpenAI API key (default: from .env)'
    )
    
    # Output options
    parser.add_argument(
        '--no-save-predictions',
        action='store_true',
        help='Do not save prediction files'
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
    
    # Validate API key
    api_key = args.api_key or settings.openai_api_key
    if not api_key:
        logger.error("OpenAI API key not found. Set OPENAI_API_KEY in .env or use --api-key")
        return 1
    
    # Update settings if custom API key provided
    if args.api_key:
        settings.openai_api_key = args.api_key
    
    try:
        # Run experiment
        results = run_experiment(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            test_split=args.test_split,
            max_samples=args.max_samples,
            model=args.model,
            temperature=args.temperature,
            save_predictions_flag=not args.no_save_predictions
        )
        
        return 0
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())