"""
Evaluation script for discourse dependency parsing models.

This script:
1. Loads gold test data
2. Loads predicted trees (or uses a parser to generate them)
3. Computes evaluation metrics (UAS, LAS, F1)
4. Generates detailed reports
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional, Dict
import sys
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import SciDTBLoader
from src.models import DiscourseTreeModel
from src.evaluation.metrics import (
    DiscourseEvaluator,
    RelationEvaluator,
    ParseMetrics
)
from src.utils.logger import get_logger, setup_logging
from config.settings import settings

logger = get_logger(__name__)


def load_predictions(pred_file: Path) -> List[DiscourseTreeModel]:
    """
    Load predicted discourse trees from file.
    
    Expected format: JSON file with list of tree dictionaries
    
    Args:
        pred_file: Path to predictions file
        
    Returns:
        List of predicted discourse trees
    """
    logger.info(f"Loading predictions from {pred_file}")
    
    with open(pred_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    trees = []
    for i, tree_data in enumerate(data):
        try:
            tree = DiscourseTreeModel.from_dict(tree_data)
            trees.append(tree)
        except Exception as e:
            logger.warning(f"Failed to load prediction {i}: {e}")
    
    logger.info(f"✓ Loaded {len(trees)} predicted trees")
    return trees


def evaluate_predictions(
    gold_trees: List[DiscourseTreeModel],
    pred_trees: List[DiscourseTreeModel],
    output_dir: Path,
    save_detailed: bool = True,
    save_alignments: bool = False
) -> ParseMetrics:
    """
    Evaluate predictions against gold standard.
    
    Args:
        gold_trees: Gold standard trees
        pred_trees: Predicted trees
        output_dir: Output directory for results
        save_detailed: Whether to save detailed results
        save_alignments: Whether to save EDU alignments
        
    Returns:
        ParseMetrics object
    """
    logger.info("="*60)
    logger.info("Evaluating Predictions")
    logger.info("="*60)
    
    if len(gold_trees) != len(pred_trees):
        logger.warning(
            f"Number of gold trees ({len(gold_trees)}) != "
            f"number of predicted trees ({len(pred_trees)})"
        )
        # Take minimum
        min_len = min(len(gold_trees), len(pred_trees))
        gold_trees = gold_trees[:min_len]
        pred_trees = pred_trees[:min_len]
        logger.info(f"Evaluating first {min_len} trees")
    
    # Run evaluation
    evaluator = DiscourseEvaluator(include_root=False)
    
    logger.info(f"\nEvaluating {len(gold_trees)} trees...")
    metrics = evaluator.evaluate_batch(
        gold_trees=gold_trees,
        pred_trees=pred_trees,
        verbose=True
    )
    
    # Display results
    metrics.print_summary()
    
    # Display detailed alignments (first 10 EDUs)
    if save_detailed:
        metrics.print_detailed_alignments(max_display=10)
    
    # Evaluate per-relation performance
    logger.info("Computing relation-level metrics...")
    relation_metrics = RelationEvaluator.evaluate_relations(gold_trees, pred_trees)
    RelationEvaluator.print_relation_report(relation_metrics)
    
    # Save results
    if save_detailed:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_dir / f"evaluation_results_{timestamp}.json"
        
        results = {
            "summary": metrics.to_dict(),
            "relation_metrics": relation_metrics,
            "metadata": {
                "num_trees": len(gold_trees),
                "timestamp": timestamp
            }
        }
        
        # Save alignments if requested
        if save_alignments:
            alignments_data = []
            for alignment in metrics.alignments[:100]:  # Limit to first 100
                alignments_data.append({
                    "edu_id": alignment.edu_id,
                    "gold_parent": alignment.gold_edu.parent if alignment.gold_edu else None,
                    "pred_parent": alignment.pred_edu.parent if alignment.pred_edu else None,
                    "gold_relation": alignment.gold_edu.relation if alignment.gold_edu else None,
                    "pred_relation": alignment.pred_edu.relation if alignment.pred_edu else None,
                    "parent_match": alignment.parent_match,
                    "full_match": alignment.full_match
                })
            results["alignments_sample"] = alignments_data
        
        results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Results saved to {results_file}")
    
    return metrics


def compare_models(
    gold_trees: List[DiscourseTreeModel],
    predictions_dict: Dict[str, List[DiscourseTreeModel]],
    output_dir: Path
):
    """
    Compare multiple models.
    
    Args:
        gold_trees: Gold standard trees
        predictions_dict: Dictionary mapping model names to predictions
        output_dir: Output directory for comparison results
    """
    logger.info("="*60)
    logger.info("Comparing Multiple Models")
    logger.info("="*60)
    
    evaluator = DiscourseEvaluator(include_root=False)
    
    comparison_results = {}
    
    for model_name, pred_trees in predictions_dict.items():
        logger.info(f"\nEvaluating {model_name}...")
        
        # Align lengths
        min_len = min(len(gold_trees), len(pred_trees))
        gold_subset = gold_trees[:min_len]
        pred_subset = pred_trees[:min_len]
        
        metrics = evaluator.evaluate_batch(gold_subset, pred_subset, verbose=False)
        
        comparison_results[model_name] = metrics.to_dict()
        
        print(f"\n{model_name}:")
        print(f"  UAS: {metrics.uas*100:.2f}%")
        print(f"  LAS: {metrics.las*100:.2f}%")
        print(f"  F1:  {metrics.f1*100:.2f}%")
    
    # Create comparison table
    print(f"\n{'='*80}")
    print(f"{'Model Comparison':^80}")
    print(f"{'='*80}")
    print(f"{'Model':>30} | {'UAS':>10} | {'LAS':>10} | {'F1':>10} | {'Samples':>8}")
    print(f"{'-'*80}")
    
    for model_name, results in comparison_results.items():
        print(
            f"{model_name:>30} | "
            f"{results['uas']:>9.2f}% | "
            f"{results['las']:>9.2f}% | "
            f"{results['f1']:>9.2f}% | "
            f"{results['total_edus']:>8}"
        )
    
    print(f"{'='*80}\n")
    
    # Save comparison
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_file = output_dir / f"model_comparison_{timestamp}.json"
    
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ Comparison saved to {comparison_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate discourse dependency parsing predictions"
    )
    
    # Data arguments
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('./data/raw'),
        help='Root directory containing raw SciDTB data'
    )
    parser.add_argument(
        '--test-split',
        type=str,
        default='test/gold',
        help='Test split name (e.g., test/gold, test/second_annotate)'
    )
    parser.add_argument(
        '--predictions',
        type=Path,
        required=True,
        help='Path to predictions file (JSON format)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('./data/results'),
        help='Output directory for evaluation results'
    )
    
    # Evaluation options
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to evaluate'
    )
    parser.add_argument(
        '--save-detailed',
        action='store_true',
        help='Save detailed evaluation results'
    )
    parser.add_argument(
        '--save-alignments',
        action='store_true',
        help='Save EDU-level alignments'
    )
    parser.add_argument(
        '--no-relations',
        action='store_true',
        help='Skip per-relation evaluation'
    )
    
    # Comparison mode
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare multiple prediction files'
    )
    parser.add_argument(
        '--prediction-files',
        type=Path,
        nargs='+',
        help='Multiple prediction files for comparison (use with --compare)'
    )
    parser.add_argument(
        '--model-names',
        type=str,
        nargs='+',
        help='Names for models (use with --compare)'
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
    
    logger.info("="*60)
    logger.info("SciDTB Evaluation")
    logger.info("="*60)
    
    try:
        # Load gold data
        logger.info(f"\nLoading gold data from {args.test_split}...")
        loader = SciDTBLoader(args.data_dir)
        gold_trees = loader.load_split(args.test_split)
        
        if not gold_trees:
            logger.error(f"No gold trees found in {args.test_split}")
            return 1
        
        logger.info(f"✓ Loaded {len(gold_trees)} gold trees")
        
        # Limit samples if requested
        if args.max_samples:
            gold_trees = gold_trees[:args.max_samples]
            logger.info(f"Limited to {len(gold_trees)} samples")
        
        # Comparison mode
        if args.compare:
            if not args.prediction_files:
                logger.error("--prediction-files required for comparison mode")
                return 1
            
            # Load all predictions
            predictions_dict = {}
            
            for i, pred_file in enumerate(args.prediction_files):
                model_name = (
                    args.model_names[i] 
                    if args.model_names and i < len(args.model_names)
                    else f"Model_{i+1}"
                )
                
                pred_trees = load_predictions(pred_file)
                predictions_dict[model_name] = pred_trees
            
            # Compare models
            compare_models(gold_trees, predictions_dict, args.output_dir)
        
        # Single evaluation mode
        else:
            # Load predictions
            pred_trees = load_predictions(args.predictions)
            
            # Evaluate
            metrics = evaluate_predictions(
                gold_trees=gold_trees,
                pred_trees=pred_trees,
                output_dir=args.output_dir,
                save_detailed=args.save_detailed,
                save_alignments=args.save_alignments
            )
            
            # Print summary to stdout (for easy parsing)
            print(f"\nFinal Results:")
            print(f"UAS: {metrics.uas*100:.2f}%")
            print(f"LAS: {metrics.las*100:.2f}%")
            print(f"F1:  {metrics.f1*100:.2f}%")
        
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())