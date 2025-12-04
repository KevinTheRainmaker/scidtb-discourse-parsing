"""
Data preparation script for SciDTB discourse parsing.

This script:
1. Loads raw SciDTB data from train/test splits
2. Validates the data structure
3. Computes statistics
4. Prepares processed datasets for training and evaluation
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import SciDTBLoader
from src.data.preprocessor import DataPreprocessor
from src.models import DiscourseTreeModel
from src.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


def validate_data_structure(data_dir: Path) -> bool:
    """
    Validate that the data directory has the expected structure.
    
    Expected structure:
        data_dir/
        ├── train/
        │   └── *.edu.txt.dep
        └── test/
            ├── gold/
            │   └── *.edu.txt.dep
            ├── second_annotate/
            │   └── *.edu.txt.dep
            └── edu/
                └── *.edu
    
    Args:
        data_dir: Root data directory
        
    Returns:
        True if structure is valid
    """
    logger.info("Validating data structure...")
    
    required_paths = [
        data_dir / "train",
        data_dir / "test" / "gold"
    ]
    
    for path in required_paths:
        if not path.exists():
            logger.error(f"Required path does not exist: {path}")
            return False
        
        # Check if there are any .dep files
        dep_files = list(path.glob("*.edu.txt.dep"))
        if not dep_files:
            logger.error(f"No .edu.txt.dep files found in: {path}")
            return False
        
        logger.info(f"✓ Found {len(dep_files)} files in {path.name}")
    
    logger.info("✓ Data structure is valid")
    return True


def load_and_validate_split(
    loader: SciDTBLoader,
    split: str,
    min_edus: int = 3
) -> List[DiscourseTreeModel]:
    """
    Load and validate a data split.
    
    Args:
        loader: SciDTBLoader instance
        split: Split name (e.g., 'train', 'test/gold')
        min_edus: Minimum number of EDUs per tree
        
    Returns:
        List of valid discourse trees
    """
    logger.info(f"\nLoading {split} split...")
    
    trees = loader.load_split(split)
    
    if not trees:
        logger.warning(f"No trees loaded from {split}")
        return []
    
    # Filter by minimum EDUs
    trees = DataPreprocessor.filter_by_length(trees, min_edus=min_edus)
    
    # Validate each tree
    valid_trees = []
    invalid_count = 0
    
    for i, tree in enumerate(trees):
        try:
            # Re-validate by reconstructing
            tree_dict = tree.to_dict()
            DiscourseTreeModel.from_dict(tree_dict)
            valid_trees.append(tree)
        except Exception as e:
            logger.warning(f"Invalid tree at index {i}: {e}")
            invalid_count += 1
    
    logger.info(f"✓ Loaded {len(valid_trees)} valid trees ({invalid_count} invalid)")
    
    return valid_trees


def compute_statistics(trees: List[DiscourseTreeModel], split_name: str) -> Dict:
    """
    Compute and display statistics for a dataset split.
    
    Args:
        trees: List of discourse trees
        split_name: Name of the split
        
    Returns:
        Statistics dictionary
    """
    logger.info(f"\nComputing statistics for {split_name}...")
    
    if not trees:
        logger.warning(f"No trees to compute statistics for {split_name}")
        return {}
    
    # Basic statistics
    num_trees = len(trees)
    num_edus = sum(len(tree.edus) - 1 for tree in trees)  # Exclude ROOT
    avg_edus = num_edus / num_trees if num_trees > 0 else 0
    
    # Depth statistics
    all_depths = []
    for tree in trees:
        for edu in tree.edus:
            if edu.id > 0:  # Exclude ROOT
                depth = tree.get_depth(edu.id)
                all_depths.append(depth)
    
    max_depth = max(all_depths) if all_depths else 0
    avg_depth = sum(all_depths) / len(all_depths) if all_depths else 0
    
    # Relation statistics
    relation_counts = DataPreprocessor.get_relation_statistics(trees)
    
    stats = {
        'num_trees': num_trees,
        'num_edus': num_edus,
        'avg_edus_per_tree': round(avg_edus, 2),
        'max_depth': max_depth,
        'avg_depth': round(avg_depth, 2),
        'unique_relations': len(relation_counts),
        'relation_counts': relation_counts
    }
    
    # Display statistics
    print(f"\n{'='*60}")
    print(f"{split_name} Statistics".center(60))
    print(f"{'='*60}")
    print(f"Number of trees:        {stats['num_trees']:>8}")
    print(f"Total EDUs:             {stats['num_edus']:>8}")
    print(f"Average EDUs per tree:  {stats['avg_edus_per_tree']:>8.2f}")
    print(f"Maximum tree depth:     {stats['max_depth']:>8}")
    print(f"Average tree depth:     {stats['avg_depth']:>8.2f}")
    print(f"Unique relation types:  {stats['unique_relations']:>8}")
    print(f"{'='*60}")
    
    # Display top relations
    print(f"\nTop 10 Relation Types:")
    print(f"{'-'*60}")
    sorted_relations = sorted(
        relation_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )
    for relation, count in sorted_relations[:10]:
        percentage = (count / num_edus * 100) if num_edus > 0 else 0
        print(f"  {relation:>25}: {count:>6} ({percentage:>5.2f}%)")
    print(f"{'='*60}\n")
    
    return stats


def save_processed_data(
    trees: List[DiscourseTreeModel],
    output_file: Path,
    include_text: bool = True
):
    """
    Save processed trees to a JSON file.
    
    Args:
        trees: List of discourse trees
        output_file: Output file path
        include_text: Whether to include extracted text
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    data = []
    for tree in trees:
        tree_data = tree.to_dict()
        
        if include_text:
            # Extract text from EDUs (excluding ROOT)
            text = ' '.join(
                edu.text.strip()
                for edu in tree.edus
                if edu.id > 0 and edu.text.strip() != "ROOT"
            )
            tree_data['full_text'] = text
        
        data.append(tree_data)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ Saved {len(data)} trees to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare SciDTB data for training and evaluation"
    )
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
        help='Output directory for processed data'
    )
    parser.add_argument(
        '--min-edus',
        type=int,
        default=3,
        help='Minimum number of EDUs per tree'
    )
    parser.add_argument(
        '--save-processed',
        action='store_true',
        help='Save processed trees to JSON files'
    )
    parser.add_argument(
        '--stats-only',
        action='store_true',
        help='Only compute and display statistics'
    )
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
    logger.info("SciDTB Data Preparation".center(60))
    logger.info("="*60)
    
    # Validate data structure
    if not validate_data_structure(args.data_dir):
        logger.error("Data structure validation failed")
        return 1
    
    # Initialize loader
    loader = SciDTBLoader(args.data_dir)
    
    # Load and validate splits
    splits_data = {}
    
    # Train split
    train_trees = load_and_validate_split(loader, 'train', args.min_edus)
    if train_trees:
        splits_data['train'] = train_trees
    
    # Test split (gold annotations)
    test_trees = load_and_validate_split(loader, 'test/gold', args.min_edus)
    if test_trees:
        splits_data['test'] = test_trees
    
    # Optional: Second annotator (if exists)
    test_second_dir = args.data_dir / 'test' / 'second_annotate'
    if test_second_dir.exists():
        test_second_trees = load_and_validate_split(
            loader, 'test/second_annotate', args.min_edus
        )
        if test_second_trees:
            splits_data['test_second'] = test_second_trees
    
    # Optional: Dev split (if exists)
    dev_dir = args.data_dir / 'dev'
    if dev_dir.exists():
        dev_trees = load_and_validate_split(loader, 'dev', args.min_edus)
        if dev_trees:
            splits_data['dev'] = dev_trees
    
    # Compute statistics for each split
    all_stats = {}
    for split_name, trees in splits_data.items():
        stats = compute_statistics(trees, split_name.upper())
        all_stats[split_name] = stats
    
    # Save processed data if requested
    if args.save_processed and not args.stats_only:
        logger.info("\nSaving processed data...")
        for split_name, trees in splits_data.items():
            output_file = args.output_dir / f"{split_name}.json"
            save_processed_data(trees, output_file)
    
    # Save statistics
    stats_file = args.output_dir / 'statistics.json'
    stats_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ Statistics saved to {stats_file}")
    
    # Summary
    print(f"\n{'='*60}")
    print("Data Preparation Complete".center(60))
    print(f"{'='*60}")
    print(f"Total splits processed: {len(splits_data)}")
    for split_name, trees in splits_data.items():
        print(f"  {split_name:>15}: {len(trees):>6} trees")
    print(f"{'='*60}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())