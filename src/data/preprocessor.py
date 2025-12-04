"""
Data preprocessing utilities.
"""
from typing import List, Dict, Tuple
from ..models import DiscourseTreeModel
from ..utils.logger import get_logger

logger = get_logger(__name__)

class DataPreprocessor:
    """Preprocess discourse trees for training and evaluation."""
    
    @staticmethod
    def filter_by_length(
        trees: List[DiscourseTreeModel],
        min_edus: int = 3,
        max_edus: int = 50
    ) -> List[DiscourseTreeModel]:
        """
        Filter trees by number of EDUs.
        
        Args:
            trees: List of discourse trees
            min_edus: Minimum number of EDUs
            max_edus: Maximum number of EDUs
            
        Returns:
            Filtered list of trees
        """
        filtered = [
            tree for tree in trees
            if min_edus <= len(tree.edus) <= max_edus
        ]
        
        logger.info(
            f"Filtered {len(trees)} trees to {len(filtered)} "
            f"(min_edus={min_edus}, max_edus={max_edus})"
        )
        
        return filtered
    
    @staticmethod
    def get_relation_statistics(
        trees: List[DiscourseTreeModel]
    ) -> Dict[str, int]:
        """
        Get statistics on relation types.
        
        Args:
            trees: List of discourse trees
            
        Returns:
            Dictionary mapping relation types to counts
        """
        relation_counts: Dict[str, int] = {}
        
        for tree in trees:
            for edu in tree.edus:
                if edu.id > 0:  # Skip ROOT
                    rel = edu.relation
                    relation_counts[rel] = relation_counts.get(rel, 0) + 1
        
        return relation_counts
    
    @staticmethod
    def split_data(
        trees: List[DiscourseTreeModel],
        train_ratio: float = 0.8,
        seed: int = 42
    ) -> Tuple[List[DiscourseTreeModel], List[DiscourseTreeModel]]:
        """
        Split data into train and validation sets.
        
        Args:
            trees: List of discourse trees
            train_ratio: Ratio of training data
            seed: Random seed
            
        Returns:
            Tuple of (train_trees, val_trees)
        """
        import random
        random.seed(seed)
        
        shuffled = trees.copy()
        random.shuffle(shuffled)
        
        split_idx = int(len(shuffled) * train_ratio)
        train_trees = shuffled[:split_idx]
        val_trees = shuffled[split_idx:]
        
        logger.info(
            f"Split {len(trees)} trees into "
            f"{len(train_trees)} train, {len(val_trees)} validation"
        )
        
        return train_trees, val_trees