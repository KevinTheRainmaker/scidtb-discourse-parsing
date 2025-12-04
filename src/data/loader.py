"""
Data loading utilities for SciDTB dataset.
"""
import json
from pathlib import Path
from typing import List, Optional
from ..models import DiscourseTreeModel
from ..utils.logger import get_logger

logger = get_logger(__name__)

class SciDTBLoader:
    """Loader for SciDTB dataset."""
    
    def __init__(self, data_dir: Path):
        """
        Initialize loader.
        
        Args:
            data_dir: Root directory containing SciDTB data
        """
        self.data_dir = Path(data_dir)
        
        if not self.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {data_dir}")
    
    def load_file(self, file_path: Path) -> Optional[DiscourseTreeModel]:
        """
        Load a single SciDTB file.
        
        Args:
            file_path: Path to .edu.txt.dep file
            
        Returns:
            DiscourseTreeModel or None if loading fails
        """
        try:
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                data = json.load(f)
            
            tree = DiscourseTreeModel.from_scidtb(data)
            return tree
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None
    
    def load_split(self, split: str = 'train') -> List[DiscourseTreeModel]:
        """
        Load a data split.
        
        Args:
            split: Split name ('train', 'test/gold', 'test/second_annotate')
            
        Returns:
            List of DiscourseTreeModel instances
        """
        split_dir = self.data_dir / split
        
        if not split_dir.exists():
            logger.warning(f"Split directory does not exist: {split_dir}")
            return []
        
        trees = []
        files_loaded = 0
        files_failed = 0
        
        for file_path in split_dir.glob("*.edu.txt.dep"):
            tree = self.load_file(file_path)
            if tree is not None:
                trees.append(tree)
                files_loaded += 1
            else:
                files_failed += 1
        
        logger.info(
            f"Loaded {split} split: {files_loaded} files loaded, "
            f"{files_failed} files failed"
        )
        
        return trees
    
    def load_all_splits(self) -> dict:
        """
        Load all available splits.
        
        Returns:
            Dictionary mapping split names to lists of trees
        """
        splits = {}
        
        # Standard splits
        for split_name in ['train', 'test/gold', 'test/second_annotate']:
            trees = self.load_split(split_name)
            if trees:
                splits[split_name] = trees
        
        return splits
    
    @staticmethod
    def extract_text(tree: DiscourseTreeModel) -> str:
        """
        Extract full text from a discourse tree.
        
        Args:
            tree: DiscourseTreeModel instance
            
        Returns:
            Concatenated text from all EDUs (excluding ROOT)
        """
        edu_texts = [
            edu.text.strip() 
            for edu in tree.edus 
            if edu.id > 0 and edu.text.strip() != "ROOT"
        ]
        return ' '.join(edu_texts)

def load_dataset(data_dir: Path, split: str = 'train') -> List[DiscourseTreeModel]:
    """
    Convenience function to load a dataset split.
    
    Args:
        data_dir: Root directory containing SciDTB data
        split: Split name to load
        
    Returns:
        List of DiscourseTreeModel instances
    """
    loader = SciDTBLoader(data_dir)
    return loader.load_split(split)