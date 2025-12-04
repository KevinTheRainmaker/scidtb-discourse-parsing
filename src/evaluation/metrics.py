"""
Evaluation metrics for discourse dependency parsing.
"""
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from ..models import DiscourseTreeModel, EDUModel
from ..utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class EDUAlignment:
    """Alignment between gold and predicted EDU."""
    edu_id: int
    gold_edu: Optional[EDUModel]
    pred_edu: Optional[EDUModel]
    gold_text: str
    pred_text: str
    
    @property
    def is_matched(self) -> bool:
        """Check if EDU is matched in both gold and prediction."""
        return self.gold_edu is not None and self.pred_edu is not None
    
    @property
    def parent_match(self) -> bool:
        """Check if parent matches (for UAS)."""
        if not self.is_matched:
            return False
        return self.gold_edu.parent == self.pred_edu.parent
    
    @property
    def full_match(self) -> bool:
        """Check if both parent and relation match (for LAS)."""
        if not self.parent_match:
            return False
        return self.gold_edu.relation == self.pred_edu.relation

@dataclass
class ParseMetrics:
    """Container for parsing evaluation metrics."""
    
    # Basic counts
    total_edus: int
    matched_edus: int
    
    # Attachment scores
    uas_correct: int
    las_correct: int
    
    # Computed metrics
    uas: float
    las: float
    f1: float
    
    # Detailed information
    alignments: List[EDUAlignment]
    
    @property
    def precision(self) -> float:
        """Precision = correct predictions / total predictions."""
        if self.matched_edus == 0:
            return 0.0
        return self.las_correct / self.matched_edus
    
    @property
    def recall(self) -> float:
        """Recall = correct predictions / total gold EDUs."""
        if self.total_edus == 0:
            return 0.0
        return self.las_correct / self.total_edus
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'total_edus': self.total_edus,
            'matched_edus': self.matched_edus,
            'uas_correct': self.uas_correct,
            'las_correct': self.las_correct,
            'uas': round(self.uas * 100, 2),  # Convert to percentage
            'las': round(self.las * 100, 2),  # Convert to percentage
            'f1': round(self.f1 * 100, 2),    # Convert to percentage
            'precision': round(self.precision * 100, 2),
            'recall': round(self.recall * 100, 2)
        }
    
    def print_summary(self, title: str = "Evaluation Results"):
        """Print formatted summary."""
        print(f"\n{'='*60}")
        print(f"{title:^60}")
        print(f"{'='*60}")
        print(f"Total EDUs:           {self.total_edus:>6}")
        print(f"Matched EDUs:         {self.matched_edus:>6}")
        print(f"-" * 60)
        print(f"UAS (Unlabeled):      {self.uas*100:>5.2f}% ({self.uas_correct}/{self.total_edus})")
        print(f"LAS (Labeled):        {self.las*100:>5.2f}% ({self.las_correct}/{self.total_edus})")
        print(f"F1 Score:             {self.f1*100:>5.2f}%")
        print(f"-" * 60)
        print(f"Precision:            {self.precision*100:>5.2f}%")
        print(f"Recall:               {self.recall*100:>5.2f}%")
        print(f"{'='*60}\n")
    
    def print_detailed_alignments(self, max_display: int = 10):
        """Print detailed EDU-level alignments."""
        print(f"\n{'='*80}")
        print(f"{'Detailed EDU Alignments':^80}")
        print(f"{'='*80}")
        print(f"{'EDU':>4} | {'Gold':>10} | {'Pred':>10} | {'Gold Rel':>15} | {'Pred Rel':>15} | {'UAS':>3} | {'LAS':>3}")
        print(f"{'-'*80}")
        
        displayed = 0
        for alignment in self.alignments:
            if displayed >= max_display:
                remaining = len(self.alignments) - displayed
                print(f"... and {remaining} more EDUs")
                break
            
            if alignment.edu_id == 0:  # Skip ROOT
                continue
            
            if not alignment.is_matched:
                print(f"{alignment.edu_id:>4} | {'MISSING':>10} | {'MISSING':>10} | {'-':>15} | {'-':>15} | {'✘':>3} | {'✘':>3}")
                displayed += 1
                continue
            
            gold_parent = alignment.gold_edu.parent
            pred_parent = alignment.pred_edu.parent
            gold_rel = alignment.gold_edu.relation[:15]  # Truncate if too long
            pred_rel = alignment.pred_edu.relation[:15]
            
            uas_mark = '✔' if alignment.parent_match else '✘'
            las_mark = '✔' if alignment.full_match else '✘'
            
            print(f"{alignment.edu_id:>4} | {gold_parent:>10} | {pred_parent:>10} | {gold_rel:>15} | {pred_rel:>15} | {uas_mark:>3} | {las_mark:>3}")
            displayed += 1
        
        print(f"{'='*80}\n")

class DiscourseEvaluator:
    """Evaluator for discourse dependency parsing."""
    
    def __init__(self, include_root: bool = False):
        """
        Initialize evaluator.
        
        Args:
            include_root: Whether to include ROOT in evaluation (typically False)
        """
        self.include_root = include_root
    
    def align_edus(
        self,
        gold_tree: DiscourseTreeModel,
        pred_tree: DiscourseTreeModel
    ) -> List[EDUAlignment]:
        """
        Align EDUs between gold and predicted trees.
        
        Args:
            gold_tree: Gold standard tree
            pred_tree: Predicted tree
            
        Returns:
            List of EDU alignments
        """
        alignments = []
        
        # Create dictionaries for fast lookup
        gold_dict = {edu.id: edu for edu in gold_tree.edus}
        pred_dict = {edu.id: edu for edu in pred_tree.edus}
        
        # Get all EDU IDs from both trees
        all_ids = set(gold_dict.keys()) | set(pred_dict.keys())
        
        for edu_id in sorted(all_ids):
            # Skip ROOT if not included
            if not self.include_root and edu_id == 0:
                continue
            
            gold_edu = gold_dict.get(edu_id)
            pred_edu = pred_dict.get(edu_id)
            
            alignment = EDUAlignment(
                edu_id=edu_id,
                gold_edu=gold_edu,
                pred_edu=pred_edu,
                gold_text=gold_edu.text if gold_edu else "",
                pred_text=pred_edu.text if pred_edu else ""
            )
            
            alignments.append(alignment)
        
        return alignments
    
    def calculate_uas(
        self,
        gold_tree: DiscourseTreeModel,
        pred_tree: DiscourseTreeModel
    ) -> float:
        """
        Calculate Unlabeled Attachment Score (UAS).
        
        UAS measures the percentage of EDUs that have the correct parent,
        regardless of the relation label.
        
        Args:
            gold_tree: Gold standard tree
            pred_tree: Predicted tree
            
        Returns:
            UAS as a percentage (0.0 to 1.0)
        """
        alignments = self.align_edus(gold_tree, pred_tree)
        
        matched = [a for a in alignments if a.is_matched]
        
        if not matched:
            return 0.0
        
        correct = sum(1 for a in matched if a.parent_match)
        
        return correct / len(matched)
    
    def calculate_las(
        self,
        gold_tree: DiscourseTreeModel,
        pred_tree: DiscourseTreeModel
    ) -> float:
        """
        Calculate Labeled Attachment Score (LAS).
        
        LAS measures the percentage of EDUs that have both the correct parent
        AND the correct relation label.
        
        Args:
            gold_tree: Gold standard tree
            pred_tree: Predicted tree
            
        Returns:
            LAS as a percentage (0.0 to 1.0)
        """
        alignments = self.align_edus(gold_tree, pred_tree)
        
        matched = [a for a in alignments if a.is_matched]
        
        if not matched:
            return 0.0
        
        correct = sum(1 for a in matched if a.full_match)
        
        return correct / len(matched)
    
    def calculate_f1(
        self,
        gold_tree: DiscourseTreeModel,
        pred_tree: DiscourseTreeModel
    ) -> float:
        """
        Calculate F1 score based on LAS.
        
        F1 = 2 * (precision * recall) / (precision + recall)
        where precision and recall are based on correctly predicted
        parent-relation pairs.
        
        Args:
            gold_tree: Gold standard tree
            pred_tree: Predicted tree
            
        Returns:
            F1 score (0.0 to 1.0)
        """
        alignments = self.align_edus(gold_tree, pred_tree)
        
        # Total gold EDUs (excluding ROOT if specified)
        gold_edus = [a for a in alignments if a.gold_edu is not None]
        # Total predicted EDUs
        pred_edus = [a for a in alignments if a.pred_edu is not None]
        # Correctly predicted (both parent and relation)
        correct = [a for a in alignments if a.full_match]
        
        if not gold_edus or not pred_edus:
            return 0.0
        
        precision = len(correct) / len(pred_edus) if pred_edus else 0.0
        recall = len(correct) / len(gold_edus) if gold_edus else 0.0
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        
        return f1
    
    def evaluate_single(
        self,
        gold_tree: DiscourseTreeModel,
        pred_tree: DiscourseTreeModel
    ) -> ParseMetrics:
        """
        Comprehensive evaluation of a single prediction.
        
        Args:
            gold_tree: Gold standard tree
            pred_tree: Predicted tree
            
        Returns:
            ParseMetrics object with all evaluation results
        """
        alignments = self.align_edus(gold_tree, pred_tree)
        
        # Filter matched EDUs
        matched = [a for a in alignments if a.is_matched]
        
        # Count correct attachments
        uas_correct = sum(1 for a in matched if a.parent_match)
        las_correct = sum(1 for a in matched if a.full_match)
        
        # Calculate scores
        total = len(matched)
        uas = uas_correct / total if total > 0 else 0.0
        las = las_correct / total if total > 0 else 0.0
        f1 = self.calculate_f1(gold_tree, pred_tree)
        
        return ParseMetrics(
            total_edus=total,
            matched_edus=len(matched),
            uas_correct=uas_correct,
            las_correct=las_correct,
            uas=uas,
            las=las,
            f1=f1,
            alignments=alignments
        )
    
    def evaluate_batch(
        self,
        gold_trees: List[DiscourseTreeModel],
        pred_trees: List[DiscourseTreeModel],
        verbose: bool = True
    ) -> ParseMetrics:
        """
        Evaluate multiple predictions and aggregate results.
        
        Args:
            gold_trees: List of gold standard trees
            pred_trees: List of predicted trees
            verbose: Whether to print progress
            
        Returns:
            Aggregated ParseMetrics
        """
        if len(gold_trees) != len(pred_trees):
            raise ValueError(
                f"Number of gold trees ({len(gold_trees)}) does not match "
                f"number of predicted trees ({len(pred_trees)})"
            )
        
        # Aggregate counts
        total_edus = 0
        total_matched = 0
        total_uas_correct = 0
        total_las_correct = 0
        all_alignments = []
        
        for i, (gold_tree, pred_tree) in enumerate(zip(gold_trees, pred_trees)):
            if verbose and (i + 1) % 10 == 0:
                logger.info(f"Evaluated {i + 1}/{len(gold_trees)} trees")
            
            metrics = self.evaluate_single(gold_tree, pred_tree)
            
            total_edus += metrics.total_edus
            total_matched += metrics.matched_edus
            total_uas_correct += metrics.uas_correct
            total_las_correct += metrics.las_correct
            all_alignments.extend(metrics.alignments)
        
        # Calculate aggregated scores
        uas = total_uas_correct / total_edus if total_edus > 0 else 0.0
        las = total_las_correct / total_edus if total_edus > 0 else 0.0
        
        # Calculate F1 from aggregated precision and recall
        precision = total_las_correct / total_matched if total_matched > 0 else 0.0
        recall = total_las_correct / total_edus if total_edus > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        if verbose:
            logger.info(f"Evaluation complete: {len(gold_trees)} trees processed")
        
        return ParseMetrics(
            total_edus=total_edus,
            matched_edus=total_matched,
            uas_correct=total_uas_correct,
            las_correct=total_las_correct,
            uas=uas,
            las=las,
            f1=f1,
            alignments=all_alignments
        )

class RelationEvaluator:
    """Evaluator for discourse relation classification."""
    
    @staticmethod
    def evaluate_relations(
        gold_trees: List[DiscourseTreeModel],
        pred_trees: List[DiscourseTreeModel]
    ) -> Dict[str, Dict]:
        """
        Evaluate relation classification performance per relation type.
        
        Args:
            gold_trees: List of gold standard trees
            pred_trees: List of predicted trees
            
        Returns:
            Dictionary mapping relation types to their metrics
        """
        from collections import defaultdict
        
        # Collect per-relation statistics
        relation_stats = defaultdict(lambda: {
            'true_positive': 0,
            'false_positive': 0,
            'false_negative': 0
        })
        
        evaluator = DiscourseEvaluator(include_root=False)
        
        for gold_tree, pred_tree in zip(gold_trees, pred_trees):
            alignments = evaluator.align_edus(gold_tree, pred_tree)
            
            for alignment in alignments:
                if not alignment.is_matched:
                    continue
                
                gold_rel = alignment.gold_edu.relation
                pred_rel = alignment.pred_edu.relation
                
                if gold_rel == pred_rel:
                    # True positive for this relation
                    relation_stats[gold_rel]['true_positive'] += 1
                else:
                    # False negative for gold relation
                    relation_stats[gold_rel]['false_negative'] += 1
                    # False positive for predicted relation
                    relation_stats[pred_rel]['false_positive'] += 1
        
        # Calculate metrics for each relation
        results = {}
        for relation, stats in relation_stats.items():
            tp = stats['true_positive']
            fp = stats['false_positive']
            fn = stats['false_negative']
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            results[relation] = {
                'precision': round(precision * 100, 2),
                'recall': round(recall * 100, 2),
                'f1': round(f1 * 100, 2),
                'support': tp + fn  # Total gold instances
            }
        
        return results
    
    @staticmethod
    def print_relation_report(relation_metrics: Dict[str, Dict]):
        """Print detailed relation classification report."""
        print(f"\n{'='*80}")
        print(f"{'Relation Classification Report':^80}")
        print(f"{'='*80}")
        print(f"{'Relation':>25} | {'Precision':>10} | {'Recall':>10} | {'F1':>10} | {'Support':>8}")
        print(f"{'-'*80}")
        
        # Sort by support (most common relations first)
        sorted_relations = sorted(
            relation_metrics.items(),
            key=lambda x: x[1]['support'],
            reverse=True
        )
        
        total_support = sum(m['support'] for _, m in sorted_relations)
        weighted_f1 = sum(m['f1'] * m['support'] for _, m in sorted_relations) / total_support if total_support > 0 else 0.0
        
        for relation, metrics in sorted_relations:
            print(
                f"{relation:>25} | "
                f"{metrics['precision']:>9.2f}% | "
                f"{metrics['recall']:>9.2f}% | "
                f"{metrics['f1']:>9.2f}% | "
                f"{metrics['support']:>8}"
            )
        
        print(f"{'-'*80}")
        print(f"{'Weighted Average':>25} | {' ':>10} | {' ':>10} | {weighted_f1:>9.2f}% | {total_support:>8}")
        print(f"{'='*80}\n")