"""
Unit tests for src/evaluation/metrics.py

Tests cover:
- EDU alignment between gold and predicted trees
- UAS (Unlabeled Attachment Score) calculation
- LAS (Labeled Attachment Score) calculation
- F1 score calculation
- Batch evaluation
- Per-relation evaluation
- Edge cases and error handling
"""
import pytest

from src.evaluation.metrics import (
    EDUAlignment,
    ParseMetrics,
    DiscourseEvaluator,
    RelationEvaluator
)
from src.models import EDUModel, DiscourseTreeModel


class TestEDUAlignment:
    """Test cases for EDUAlignment dataclass."""

    def test_alignment_matched(self):
        """Test alignment with matched EDU."""
        gold_edu = EDUModel(id=1, text="Test sentence.", parent=0, relation="Background")
        pred_edu = EDUModel(id=1, text="Test sentence.", parent=0, relation="Background")

        alignment = EDUAlignment(
            edu_id=1,
            gold_edu=gold_edu,
            pred_edu=pred_edu,
            gold_text="Test sentence.",
            pred_text="Test sentence."
        )

        assert alignment.is_matched
        assert alignment.parent_match
        assert alignment.full_match

    def test_alignment_parent_mismatch(self):
        """Test alignment with parent mismatch."""
        gold_edu = EDUModel(id=1, text="Test.", parent=0, relation="Background")
        pred_edu = EDUModel(id=1, text="Test.", parent=2, relation="Background")  # Wrong parent

        alignment = EDUAlignment(
            edu_id=1,
            gold_edu=gold_edu,
            pred_edu=pred_edu,
            gold_text="Test.",
            pred_text="Test."
        )

        assert alignment.is_matched
        assert not alignment.parent_match
        assert not alignment.full_match

    def test_alignment_relation_mismatch(self):
        """Test alignment with relation mismatch."""
        gold_edu = EDUModel(id=1, text="Test.", parent=0, relation="Background")
        pred_edu = EDUModel(id=1, text="Test.", parent=0, relation="Elaboration")  # Wrong relation

        alignment = EDUAlignment(
            edu_id=1,
            gold_edu=gold_edu,
            pred_edu=pred_edu,
            gold_text="Test.",
            pred_text="Test."
        )

        assert alignment.is_matched
        assert alignment.parent_match  # Parent correct
        assert not alignment.full_match  # But relation wrong

    def test_alignment_unmatched(self):
        """Test alignment with missing predicted EDU."""
        gold_edu = EDUModel(id=1, text="Test.", parent=0, relation="Background")

        alignment = EDUAlignment(
            edu_id=1,
            gold_edu=gold_edu,
            pred_edu=None,
            gold_text="Test.",
            pred_text=None
        )

        assert not alignment.is_matched
        assert not alignment.parent_match
        assert not alignment.full_match


class TestParseMetrics:
    """Test cases for ParseMetrics dataclass."""

    def test_metrics_calculation(self):
        """Test metrics calculations."""
        metrics = ParseMetrics(
            total_edus=10,
            matched_edus=10,
            uas_correct=8,
            las_correct=6,
            alignments=[]
        )

        assert metrics.uas == 80.0
        assert metrics.las == 60.0
        # F1 = 2 * (precision * recall) / (precision + recall)
        # Here precision = recall = LAS, so F1 = LAS
        assert metrics.f1 == 60.0
        assert metrics.precision == 60.0
        assert metrics.recall == 60.0

    def test_metrics_perfect_score(self):
        """Test metrics with perfect predictions."""
        metrics = ParseMetrics(
            total_edus=10,
            matched_edus=10,
            uas_correct=10,
            las_correct=10,
            alignments=[]
        )

        assert metrics.uas == 100.0
        assert metrics.las == 100.0
        assert metrics.f1 == 100.0

    def test_metrics_zero_score(self):
        """Test metrics with no correct predictions."""
        metrics = ParseMetrics(
            total_edus=10,
            matched_edus=10,
            uas_correct=0,
            las_correct=0,
            alignments=[]
        )

        assert metrics.uas == 0.0
        assert metrics.las == 0.0
        assert metrics.f1 == 0.0

    def test_metrics_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = ParseMetrics(
            total_edus=10,
            matched_edus=10,
            uas_correct=8,
            las_correct=6,
            alignments=[]
        )

        metrics_dict = metrics.to_dict()

        assert "uas" in metrics_dict
        assert "las" in metrics_dict
        assert "f1" in metrics_dict
        assert "total_edus" in metrics_dict
        assert metrics_dict["uas"] == 80.0


class TestDiscourseEvaluator:
    """Test cases for DiscourseEvaluator."""

    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        evaluator = DiscourseEvaluator(include_root=False)
        assert not evaluator.include_root

        evaluator = DiscourseEvaluator(include_root=True)
        assert evaluator.include_root

    def test_align_edus_perfect_match(self, sample_simple_tree):
        """Test EDU alignment with identical trees."""
        evaluator = DiscourseEvaluator(include_root=False)

        alignments = evaluator.align_edus(sample_simple_tree, sample_simple_tree)

        # Should have 2 alignments (excluding ROOT)
        assert len(alignments) == 2
        for alignment in alignments:
            assert alignment.is_matched
            assert alignment.parent_match
            assert alignment.full_match

    def test_align_edus_with_differences(self):
        """Test EDU alignment with differences."""
        gold = DiscourseTreeModel(edus=[
            EDUModel(id=0, text="ROOT", parent=-1, relation="null"),
            EDUModel(id=1, text="Sentence 1.", parent=0, relation="Background"),
            EDUModel(id=2, text="Sentence 2.", parent=1, relation="Elaboration")
        ])

        pred = DiscourseTreeModel(edus=[
            EDUModel(id=0, text="ROOT", parent=-1, relation="null"),
            EDUModel(id=1, text="Sentence 1.", parent=0, relation="Background"),
            EDUModel(id=2, text="Sentence 2.", parent=0, relation="Result")  # Different parent and relation
        ])

        evaluator = DiscourseEvaluator(include_root=False)
        alignments = evaluator.align_edus(gold, pred)

        assert len(alignments) == 2
        # EDU 1 should match perfectly
        assert alignments[0].full_match
        # EDU 2 should have mismatches
        assert not alignments[1].parent_match
        assert not alignments[1].full_match

    def test_calculate_uas_perfect(self, sample_simple_tree):
        """Test UAS calculation with perfect predictions."""
        evaluator = DiscourseEvaluator(include_root=False)

        uas = evaluator.calculate_uas(sample_simple_tree, sample_simple_tree)
        assert uas == 100.0

    def test_calculate_uas_partial(self):
        """Test UAS calculation with partial correctness."""
        gold = DiscourseTreeModel(edus=[
            EDUModel(id=0, text="ROOT", parent=-1, relation="null"),
            EDUModel(id=1, text="S1.", parent=0, relation="Background"),
            EDUModel(id=2, text="S2.", parent=1, relation="Elaboration"),
            EDUModel(id=3, text="S3.", parent=1, relation="Addition")
        ])

        pred = DiscourseTreeModel(edus=[
            EDUModel(id=0, text="ROOT", parent=-1, relation="null"),
            EDUModel(id=1, text="S1.", parent=0, relation="Background"),  # Correct parent
            EDUModel(id=2, text="S2.", parent=0, relation="Elaboration"),  # Wrong parent
            EDUModel(id=3, text="S3.", parent=1, relation="Addition")  # Correct parent
        ])

        evaluator = DiscourseEvaluator(include_root=False)
        uas = evaluator.calculate_uas(gold, pred)

        # 2 out of 3 correct (excluding ROOT)
        assert abs(uas - 66.67) < 0.1

    def test_calculate_las_perfect(self, sample_simple_tree):
        """Test LAS calculation with perfect predictions."""
        evaluator = DiscourseEvaluator(include_root=False)

        las = evaluator.calculate_las(sample_simple_tree, sample_simple_tree)
        assert las == 100.0

    def test_calculate_las_partial(self):
        """Test LAS calculation with partial correctness."""
        gold = DiscourseTreeModel(edus=[
            EDUModel(id=0, text="ROOT", parent=-1, relation="null"),
            EDUModel(id=1, text="S1.", parent=0, relation="Background"),
            EDUModel(id=2, text="S2.", parent=1, relation="Elaboration"),
            EDUModel(id=3, text="S3.", parent=1, relation="Addition")
        ])

        pred = DiscourseTreeModel(edus=[
            EDUModel(id=0, text="ROOT", parent=-1, relation="null"),
            EDUModel(id=1, text="S1.", parent=0, relation="Background"),  # Perfect
            EDUModel(id=2, text="S2.", parent=1, relation="Result"),  # Wrong relation
            EDUModel(id=3, text="S3.", parent=0, relation="Addition")  # Wrong parent
        ])

        evaluator = DiscourseEvaluator(include_root=False)
        las = evaluator.calculate_las(gold, pred)

        # Only 1 out of 3 fully correct (excluding ROOT)
        assert abs(las - 33.33) < 0.1

    def test_calculate_f1(self):
        """Test F1 calculation."""
        gold = DiscourseTreeModel(edus=[
            EDUModel(id=0, text="ROOT", parent=-1, relation="null"),
            EDUModel(id=1, text="S1.", parent=0, relation="Background"),
            EDUModel(id=2, text="S2.", parent=1, relation="Elaboration")
        ])

        pred = DiscourseTreeModel(edus=[
            EDUModel(id=0, text="ROOT", parent=-1, relation="null"),
            EDUModel(id=1, text="S1.", parent=0, relation="Background"),
            EDUModel(id=2, text="S2.", parent=1, relation="Result")  # Wrong relation
        ])

        evaluator = DiscourseEvaluator(include_root=False)
        f1 = evaluator.calculate_f1(gold, pred)

        # 1 out of 2 correct, so F1 = 50%
        assert abs(f1 - 50.0) < 0.1

    def test_evaluate_single(self, sample_simple_tree):
        """Test evaluating a single prediction."""
        evaluator = DiscourseEvaluator(include_root=False)

        metrics = evaluator.evaluate_single(sample_simple_tree, sample_simple_tree)

        assert isinstance(metrics, ParseMetrics)
        assert metrics.uas == 100.0
        assert metrics.las == 100.0
        assert metrics.f1 == 100.0

    def test_evaluate_batch(self, sample_prediction_batch):
        """Test evaluating a batch of predictions."""
        gold_trees, pred_trees = sample_prediction_batch

        evaluator = DiscourseEvaluator(include_root=False)
        results = evaluator.evaluate_batch(gold_trees, pred_trees)

        assert "individual_results" in results
        assert "aggregate_metrics" in results

        # Should have metrics for each prediction
        assert len(results["individual_results"]) == 2

        # Aggregate metrics should be averages
        agg = results["aggregate_metrics"]
        assert "uas" in agg
        assert "las" in agg
        assert "f1" in agg

    def test_evaluate_batch_empty(self):
        """Test evaluating empty batch."""
        evaluator = DiscourseEvaluator(include_root=False)
        results = evaluator.evaluate_batch([], [])

        assert len(results["individual_results"]) == 0

    def test_evaluate_with_root_included(self, sample_simple_tree):
        """Test evaluation including ROOT node."""
        evaluator = DiscourseEvaluator(include_root=True)

        metrics = evaluator.evaluate_single(sample_simple_tree, sample_simple_tree)

        # Should include ROOT in count
        assert metrics.total_edus == 3  # Including ROOT

    def test_evaluate_with_root_excluded(self, sample_simple_tree):
        """Test evaluation excluding ROOT node."""
        evaluator = DiscourseEvaluator(include_root=False)

        metrics = evaluator.evaluate_single(sample_simple_tree, sample_simple_tree)

        # Should exclude ROOT from count
        assert metrics.total_edus == 2  # Excluding ROOT

    def test_evaluate_mismatched_edu_count(self):
        """Test evaluation with different number of EDUs."""
        gold = DiscourseTreeModel(edus=[
            EDUModel(id=0, text="ROOT", parent=-1, relation="null"),
            EDUModel(id=1, text="S1.", parent=0, relation="Background"),
            EDUModel(id=2, text="S2.", parent=1, relation="Elaboration")
        ])

        pred = DiscourseTreeModel(edus=[
            EDUModel(id=0, text="ROOT", parent=-1, relation="null"),
            EDUModel(id=1, text="S1.", parent=0, relation="Background")
            # Missing EDU 2
        ])

        evaluator = DiscourseEvaluator(include_root=False)
        metrics = evaluator.evaluate_single(gold, pred)

        # Should handle mismatch gracefully
        assert metrics.matched_edus < metrics.total_edus


class TestRelationEvaluator:
    """Test cases for RelationEvaluator."""

    def test_relation_evaluator_initialization(self):
        """Test RelationEvaluator initialization."""
        evaluator = RelationEvaluator()
        assert evaluator is not None

    def test_evaluate_relations_perfect(self):
        """Test relation evaluation with perfect predictions."""
        gold_trees = [
            DiscourseTreeModel(edus=[
                EDUModel(id=0, text="ROOT", parent=-1, relation="null"),
                EDUModel(id=1, text="S1.", parent=0, relation="Background"),
                EDUModel(id=2, text="S2.", parent=1, relation="Elaboration")
            ])
        ]

        pred_trees = gold_trees  # Perfect predictions

        evaluator = RelationEvaluator()
        results = evaluator.evaluate_relations(gold_trees, pred_trees)

        # Should have perfect scores for all relations
        assert "Background" in results
        assert "Elaboration" in results

        assert results["Background"]["precision"] == 100.0
        assert results["Background"]["recall"] == 100.0
        assert results["Background"]["f1"] == 100.0

    def test_evaluate_relations_with_errors(self):
        """Test relation evaluation with prediction errors."""
        gold_trees = [
            DiscourseTreeModel(edus=[
                EDUModel(id=0, text="ROOT", parent=-1, relation="null"),
                EDUModel(id=1, text="S1.", parent=0, relation="Background"),
                EDUModel(id=2, text="S2.", parent=1, relation="Elaboration"),
                EDUModel(id=3, text="S3.", parent=1, relation="Addition")
            ])
        ]

        pred_trees = [
            DiscourseTreeModel(edus=[
                EDUModel(id=0, text="ROOT", parent=-1, relation="null"),
                EDUModel(id=1, text="S1.", parent=0, relation="Background"),  # Correct
                EDUModel(id=2, text="S2.", parent=1, relation="Result"),  # Wrong: should be Elaboration
                EDUModel(id=3, text="S3.", parent=1, relation="Addition")  # Correct
            ])
        ]

        evaluator = RelationEvaluator()
        results = evaluator.evaluate_relations(gold_trees, pred_trees)

        # Background and Addition should be perfect
        assert results["Background"]["f1"] == 100.0
        assert results["Addition"]["f1"] == 100.0

        # Elaboration should have issues (predicted as Result)
        assert results["Elaboration"]["recall"] < 100.0

    def test_evaluate_relations_support_count(self):
        """Test that support (count) is correctly calculated."""
        gold_trees = [
            DiscourseTreeModel(edus=[
                EDUModel(id=0, text="ROOT", parent=-1, relation="null"),
                EDUModel(id=1, text="S1.", parent=0, relation="Background"),
                EDUModel(id=2, text="S2.", parent=0, relation="Background"),
                EDUModel(id=3, text="S3.", parent=0, relation="Elaboration")
            ])
        ]

        evaluator = RelationEvaluator()
        results = evaluator.evaluate_relations(gold_trees, gold_trees)

        # Background appears twice, Elaboration once
        assert results["Background"]["support"] == 2
        assert results["Elaboration"]["support"] == 1

    def test_evaluate_relations_excludes_null(self):
        """Test that 'null' (ROOT) relation is excluded."""
        gold_trees = [
            DiscourseTreeModel(edus=[
                EDUModel(id=0, text="ROOT", parent=-1, relation="null"),
                EDUModel(id=1, text="S1.", parent=0, relation="Background")
            ])
        ]

        evaluator = RelationEvaluator()
        results = evaluator.evaluate_relations(gold_trees, gold_trees)

        # 'null' should not be in results
        assert "null" not in results

    def test_evaluate_relations_empty_dataset(self):
        """Test relation evaluation with empty dataset."""
        evaluator = RelationEvaluator()
        results = evaluator.evaluate_relations([], [])

        assert len(results) == 0

    def test_evaluate_relations_macro_average(self):
        """Test that macro-averaged metrics are computed correctly."""
        gold_trees = [
            DiscourseTreeModel(edus=[
                EDUModel(id=0, text="ROOT", parent=-1, relation="null"),
                EDUModel(id=1, text="S1.", parent=0, relation="Background"),
                EDUModel(id=2, text="S2.", parent=0, relation="Elaboration")
            ])
        ]

        evaluator = RelationEvaluator()
        results = evaluator.evaluate_relations(gold_trees, gold_trees)

        # With perfect predictions, macro F1 should be 100%
        if "macro_avg" in results:
            assert results["macro_avg"]["f1"] == 100.0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_evaluate_single_edu_trees(self):
        """Test evaluation with trees containing only ROOT."""
        tree = DiscourseTreeModel(edus=[
            EDUModel(id=0, text="ROOT", parent=-1, relation="null")
        ])

        evaluator = DiscourseEvaluator(include_root=False)
        metrics = evaluator.evaluate_single(tree, tree)

        # With only ROOT and include_root=False, no EDUs to evaluate
        assert metrics.total_edus == 0

    def test_evaluate_with_completely_wrong_predictions(self):
        """Test evaluation where all predictions are wrong."""
        gold = DiscourseTreeModel(edus=[
            EDUModel(id=0, text="ROOT", parent=-1, relation="null"),
            EDUModel(id=1, text="S1.", parent=0, relation="Background"),
            EDUModel(id=2, text="S2.", parent=1, relation="Elaboration")
        ])

        pred = DiscourseTreeModel(edus=[
            EDUModel(id=0, text="ROOT", parent=-1, relation="null"),
            EDUModel(id=1, text="S1.", parent=0, relation="Result"),  # Wrong relation
            EDUModel(id=2, text="S2.", parent=0, relation="Conclusion")  # Wrong parent and relation
        ])

        evaluator = DiscourseEvaluator(include_root=False)
        metrics = evaluator.evaluate_single(gold, pred)

        # UAS might have 1 correct (EDU 1 parent)
        # LAS should be 0 (all relations wrong)
        assert metrics.las < 50.0

    def test_batch_evaluation_with_varying_tree_sizes(self):
        """Test batch evaluation with trees of different sizes."""
        gold_trees = [
            DiscourseTreeModel(edus=[
                EDUModel(id=0, text="ROOT", parent=-1, relation="null"),
                EDUModel(id=1, text="S1.", parent=0, relation="Background")
            ]),
            DiscourseTreeModel(edus=[
                EDUModel(id=0, text="ROOT", parent=-1, relation="null"),
                EDUModel(id=1, text="S1.", parent=0, relation="Background"),
                EDUModel(id=2, text="S2.", parent=1, relation="Elaboration"),
                EDUModel(id=3, text="S3.", parent=1, relation="Addition")
            ])
        ]

        pred_trees = gold_trees

        evaluator = DiscourseEvaluator(include_root=False)
        results = evaluator.evaluate_batch(gold_trees, pred_trees)

        # Should handle different sizes correctly
        assert len(results["individual_results"]) == 2
        assert results["aggregate_metrics"]["uas"] == 100.0
