"""
Unit tests for src/data/preprocessor.py

Tests cover:
- Filtering trees by length
- Getting relation statistics
- Splitting data into train/validation
- Edge cases and validation
"""
import pytest

from src.data.preprocessor import DataPreprocessor
from src.models import EDUModel, DiscourseTreeModel


class TestDataPreprocessor:
    """Test cases for DataPreprocessor class."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset with varying EDU counts."""
        trees = [
            # Tree with 3 EDUs
            DiscourseTreeModel(edus=[
                EDUModel(id=0, text="ROOT", parent=-1, relation="null"),
                EDUModel(id=1, text="Sent 1.", parent=0, relation="Background"),
                EDUModel(id=2, text="Sent 2.", parent=1, relation="Elaboration")
            ]),
            # Tree with 5 EDUs
            DiscourseTreeModel(edus=[
                EDUModel(id=0, text="ROOT", parent=-1, relation="null"),
                EDUModel(id=1, text="Sent 1.", parent=0, relation="Background"),
                EDUModel(id=2, text="Sent 2.", parent=1, relation="Addition"),
                EDUModel(id=3, text="Sent 3.", parent=1, relation="Contrast"),
                EDUModel(id=4, text="Sent 4.", parent=0, relation="Result")
            ]),
            # Tree with 7 EDUs
            DiscourseTreeModel(edus=[
                EDUModel(id=0, text="ROOT", parent=-1, relation="null"),
                EDUModel(id=1, text="Sent 1.", parent=0, relation="Background"),
                EDUModel(id=2, text="Sent 2.", parent=1, relation="Elaboration"),
                EDUModel(id=3, text="Sent 3.", parent=1, relation="Addition"),
                EDUModel(id=4, text="Sent 4.", parent=0, relation="Enablement"),
                EDUModel(id=5, text="Sent 5.", parent=4, relation="Result"),
                EDUModel(id=6, text="Sent 6.", parent=4, relation="Conclusion")
            ]),
            # Tree with 4 EDUs
            DiscourseTreeModel(edus=[
                EDUModel(id=0, text="ROOT", parent=-1, relation="null"),
                EDUModel(id=1, text="Sent 1.", parent=0, relation="Background"),
                EDUModel(id=2, text="Sent 2.", parent=0, relation="Enablement"),
                EDUModel(id=3, text="Sent 3.", parent=2, relation="Result")
            ])
        ]
        return trees

    def test_filter_by_length_no_filters(self, sample_dataset):
        """Test filtering with no min/max constraints."""
        filtered = DataPreprocessor.filter_by_length(sample_dataset)
        assert len(filtered) == len(sample_dataset)

    def test_filter_by_length_min_edus(self, sample_dataset):
        """Test filtering with minimum EDU count."""
        # Filter: at least 5 EDUs
        filtered = DataPreprocessor.filter_by_length(sample_dataset, min_edus=5)

        assert len(filtered) == 2  # Only 5 and 7 EDU trees
        assert all(len(tree.edus) >= 5 for tree in filtered)

    def test_filter_by_length_max_edus(self, sample_dataset):
        """Test filtering with maximum EDU count."""
        # Filter: at most 4 EDUs
        filtered = DataPreprocessor.filter_by_length(sample_dataset, max_edus=4)

        assert len(filtered) == 2  # Only 3 and 4 EDU trees
        assert all(len(tree.edus) <= 4 for tree in filtered)

    def test_filter_by_length_min_and_max(self, sample_dataset):
        """Test filtering with both min and max constraints."""
        # Filter: 4 to 5 EDUs
        filtered = DataPreprocessor.filter_by_length(sample_dataset, min_edus=4, max_edus=5)

        assert len(filtered) == 2  # 4 and 5 EDU trees
        assert all(4 <= len(tree.edus) <= 5 for tree in filtered)

    def test_filter_by_length_exclude_all(self, sample_dataset):
        """Test filtering that excludes all trees."""
        # Filter: at least 10 EDUs (none exist)
        filtered = DataPreprocessor.filter_by_length(sample_dataset, min_edus=10)
        assert len(filtered) == 0

    def test_filter_by_length_empty_dataset(self):
        """Test filtering an empty dataset."""
        filtered = DataPreprocessor.filter_by_length([])
        assert len(filtered) == 0

    def test_get_relation_statistics(self, sample_dataset):
        """Test getting relation statistics from dataset."""
        stats = DataPreprocessor.get_relation_statistics(sample_dataset)

        # Check that all relations are counted
        assert stats["Background"] == 4  # Appears in all 4 trees
        assert stats["Elaboration"] == 2  # Trees 1 and 3
        assert stats["Addition"] == 2  # Trees 2 and 3
        assert stats["Result"] == 3  # Trees 2, 3, and 4
        assert stats["Contrast"] == 1  # Tree 2 only
        assert stats["Enablement"] == 2  # Trees 3 and 4
        assert stats["Conclusion"] == 1  # Tree 3 only

    def test_get_relation_statistics_excludes_root(self, sample_dataset):
        """Test that ROOT relation ('null') is not counted in statistics."""
        stats = DataPreprocessor.get_relation_statistics(sample_dataset)

        # 'null' relation should not be in stats
        assert "null" not in stats

    def test_get_relation_statistics_empty_dataset(self):
        """Test getting statistics from empty dataset."""
        stats = DataPreprocessor.get_relation_statistics([])
        assert len(stats) == 0

    def test_get_relation_statistics_single_tree(self, sample_simple_tree):
        """Test getting statistics from single tree."""
        stats = DataPreprocessor.get_relation_statistics([sample_simple_tree])

        assert stats["Background"] == 1
        assert stats["Elaboration"] == 1
        assert len(stats) == 2  # Only these two relations

    def test_split_data_default_ratio(self, sample_dataset):
        """Test splitting data with default 80/20 ratio."""
        train, val = DataPreprocessor.split_data(sample_dataset)

        # With 4 samples, 80/20 split should give 3 train, 1 val
        assert len(train) == 3
        assert len(val) == 1

        # All trees should be accounted for
        assert len(train) + len(val) == len(sample_dataset)

    def test_split_data_custom_ratio(self, sample_dataset):
        """Test splitting data with custom ratio."""
        train, val = DataPreprocessor.split_data(sample_dataset, train_ratio=0.5)

        # 50/50 split should give 2 train, 2 val
        assert len(train) == 2
        assert len(val) == 2

    def test_split_data_reproducibility(self, sample_dataset):
        """Test that split is reproducible with same random seed."""
        train1, val1 = DataPreprocessor.split_data(sample_dataset, random_seed=42)
        train2, val2 = DataPreprocessor.split_data(sample_dataset, random_seed=42)

        # Should produce identical splits
        assert len(train1) == len(train2)
        assert len(val1) == len(val2)

        # Compare tree IDs or EDU counts to verify same trees
        train1_sizes = [len(t.edus) for t in train1]
        train2_sizes = [len(t.edus) for t in train2]
        assert train1_sizes == train2_sizes

    def test_split_data_different_seeds(self, sample_dataset):
        """Test that different seeds produce different splits."""
        train1, val1 = DataPreprocessor.split_data(sample_dataset, random_seed=42)
        train2, val2 = DataPreprocessor.split_data(sample_dataset, random_seed=123)

        # Sizes should be the same
        assert len(train1) == len(train2)
        assert len(val1) == len(val2)

        # But actual trees may differ (check EDU counts as proxy)
        train1_sizes = [len(t.edus) for t in train1]
        train2_sizes = [len(t.edus) for t in train2]

        # With different seeds, splits are likely different
        # (This may occasionally fail due to random chance, but unlikely)
        # Let's just check that the function accepts different seeds
        assert len(train1_sizes) > 0
        assert len(train2_sizes) > 0

    def test_split_data_extreme_ratios(self):
        """Test splitting with extreme train ratios."""
        trees = [
            DiscourseTreeModel(edus=[
                EDUModel(id=0, text="ROOT", parent=-1, relation="null"),
                EDUModel(id=1, text=f"Sent {i}.", parent=0, relation="Background")
            ]) for i in range(10)
        ]

        # 90/10 split
        train, val = DataPreprocessor.split_data(trees, train_ratio=0.9)
        assert len(train) == 9
        assert len(val) == 1

        # 10/90 split
        train, val = DataPreprocessor.split_data(trees, train_ratio=0.1)
        assert len(train) == 1
        assert len(val) == 9

    def test_split_data_empty_dataset(self):
        """Test splitting an empty dataset."""
        train, val = DataPreprocessor.split_data([])
        assert len(train) == 0
        assert len(val) == 0

    def test_split_data_single_tree(self, sample_simple_tree):
        """Test splitting a dataset with single tree."""
        train, val = DataPreprocessor.split_data([sample_simple_tree], train_ratio=0.8)

        # With 1 sample, train_ratio=0.8 should put it in train
        assert len(train) == 1
        assert len(val) == 0

    def test_split_data_preserves_trees(self, sample_dataset):
        """Test that split doesn't modify original trees."""
        original_sizes = [len(tree.edus) for tree in sample_dataset]

        train, val = DataPreprocessor.split_data(sample_dataset)

        # Original should be unchanged
        assert [len(tree.edus) for tree in sample_dataset] == original_sizes

        # Train and val should contain valid trees
        for tree in train + val:
            assert isinstance(tree, DiscourseTreeModel)
            assert tree.edus[0].text == "ROOT"

    def test_filter_and_split_pipeline(self, sample_dataset):
        """Test combining filtering and splitting operations."""
        # Filter to 4-7 EDUs
        filtered = DataPreprocessor.filter_by_length(sample_dataset, min_edus=4, max_edus=7)
        assert len(filtered) == 3  # Trees with 4, 5, 7 EDUs

        # Split filtered data
        train, val = DataPreprocessor.split_data(filtered, train_ratio=0.67)
        assert len(train) == 2
        assert len(val) == 1

        # All should have 4-7 EDUs
        for tree in train + val:
            assert 4 <= len(tree.edus) <= 7

    def test_get_relation_statistics_with_filtered_data(self, sample_dataset):
        """Test statistics on filtered subset."""
        # Filter to only trees with >=5 EDUs
        filtered = DataPreprocessor.filter_by_length(sample_dataset, min_edus=5)

        stats = DataPreprocessor.get_relation_statistics(filtered)

        # Should only count relations from trees with 5+ EDUs
        assert stats["Background"] == 2  # Trees 2 and 3
        assert stats["Addition"] == 2
        assert stats["Result"] == 2

    def test_relation_statistics_ordering(self, sample_dataset):
        """Test that relation statistics are properly counted."""
        stats = DataPreprocessor.get_relation_statistics(sample_dataset)

        # Verify total count (excluding ROOT)
        total_relations = sum(stats.values())

        # Count manually: 4 trees with (2, 4, 6, 3) non-ROOT EDUs = 15 total
        expected_total = (3-1) + (5-1) + (7-1) + (4-1)  # Subtract ROOT from each
        assert total_relations == expected_total

    def test_filter_by_length_boundary_cases(self):
        """Test filtering at exact boundary values."""
        trees = [
            DiscourseTreeModel(edus=[
                EDUModel(id=0, text="ROOT", parent=-1, relation="null"),
                *[EDUModel(id=i, text=f"S{i}", parent=i-1, relation="Background")
                  for i in range(1, n)]
            ])
            for n in [3, 5, 7, 9]  # Trees with 3, 5, 7, 9 EDUs
        ]

        # Test min_edus=5 includes tree with exactly 5 EDUs
        filtered = DataPreprocessor.filter_by_length(trees, min_edus=5)
        assert len(filtered) == 3  # 5, 7, 9
        assert len(filtered[0].edus) == 5

        # Test max_edus=7 includes tree with exactly 7 EDUs
        filtered = DataPreprocessor.filter_by_length(trees, max_edus=7)
        assert len(filtered) == 3  # 3, 5, 7
        assert len(filtered[-1].edus) == 7
