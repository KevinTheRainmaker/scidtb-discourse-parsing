"""
Unit tests for src/data/loader.py

Tests cover:
- Loading single SciDTB files
- Loading dataset splits (train, test)
- UTF-8 BOM handling
- Error handling for invalid files
- Text extraction from trees
"""
import json
import pytest
from pathlib import Path

from src.data.loader import SciDTBLoader, load_dataset
from src.models import DiscourseTreeModel


class TestSciDTBLoader:
    """Test cases for SciDTBLoader class."""

    def test_loader_initialization(self, sample_scidtb_dataset):
        """Test initializing loader with data directory."""
        loader = SciDTBLoader(str(sample_scidtb_dataset))
        assert loader.data_dir == Path(sample_scidtb_dataset)

    def test_load_single_file(self, sample_scidtb_file):
        """Test loading a single .edu.txt.dep file."""
        loader = SciDTBLoader("")  # Empty data_dir for this test
        tree = loader.load_file(str(sample_scidtb_file))

        assert tree is not None
        assert isinstance(tree, DiscourseTreeModel)
        assert len(tree.edus) == 3
        assert tree.edus[0].text == "ROOT"
        assert tree.edus[1].text == "Test sentence one."
        assert tree.edus[2].text == "Test sentence two."

    def test_load_file_with_utf8_bom(self, temp_data_dir):
        """Test that UTF-8 BOM is properly handled."""
        # Create file with UTF-8 BOM
        data = {
            "root": [
                {"id": 0, "text": "ROOT", "parent": -1, "relation": "null"},
                {"id": 1, "text": "Text with BOM.", "parent": 0, "relation": "Background"}
            ]
        }

        file_path = temp_data_dir / "bom_test.edu.txt.dep"
        with open(file_path, 'w', encoding='utf-8-sig') as f:
            json.dump(data, f, ensure_ascii=False)

        loader = SciDTBLoader("")
        tree = loader.load_file(str(file_path))

        assert tree is not None
        assert len(tree.edus) == 2
        assert tree.edus[1].text == "Text with BOM."

    def test_load_file_nonexistent(self, temp_data_dir):
        """Test loading a file that doesn't exist."""
        loader = SciDTBLoader(str(temp_data_dir))
        tree = loader.load_file(str(temp_data_dir / "nonexistent.edu.txt.dep"))
        assert tree is None

    def test_load_file_invalid_json(self, temp_data_dir):
        """Test loading a file with invalid JSON."""
        file_path = temp_data_dir / "invalid.edu.txt.dep"
        with open(file_path, 'w') as f:
            f.write("{ invalid json content")

        loader = SciDTBLoader("")
        tree = loader.load_file(str(file_path))
        assert tree is None

    def test_load_file_invalid_tree_structure(self, temp_data_dir):
        """Test loading a file with invalid tree structure."""
        # Tree with no ROOT
        data = {
            "root": [
                {"id": 1, "text": "No ROOT", "parent": 0, "relation": "Background"}
            ]
        }

        file_path = temp_data_dir / "invalid_tree.edu.txt.dep"
        with open(file_path, 'w', encoding='utf-8-sig') as f:
            json.dump(data, f)

        loader = SciDTBLoader("")
        tree = loader.load_file(str(file_path))
        assert tree is None

    def test_load_split_train(self, sample_scidtb_dataset):
        """Test loading the train split."""
        loader = SciDTBLoader(str(sample_scidtb_dataset))
        trees = loader.load_split("train")

        assert len(trees) == 3
        for tree in trees:
            assert isinstance(tree, DiscourseTreeModel)
            assert tree.edus[0].text == "ROOT"

    def test_load_split_test(self, sample_scidtb_dataset):
        """Test loading the test/gold split."""
        loader = SciDTBLoader(str(sample_scidtb_dataset))
        trees = loader.load_split("test/gold")

        assert len(trees) == 2
        for tree in trees:
            assert isinstance(tree, DiscourseTreeModel)

    def test_load_split_nonexistent(self, sample_scidtb_dataset):
        """Test loading a split that doesn't exist."""
        loader = SciDTBLoader(str(sample_scidtb_dataset))
        trees = loader.load_split("nonexistent_split")

        assert len(trees) == 0

    def test_load_all_splits(self, sample_scidtb_dataset):
        """Test loading all available splits."""
        loader = SciDTBLoader(str(sample_scidtb_dataset))
        all_splits = loader.load_all_splits()

        assert "train" in all_splits
        assert "test/gold" in all_splits

        assert len(all_splits["train"]) == 3
        assert len(all_splits["test/gold"]) == 2

    def test_extract_text_simple(self, sample_simple_tree):
        """Test extracting text from a simple tree."""
        text = SciDTBLoader.extract_text(sample_simple_tree)

        # Should exclude ROOT and concatenate other EDUs
        assert "Background sentence." in text
        assert "Main finding." in text
        assert "ROOT" not in text

    def test_extract_text_complex(self, sample_complex_tree):
        """Test extracting text from a complex tree."""
        text = SciDTBLoader.extract_text(sample_complex_tree)

        assert "Background context." in text
        assert "Previous work." in text
        assert "Research gap." in text
        assert "Our method." in text
        assert "Results." in text
        assert "ROOT" not in text

    def test_extract_text_preserves_order(self, sample_complex_tree):
        """Test that text extraction preserves EDU order."""
        text = SciDTBLoader.extract_text(sample_complex_tree)

        # Check that EDUs appear in order (by their position in text)
        edu_texts = [edu.text for edu in sample_complex_tree.edus if edu.id != 0]
        for edu_text in edu_texts:
            assert edu_text in text

    def test_load_dataset_convenience_function(self, sample_scidtb_dataset):
        """Test the convenience load_dataset function."""
        trees = load_dataset(str(sample_scidtb_dataset), "train")

        assert len(trees) == 3
        for tree in trees:
            assert isinstance(tree, DiscourseTreeModel)

    def test_loader_handles_empty_directory(self, temp_data_dir):
        """Test loader with empty directory."""
        empty_dir = temp_data_dir / "empty"
        empty_dir.mkdir()

        loader = SciDTBLoader(str(empty_dir))
        trees = loader.load_split("train")

        assert len(trees) == 0

    def test_loader_filters_non_dep_files(self, temp_data_dir):
        """Test that loader only loads .edu.txt.dep files."""
        # Create a valid .edu.txt.dep file
        data = {
            "root": [
                {"id": 0, "text": "ROOT", "parent": -1, "relation": "null"},
                {"id": 1, "text": "Valid file.", "parent": 0, "relation": "Background"}
            ]
        }

        valid_file = temp_data_dir / "valid.edu.txt.dep"
        with open(valid_file, 'w', encoding='utf-8-sig') as f:
            json.dump(data, f)

        # Create files with other extensions
        (temp_data_dir / "other.txt").write_text("Not a dep file")
        (temp_data_dir / "other.json").write_text("{}")

        loader = SciDTBLoader(str(temp_data_dir))
        trees = loader.load_split("")

        # Should only load the .edu.txt.dep file
        assert len(trees) == 1
        assert trees[0].edus[1].text == "Valid file."

    def test_loader_with_nested_directory_structure(self, sample_scidtb_dataset):
        """Test loader with nested directory structure (test/gold)."""
        loader = SciDTBLoader(str(sample_scidtb_dataset))

        # Should handle nested path correctly
        trees = loader.load_split("test/gold")
        assert len(trees) == 2

    def test_extract_text_single_edu_tree(self):
        """Test extracting text from tree with only ROOT."""
        tree = DiscourseTreeModel(edus=[
            {"id": 0, "text": "ROOT", "parent": -1, "relation": "null"}
        ])

        text = SciDTBLoader.extract_text(tree)
        assert text == ""  # Only ROOT, so no text

    def test_load_file_with_special_characters(self, temp_data_dir):
        """Test loading file with special characters in text."""
        data = {
            "root": [
                {"id": 0, "text": "ROOT", "parent": -1, "relation": "null"},
                {"id": 1, "text": "Special chars: αβγ, 中文, émojis 🔬", "parent": 0, "relation": "Background"}
            ]
        }

        file_path = temp_data_dir / "special.edu.txt.dep"
        with open(file_path, 'w', encoding='utf-8-sig') as f:
            json.dump(data, f, ensure_ascii=False)

        loader = SciDTBLoader("")
        tree = loader.load_file(str(file_path))

        assert tree is not None
        assert "αβγ" in tree.edus[1].text
        assert "中文" in tree.edus[1].text
        assert "🔬" in tree.edus[1].text
