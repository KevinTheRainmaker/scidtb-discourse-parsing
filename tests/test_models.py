"""
Unit tests for src/models/edu.py and src/models/tree.py

Tests cover:
- EDUModel validation
- DiscourseTreeModel validation
- Tree navigation methods
- Serialization and deserialization
"""
import pytest
from pydantic import ValidationError

from src.models import EDUModel, DiscourseTreeModel


class TestEDUModel:
    """Test cases for EDUModel validation and behavior."""

    def test_valid_root_edu(self, sample_edu_root):
        """Test creating a valid ROOT EDU."""
        assert sample_edu_root.id == 0
        assert sample_edu_root.text == "ROOT"
        assert sample_edu_root.parent == -1
        assert sample_edu_root.relation == "null"

    def test_valid_non_root_edu(self, sample_edu_background):
        """Test creating a valid non-ROOT EDU."""
        assert sample_edu_background.id == 1
        assert sample_edu_background.text == "This is the background context."
        assert sample_edu_background.parent == 0
        assert sample_edu_background.relation == "Background"

    def test_edu_with_valid_relations(self):
        """Test EDU creation with various valid relation types."""
        valid_relations = [
            "Background", "Elaboration", "Addition", "Cause-effect",
            "Result", "Enablement", "Contrast", "Comparison",
            "Temporal", "Condition", "Evaluation", "Conclusion"
        ]

        for relation in valid_relations:
            edu = EDUModel(
                id=1,
                text="Test sentence.",
                parent=0,
                relation=relation
            )
            assert edu.relation == relation

    def test_edu_root_must_have_parent_minus_one(self):
        """Test that ROOT EDU must have parent=-1."""
        with pytest.raises(ValidationError, match="ROOT EDU must have parent=-1"):
            EDUModel(
                id=0,
                text="ROOT",
                parent=0,  # Invalid
                relation="null"
            )

    def test_edu_non_root_cannot_have_parent_minus_one(self):
        """Test that non-ROOT EDU cannot have parent=-1."""
        with pytest.raises(ValidationError, match="Only ROOT EDU can have parent=-1"):
            EDUModel(
                id=1,
                text="Not ROOT",
                parent=-1,  # Invalid
                relation="Background"
            )

    def test_edu_cannot_reference_itself(self):
        """Test that EDU cannot have itself as parent."""
        with pytest.raises(ValidationError, match="EDU cannot reference itself as parent"):
            EDUModel(
                id=1,
                text="Self-referencing",
                parent=1,  # Invalid
                relation="Background"
            )

    def test_edu_invalid_relation_type(self):
        """Test that invalid relation types are rejected."""
        with pytest.raises(ValidationError):
            EDUModel(
                id=1,
                text="Test sentence.",
                parent=0,
                relation="InvalidRelation"  # Not in valid relation list
            )

    def test_edu_parent_must_be_less_than_id(self):
        """Test that parent must have smaller ID than child (no forward references)."""
        with pytest.raises(ValidationError, match="Parent must have a smaller ID"):
            EDUModel(
                id=1,
                text="Forward reference",
                parent=2,  # Invalid: parent id > current id
                relation="Background"
            )

    def test_edu_dict_conversion(self, sample_edu_background):
        """Test converting EDU to dictionary."""
        edu_dict = sample_edu_background.model_dump()
        assert edu_dict["id"] == 1
        assert edu_dict["text"] == "This is the background context."
        assert edu_dict["parent"] == 0
        assert edu_dict["relation"] == "Background"


class TestDiscourseTreeModel:
    """Test cases for DiscourseTreeModel validation and behavior."""

    def test_valid_simple_tree(self, sample_simple_tree):
        """Test creating a valid simple tree."""
        assert len(sample_simple_tree.edus) == 3
        assert sample_simple_tree.edus[0].id == 0
        assert sample_simple_tree.edus[0].text == "ROOT"

    def test_valid_complex_tree(self, sample_complex_tree):
        """Test creating a valid complex tree."""
        assert len(sample_complex_tree.edus) == 6
        assert sample_complex_tree.edus[0].id == 0

    def test_tree_must_have_exactly_one_root(self, invalid_tree_no_root, invalid_tree_multiple_roots):
        """Test that tree must have exactly one ROOT node."""
        # No ROOT
        with pytest.raises(ValidationError, match="Tree must have exactly one ROOT node"):
            DiscourseTreeModel.from_dict(invalid_tree_no_root)

        # Multiple ROOTs
        with pytest.raises(ValidationError, match="Tree must have exactly one ROOT node"):
            DiscourseTreeModel.from_dict(invalid_tree_multiple_roots)

    def test_tree_must_have_consecutive_ids(self, invalid_tree_non_consecutive_ids):
        """Test that tree must have consecutive EDU IDs."""
        with pytest.raises(ValidationError, match="EDU IDs must be consecutive"):
            DiscourseTreeModel.from_dict(invalid_tree_non_consecutive_ids)

    def test_tree_cannot_have_cycles(self, invalid_tree_cycle):
        """Test that tree cannot contain cycles."""
        with pytest.raises(ValidationError, match="Tree contains cycle"):
            DiscourseTreeModel.from_dict(invalid_tree_cycle)

    def test_tree_cannot_have_forward_references(self, invalid_tree_forward_reference):
        """Test that tree cannot have forward references."""
        with pytest.raises(ValidationError, match="Parent must have a smaller ID"):
            DiscourseTreeModel.from_dict(invalid_tree_forward_reference)

    def test_tree_to_dict(self, sample_simple_tree):
        """Test converting tree to dictionary."""
        tree_dict = sample_simple_tree.to_dict()
        assert "edus" in tree_dict
        assert len(tree_dict["edus"]) == 3
        assert tree_dict["edus"][0]["id"] == 0
        assert tree_dict["edus"][0]["text"] == "ROOT"

    def test_tree_from_dict_standard_format(self, sample_standard_format):
        """Test loading tree from standard format (with 'edus' key)."""
        tree = DiscourseTreeModel.from_dict(sample_standard_format)
        assert len(tree.edus) == 3
        assert tree.edus[0].text == "ROOT"
        assert tree.edus[1].relation == "Background"

    def test_tree_from_dict_scidtb_format(self, sample_scidtb_format):
        """Test loading tree from SciDTB format (with 'root' key)."""
        tree = DiscourseTreeModel.from_dict(sample_scidtb_format)
        assert len(tree.edus) == 3
        assert tree.edus[0].text == "ROOT"
        assert tree.edus[2].relation == "Enablement"

    def test_tree_from_scidtb(self, sample_scidtb_format):
        """Test loading tree using from_scidtb method."""
        tree = DiscourseTreeModel.from_scidtb(sample_scidtb_format)
        assert len(tree.edus) == 3
        assert tree.edus[0].text == "ROOT"

    def test_get_edu_by_id(self, sample_complex_tree):
        """Test retrieving EDU by ID."""
        edu = sample_complex_tree.get_edu_by_id(3)
        assert edu is not None
        assert edu.id == 3
        assert edu.text == "Research gap."

        # Non-existent ID
        edu = sample_complex_tree.get_edu_by_id(99)
        assert edu is None

    def test_get_children(self, sample_complex_tree):
        """Test getting children of an EDU."""
        # ROOT (id=0) has children [1, 4]
        children = sample_complex_tree.get_children(0)
        assert len(children) == 2
        assert children[0].id == 1
        assert children[1].id == 4

        # EDU 1 has children [2, 3]
        children = sample_complex_tree.get_children(1)
        assert len(children) == 2
        assert children[0].id == 2
        assert children[1].id == 3

        # EDU 5 has no children
        children = sample_complex_tree.get_children(5)
        assert len(children) == 0

    def test_get_depth(self, sample_complex_tree):
        """Test calculating EDU depth in tree."""
        # ROOT has depth 0
        assert sample_complex_tree.get_depth(0) == 0

        # Direct children of ROOT have depth 1
        assert sample_complex_tree.get_depth(1) == 1
        assert sample_complex_tree.get_depth(4) == 1

        # Grandchildren have depth 2
        assert sample_complex_tree.get_depth(2) == 2
        assert sample_complex_tree.get_depth(3) == 2
        assert sample_complex_tree.get_depth(5) == 2

    def test_get_statistics(self, sample_complex_tree):
        """Test getting tree statistics."""
        stats = sample_complex_tree.get_statistics()

        assert stats["num_edus"] == 6
        assert stats["max_depth"] == 2
        assert "avg_depth" in stats
        assert "relation_distribution" in stats

        # Check relation distribution
        rel_dist = stats["relation_distribution"]
        assert rel_dist["Background"] == 1
        assert rel_dist["Addition"] == 1
        assert rel_dist["Contrast"] == 1
        assert rel_dist["Enablement"] == 1
        assert rel_dist["Result"] == 1

    def test_tree_with_single_root_only(self):
        """Test tree with only ROOT node."""
        tree = DiscourseTreeModel(edus=[
            EDUModel(id=0, text="ROOT", parent=-1, relation="null")
        ])
        assert len(tree.edus) == 1
        assert tree.get_depth(0) == 0
        assert len(tree.get_children(0)) == 0

    def test_tree_parent_references_are_valid(self, sample_complex_tree):
        """Test that all parent references are valid."""
        for edu in sample_complex_tree.edus:
            if edu.id == 0:
                # ROOT should have parent=-1
                assert edu.parent == -1
            else:
                # Non-ROOT should have valid parent
                parent = sample_complex_tree.get_edu_by_id(edu.parent)
                assert parent is not None
                assert parent.id < edu.id

    def test_tree_serialization_roundtrip(self, sample_complex_tree):
        """Test that tree can be serialized and deserialized without loss."""
        # Convert to dict
        tree_dict = sample_complex_tree.to_dict()

        # Load from dict
        tree_loaded = DiscourseTreeModel.from_dict(tree_dict)

        # Compare
        assert len(tree_loaded.edus) == len(sample_complex_tree.edus)
        for original, loaded in zip(sample_complex_tree.edus, tree_loaded.edus):
            assert original.id == loaded.id
            assert original.text == loaded.text
            assert original.parent == loaded.parent
            assert original.relation == loaded.relation

    def test_tree_empty_edus_list(self):
        """Test that tree cannot be created with empty EDU list."""
        with pytest.raises(ValidationError):
            DiscourseTreeModel(edus=[])

    def test_tree_invalid_parent_reference(self):
        """Test that tree rejects invalid parent references."""
        with pytest.raises(ValidationError, match="EDU .* references non-existent parent"):
            DiscourseTreeModel(edus=[
                EDUModel(id=0, text="ROOT", parent=-1, relation="null"),
                EDUModel(id=1, text="Text", parent=5, relation="Background")  # Parent 5 doesn't exist
            ])
