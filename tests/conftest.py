"""
Pytest configuration and shared fixtures for testing.
"""
import json
import tempfile
from pathlib import Path
from typing import Dict, List

import pytest

from src.models import EDUModel, DiscourseTreeModel


@pytest.fixture
def sample_edu_root():
    """Create a sample ROOT EDU."""
    return EDUModel(
        id=0,
        text="ROOT",
        parent=-1,
        relation="null"
    )


@pytest.fixture
def sample_edu_background():
    """Create a sample Background EDU."""
    return EDUModel(
        id=1,
        text="This is the background context.",
        parent=0,
        relation="Background"
    )


@pytest.fixture
def sample_edu_elaboration():
    """Create a sample Elaboration EDU."""
    return EDUModel(
        id=2,
        text="This provides more details.",
        parent=1,
        relation="Elaboration"
    )


@pytest.fixture
def sample_simple_tree():
    """Create a simple discourse tree with 3 EDUs."""
    return DiscourseTreeModel(edus=[
        EDUModel(id=0, text="ROOT", parent=-1, relation="null"),
        EDUModel(id=1, text="Background sentence.", parent=0, relation="Background"),
        EDUModel(id=2, text="Main finding.", parent=1, relation="Elaboration")
    ])


@pytest.fixture
def sample_complex_tree():
    """Create a more complex discourse tree with 6 EDUs."""
    return DiscourseTreeModel(edus=[
        EDUModel(id=0, text="ROOT", parent=-1, relation="null"),
        EDUModel(id=1, text="Background context.", parent=0, relation="Background"),
        EDUModel(id=2, text="Previous work.", parent=1, relation="Addition"),
        EDUModel(id=3, text="Research gap.", parent=1, relation="Contrast"),
        EDUModel(id=4, text="Our method.", parent=0, relation="Enablement"),
        EDUModel(id=5, text="Results.", parent=4, relation="Result")
    ])


@pytest.fixture
def sample_scidtb_format():
    """Sample data in SciDTB format (with 'root' key)."""
    return {
        "root": [
            {"id": 0, "text": "ROOT", "parent": -1, "relation": "null"},
            {"id": 1, "text": "Background.", parent=0, "relation": "Background"},
            {"id": 2, "text": "Method.", parent=0, "relation": "Enablement"}
        ]
    }


@pytest.fixture
def sample_standard_format():
    """Sample data in standard format (with 'edus' key)."""
    return {
        "edus": [
            {"id": 0, "text": "ROOT", "parent": -1, "relation": "null"},
            {"id": 1, "text": "Background.", "parent": 0, "relation": "Background"},
            {"id": 2, "text": "Method.", "parent": 0, "relation": "Enablement"}
        ]
    }


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_scidtb_file(temp_data_dir):
    """Create a sample .edu.txt.dep file in SciDTB format."""
    data = {
        "root": [
            {"id": 0, "text": "ROOT", "parent": -1, "relation": "null"},
            {"id": 1, "text": "Test sentence one.", "parent": 0, "relation": "Background"},
            {"id": 2, "text": "Test sentence two.", "parent": 1, "relation": "Elaboration"}
        ]
    }

    file_path = temp_data_dir / "test.edu.txt.dep"
    with open(file_path, 'w', encoding='utf-8-sig') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return file_path


@pytest.fixture
def sample_scidtb_dataset(temp_data_dir):
    """Create a sample SciDTB dataset structure with train and test splits."""
    # Create train directory
    train_dir = temp_data_dir / "train"
    train_dir.mkdir()

    # Create test/gold directory
    test_dir = temp_data_dir / "test" / "gold"
    test_dir.mkdir(parents=True)

    # Create train files
    for i in range(3):
        data = {
            "root": [
                {"id": 0, "text": "ROOT", "parent": -1, "relation": "null"},
                {"id": 1, "text": f"Train sentence {i} EDU 1.", "parent": 0, "relation": "Background"},
                {"id": 2, "text": f"Train sentence {i} EDU 2.", "parent": 1, "relation": "Elaboration"}
            ]
        }
        file_path = train_dir / f"train_{i}.edu.txt.dep"
        with open(file_path, 'w', encoding='utf-8-sig') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # Create test files
    for i in range(2):
        data = {
            "root": [
                {"id": 0, "text": "ROOT", "parent": -1, "relation": "null"},
                {"id": 1, "text": f"Test sentence {i} EDU 1.", "parent": 0, "relation": "Background"},
                {"id": 2, "text": f"Test sentence {i} EDU 2.", "parent": 1, "relation": "Result"}
            ]
        }
        file_path = test_dir / f"test_{i}.edu.txt.dep"
        with open(file_path, 'w', encoding='utf-8-sig') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    return temp_data_dir


@pytest.fixture
def sample_finetuning_jsonl(temp_data_dir):
    """Create a sample JSONL file for fine-tuning."""
    data = [
        {
            "messages": [
                {"role": "system", "content": "Parse discourse structure."},
                {"role": "user", "content": "Background. Method."},
                {
                    "role": "assistant",
                    "content": json.dumps({
                        "edus": [
                            {"id": 0, "text": "ROOT", "parent": -1, "relation": "null"},
                            {"id": 1, "text": "Background.", "parent": 0, "relation": "Background"},
                            {"id": 2, "text": "Method.", "parent": 0, "relation": "Enablement"}
                        ]
                    })
                }
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "Parse discourse structure."},
                {"role": "user", "content": "Result. Conclusion."},
                {
                    "role": "assistant",
                    "content": json.dumps({
                        "edus": [
                            {"id": 0, "text": "ROOT", "parent": -1, "relation": "null"},
                            {"id": 1, "text": "Result.", "parent": 0, "relation": "Result"},
                            {"id": 2, "text": "Conclusion.", "parent": 1, "relation": "Conclusion"}
                        ]
                    })
                }
            ]
        }
    ]

    file_path = temp_data_dir / "training.jsonl"
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

    return file_path


@pytest.fixture
def mock_openai_response():
    """Mock response from OpenAI API."""
    return {
        "edus": [
            {"id": 0, "text": "ROOT", "parent": -1, "relation": "null"},
            {"id": 1, "text": "First sentence.", "parent": 0, "relation": "Background"},
            {"id": 2, "text": "Second sentence.", "parent": 1, "relation": "Elaboration"}
        ]
    }


@pytest.fixture
def invalid_tree_no_root():
    """Invalid tree data with no ROOT node."""
    return {
        "edus": [
            {"id": 1, "text": "First sentence.", "parent": 0, "relation": "Background"},
            {"id": 2, "text": "Second sentence.", "parent": 1, "relation": "Elaboration"}
        ]
    }


@pytest.fixture
def invalid_tree_multiple_roots():
    """Invalid tree data with multiple ROOT nodes."""
    return {
        "edus": [
            {"id": 0, "text": "ROOT", "parent": -1, "relation": "null"},
            {"id": 1, "text": "Another ROOT", "parent": -1, "relation": "null"},
            {"id": 2, "text": "First sentence.", "parent": 0, "relation": "Background"}
        ]
    }


@pytest.fixture
def invalid_tree_cycle():
    """Invalid tree data with a cycle."""
    return {
        "edus": [
            {"id": 0, "text": "ROOT", "parent": -1, "relation": "null"},
            {"id": 1, "text": "First sentence.", "parent": 2, "relation": "Background"},
            {"id": 2, "text": "Second sentence.", "parent": 1, "relation": "Elaboration"}
        ]
    }


@pytest.fixture
def invalid_tree_forward_reference():
    """Invalid tree data with forward reference (parent >= id)."""
    return {
        "edus": [
            {"id": 0, "text": "ROOT", "parent": -1, "relation": "null"},
            {"id": 1, "text": "First sentence.", "parent": 2, "relation": "Background"},
            {"id": 2, "text": "Second sentence.", "parent": 0, "relation": "Elaboration"}
        ]
    }


@pytest.fixture
def invalid_tree_non_consecutive_ids():
    """Invalid tree data with non-consecutive IDs."""
    return {
        "edus": [
            {"id": 0, "text": "ROOT", "parent": -1, "relation": "null"},
            {"id": 1, "text": "First sentence.", "parent": 0, "relation": "Background"},
            {"id": 3, "text": "Second sentence.", "parent": 1, "relation": "Elaboration"}
        ]
    }


@pytest.fixture
def sample_prediction_batch():
    """Create a batch of gold and predicted trees for evaluation."""
    gold_trees = [
        DiscourseTreeModel(edus=[
            EDUModel(id=0, text="ROOT", parent=-1, relation="null"),
            EDUModel(id=1, text="Background.", parent=0, relation="Background"),
            EDUModel(id=2, text="Method.", parent=1, relation="Elaboration")
        ]),
        DiscourseTreeModel(edus=[
            EDUModel(id=0, text="ROOT", parent=-1, relation="null"),
            EDUModel(id=1, text="Context.", parent=0, relation="Background"),
            EDUModel(id=2, text="Result.", parent=0, relation="Result")
        ])
    ]

    pred_trees = [
        DiscourseTreeModel(edus=[
            EDUModel(id=0, text="ROOT", parent=-1, relation="null"),
            EDUModel(id=1, text="Background.", parent=0, relation="Background"),
            EDUModel(id=2, text="Method.", parent=0, relation="Elaboration")  # Wrong parent
        ]),
        DiscourseTreeModel(edus=[
            EDUModel(id=0, text="ROOT", parent=-1, relation="null"),
            EDUModel(id=1, text="Context.", parent=0, relation="Background"),
            EDUModel(id=2, text="Result.", parent=0, relation="Conclusion")  # Wrong relation
        ])
    ]

    return gold_trees, pred_trees


@pytest.fixture
def api_key():
    """Provide a test API key."""
    return "sk-test-key-12345"
