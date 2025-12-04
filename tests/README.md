# Test Suite Documentation

This directory contains comprehensive unit tests for the SciDTB discourse parsing project following Test-Driven Development (TDD) principles.

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and test configuration
├── test_models.py           # Tests for EDUModel and DiscourseTreeModel
├── test_data_loader.py      # Tests for SciDTBLoader
├── test_data_preprocessor.py # Tests for DataPreprocessor
├── test_parsers.py          # Tests for all parser classes (with mocked APIs)
├── test_evaluation.py       # Tests for evaluation metrics
├── test_training.py         # Tests for fine-tuning utilities (with mocked APIs)
└── README.md               # This file
```

## Running Tests

### Install Dependencies

First, install the test dependencies:

```bash
pip install -r requirements-dev.txt
```

### Run All Tests

```bash
# Run all tests with coverage
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_models.py

# Run specific test class
pytest tests/test_models.py::TestEDUModel

# Run specific test function
pytest tests/test_models.py::TestEDUModel::test_valid_root_edu
```

### Run Tests in Parallel

```bash
# Use multiple CPUs for faster execution
pytest -n auto
```

### Generate Coverage Report

```bash
# Generate HTML coverage report
pytest --cov=src --cov-report=html

# View report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### Run Only Fast Tests

```bash
# Skip slow tests
pytest -m "not slow"
```

### Run Tests by Category

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Skip API-dependent tests (useful for CI without API keys)
pytest -m "not requires_api"
```

## Test Coverage

The test suite aims for **>80% code coverage** across all modules:

- ✅ **src/models/** - EDU and Tree validation, serialization, navigation
- ✅ **src/data/loader.py** - File loading, UTF-8 BOM handling, split management
- ✅ **src/data/preprocessor.py** - Filtering, statistics, data splitting
- ✅ **src/parsers/** - All parser types, retry logic, statistics tracking (mocked APIs)
- ✅ **src/evaluation/** - UAS, LAS, F1 metrics, batch evaluation, per-relation metrics
- ✅ **src/training/** - Dataset preparation, JSONL creation, fine-tuning pipeline (mocked APIs)

## Test Philosophy

### Test-Driven Development (TDD)

All tests are written following TDD principles:

1. **Write tests first** - Tests define expected behavior
2. **Run tests (they fail)** - Verify tests catch issues
3. **Implement code** - Write minimal code to pass tests
4. **Run tests (they pass)** - Verify implementation is correct
5. **Refactor** - Improve code while keeping tests green

### Test Categories

- **Unit Tests**: Test individual functions/classes in isolation
- **Integration Tests**: Test interactions between components
- **Edge Cases**: Test boundary conditions and error handling
- **Mocked Tests**: Test components that depend on external APIs without making real calls

## Shared Fixtures (conftest.py)

Common test data is defined in `conftest.py` as pytest fixtures:

### EDU Fixtures
- `sample_edu_root` - ROOT EDU node
- `sample_edu_background` - Background relation EDU
- `sample_edu_elaboration` - Elaboration relation EDU

### Tree Fixtures
- `sample_simple_tree` - 3-EDU tree
- `sample_complex_tree` - 6-EDU tree with multiple levels
- `sample_scidtb_format` - Data in SciDTB format (with 'root' key)
- `sample_standard_format` - Data in standard format (with 'edus' key)

### Invalid Tree Fixtures (for validation testing)
- `invalid_tree_no_root` - Missing ROOT node
- `invalid_tree_multiple_roots` - Multiple ROOT nodes
- `invalid_tree_cycle` - Circular dependency
- `invalid_tree_forward_reference` - Parent ID >= child ID
- `invalid_tree_non_consecutive_ids` - Skipped EDU IDs

### File Fixtures
- `temp_data_dir` - Temporary directory for test files
- `sample_scidtb_file` - Single .edu.txt.dep file
- `sample_scidtb_dataset` - Full dataset structure (train/test splits)
- `sample_finetuning_jsonl` - JSONL training file

### API Fixtures
- `api_key` - Test API key
- `mock_openai_response` - Mocked OpenAI API response

## Writing New Tests

When adding new functionality, follow this pattern:

```python
# tests/test_new_feature.py
"""
Unit tests for src/new_feature.py

Tests cover:
- Primary functionality
- Edge cases
- Error handling
"""
import pytest
from src.new_feature import NewClass


class TestNewClass:
    """Test cases for NewClass."""

    def test_basic_functionality(self):
        """Test the main use case."""
        obj = NewClass()
        result = obj.do_something()
        assert result is not None

    def test_edge_case(self):
        """Test boundary condition."""
        obj = NewClass()
        result = obj.do_something(edge_input=True)
        assert result == expected_value

    def test_error_handling(self):
        """Test that errors are handled properly."""
        obj = NewClass()
        with pytest.raises(ValueError):
            obj.do_something(invalid_input)
```

## Continuous Integration

These tests should be run automatically in CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    pip install -r requirements-dev.txt
    pytest --cov=src --cov-report=xml

- name: Upload coverage
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
```

## Best Practices

1. **Keep tests isolated** - Each test should be independent
2. **Use fixtures** - Reuse common test data via conftest.py
3. **Test one thing** - Each test should verify a single behavior
4. **Use descriptive names** - Test names should explain what they test
5. **Mock external dependencies** - Don't make real API calls in tests
6. **Test edge cases** - Include boundary conditions and error paths
7. **Maintain high coverage** - Aim for >80% code coverage
8. **Run tests frequently** - After every change, before commits

## Troubleshooting

### Tests Fail Due to Missing Dependencies

```bash
pip install -r requirements-dev.txt
```

### Tests Fail Due to Import Errors

Make sure you're running from the project root:

```bash
cd /path/to/scidtb-discourse-parsing
pytest
```

### Coverage Report Not Generated

Install pytest-cov:

```bash
pip install pytest-cov
```

### Tests Run Slowly

Use parallel execution:

```bash
pytest -n auto
```

## Future Enhancements

- [ ] Add performance benchmarks
- [ ] Add integration tests with real (small) dataset
- [ ] Add tests for experiment scripts
- [ ] Add mutation testing
- [ ] Add property-based testing with Hypothesis

## Contact

For questions about tests, refer to the main project documentation or open an issue on GitHub.
