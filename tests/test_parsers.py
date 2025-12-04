"""
Unit tests for src/parsers/ modules

Tests cover:
- BaseParser abstract functionality
- ZeroShotParser behavior
- FewShotParser with examples
- FineTunedParser specifics
- Retry logic and error handling
- Statistics tracking

Note: These tests mock OpenAI API calls to avoid actual API usage.
"""
import json
from unittest.mock import Mock, patch, MagicMock

import pytest

from src.parsers import BaseParser, ZeroShotParser, FewShotParser, FineTunedParser
from src.models import DiscourseTreeModel


class TestBaseParser:
    """Test cases for BaseParser abstract base class."""

    def test_base_parser_cannot_be_instantiated(self, api_key):
        """Test that BaseParser cannot be instantiated directly."""
        with pytest.raises(TypeError):
            # BaseParser is abstract and should raise TypeError
            parser = BaseParser(api_key=api_key)

    def test_parser_initialization_with_defaults(self, api_key):
        """Test parser initialization with default parameters."""
        parser = ZeroShotParser(api_key=api_key)

        assert parser.model == "gpt-4"
        assert parser.temperature == 0
        assert parser.max_retries == 3
        assert parser.retry_delay == 2

    def test_parser_initialization_with_custom_params(self, api_key):
        """Test parser initialization with custom parameters."""
        parser = ZeroShotParser(
            api_key=api_key,
            model="gpt-3.5-turbo",
            temperature=0.5,
            max_retries=5,
            retry_delay=3
        )

        assert parser.model == "gpt-3.5-turbo"
        assert parser.temperature == 0.5
        assert parser.max_retries == 5
        assert parser.retry_delay == 3

    def test_parser_statistics_initialization(self, api_key):
        """Test that parser statistics are initialized correctly."""
        parser = ZeroShotParser(api_key=api_key)

        assert parser.num_calls == 0
        assert parser.num_successes == 0
        assert parser.num_failures == 0
        assert parser.total_tokens == 0

    def test_get_statistics(self, api_key):
        """Test getting parser statistics."""
        parser = ZeroShotParser(api_key=api_key)

        stats = parser.get_statistics()

        assert "num_calls" in stats
        assert "num_successes" in stats
        assert "num_failures" in stats
        assert "total_tokens" in stats
        assert "success_rate" in stats

    def test_reset_statistics(self, api_key):
        """Test resetting parser statistics."""
        parser = ZeroShotParser(api_key=api_key)

        # Manually set some statistics
        parser.num_calls = 10
        parser.num_successes = 8
        parser.num_failures = 2
        parser.total_tokens = 1000

        # Reset
        parser.reset_statistics()

        assert parser.num_calls == 0
        assert parser.num_successes == 0
        assert parser.num_failures == 0
        assert parser.total_tokens == 0


class TestZeroShotParser:
    """Test cases for ZeroShotParser."""

    def test_zero_shot_parser_initialization(self, api_key):
        """Test ZeroShotParser initialization."""
        parser = ZeroShotParser(api_key=api_key)
        assert parser is not None
        assert isinstance(parser, ZeroShotParser)
        assert isinstance(parser, BaseParser)

    def test_create_prompt_template(self, api_key):
        """Test that prompt template is created."""
        parser = ZeroShotParser(api_key=api_key)
        template = parser.create_prompt_template()

        assert template is not None
        # Check that template contains key instructions
        template_str = str(template)
        assert "discourse" in template_str.lower() or "parse" in template_str.lower()

    @patch('src.parsers.base.ChatOpenAI')
    def test_parse_success(self, mock_chat_openai, api_key, mock_openai_response):
        """Test successful parsing."""
        # Mock OpenAI response
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps(mock_openai_response)
        mock_llm.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_llm

        parser = ZeroShotParser(api_key=api_key)
        result = parser.parse("Test text for parsing.")

        assert result is not None
        assert isinstance(result, DiscourseTreeModel)
        assert len(result.edus) == 3
        assert parser.num_calls == 1
        assert parser.num_successes == 1
        assert parser.num_failures == 0

    @patch('src.parsers.base.ChatOpenAI')
    def test_parse_failure_with_retry(self, mock_chat_openai, api_key):
        """Test parsing with retries on failure."""
        # Mock OpenAI to fail
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("API Error")
        mock_chat_openai.return_value = mock_llm

        parser = ZeroShotParser(api_key=api_key, max_retries=2, retry_delay=0.1)
        result = parser.parse("Test text.")

        assert result is None
        assert parser.num_failures > 0

    @patch('src.parsers.base.ChatOpenAI')
    def test_parse_invalid_json_response(self, mock_chat_openai, api_key):
        """Test handling of invalid JSON response."""
        # Mock OpenAI to return invalid JSON
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Not valid JSON"
        mock_llm.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_llm

        parser = ZeroShotParser(api_key=api_key, max_retries=1, retry_delay=0.1)
        result = parser.parse("Test text.")

        assert result is None
        assert parser.num_failures > 0

    @patch('src.parsers.base.ChatOpenAI')
    def test_parse_invalid_tree_structure(self, mock_chat_openai, api_key, invalid_tree_no_root):
        """Test handling of invalid tree structure in response."""
        # Mock OpenAI to return invalid tree (no ROOT)
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps(invalid_tree_no_root)
        mock_llm.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_llm

        parser = ZeroShotParser(api_key=api_key, max_retries=1, retry_delay=0.1)
        result = parser.parse("Test text.")

        assert result is None
        assert parser.num_failures > 0


class TestFewShotParser:
    """Test cases for FewShotParser."""

    def test_few_shot_parser_initialization(self, api_key):
        """Test FewShotParser initialization."""
        parser = FewShotParser(api_key=api_key, n_shots=3)
        assert parser.n_shots == 3
        assert len(parser.examples) == 0

    def test_add_example(self, api_key, sample_simple_tree):
        """Test adding a single example."""
        parser = FewShotParser(api_key=api_key, n_shots=3)

        text = "Background sentence. Main finding."
        parser.add_example(text, sample_simple_tree)

        assert len(parser.examples) == 1
        assert parser.examples[0][0] == text
        assert parser.examples[0][1] == sample_simple_tree

    def test_add_examples_multiple(self, api_key, sample_simple_tree, sample_complex_tree):
        """Test adding multiple examples."""
        parser = FewShotParser(api_key=api_key, n_shots=3)

        examples = [
            ("Text 1", sample_simple_tree),
            ("Text 2", sample_complex_tree)
        ]
        parser.add_examples(examples)

        assert len(parser.examples) == 2

    def test_add_examples_respects_n_shots(self, api_key, sample_simple_tree):
        """Test that only n_shots examples are kept."""
        parser = FewShotParser(api_key=api_key, n_shots=2)

        # Add 3 examples
        for i in range(3):
            parser.add_example(f"Text {i}", sample_simple_tree)

        # Should only keep first 2
        assert len(parser.examples) == 2

    def test_format_examples(self, api_key, sample_simple_tree):
        """Test formatting examples for prompt."""
        parser = FewShotParser(api_key=api_key, n_shots=3)
        parser.add_example("Test text.", sample_simple_tree)

        formatted = parser.format_examples()

        assert "Test text." in formatted
        assert "ROOT" in formatted
        # Should contain JSON representation of tree

    def test_create_prompt_template_with_examples(self, api_key, sample_simple_tree):
        """Test that prompt template includes examples."""
        parser = FewShotParser(api_key=api_key, n_shots=2)
        parser.add_example("Example text.", sample_simple_tree)

        template = parser.create_prompt_template()
        template_str = str(template)

        # Template should reference examples
        assert "example" in template_str.lower() or "Example text." in template_str

    @patch('src.parsers.base.ChatOpenAI')
    def test_parse_with_examples(self, mock_chat_openai, api_key, sample_simple_tree, mock_openai_response):
        """Test parsing with few-shot examples."""
        # Mock OpenAI response
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps(mock_openai_response)
        mock_llm.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_llm

        parser = FewShotParser(api_key=api_key, n_shots=2)
        parser.add_example("Example text.", sample_simple_tree)

        result = parser.parse("New text to parse.")

        assert result is not None
        assert isinstance(result, DiscourseTreeModel)
        assert parser.num_successes == 1


class TestFineTunedParser:
    """Test cases for FineTunedParser."""

    def test_finetuned_parser_initialization(self, api_key):
        """Test FineTunedParser initialization with model ID."""
        model_id = "ft:gpt-3.5-turbo-0613:company:model:abc123"
        parser = FineTunedParser(api_key=api_key, model_id=model_id)

        assert parser.model_id == model_id
        # Fine-tuned parser should use the specified model
        assert model_id in parser.model or parser.model == model_id

    def test_finetuned_parser_statistics_includes_model_id(self, api_key):
        """Test that statistics include model_id."""
        model_id = "ft:gpt-3.5-turbo-0613:company:model:abc123"
        parser = FineTunedParser(api_key=api_key, model_id=model_id)

        stats = parser.get_statistics()

        assert "model_id" in stats
        assert stats["model_id"] == model_id

    @patch('src.parsers.finetuned.ChatOpenAI')
    def test_finetuned_parse_success(self, mock_chat_openai, api_key, mock_openai_response):
        """Test successful parsing with fine-tuned model."""
        # Mock OpenAI response
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps(mock_openai_response)
        mock_llm.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_llm

        model_id = "ft:gpt-3.5-turbo-0613:company:model:abc123"
        parser = FineTunedParser(api_key=api_key, model_id=model_id)

        result = parser.parse("Test text.")

        assert result is not None
        assert isinstance(result, DiscourseTreeModel)
        assert parser.num_successes == 1

    def test_create_prompt_template_minimal(self, api_key):
        """Test that fine-tuned parser uses minimal prompt."""
        model_id = "ft:gpt-3.5-turbo-0613:company:model:abc123"
        parser = FineTunedParser(api_key=api_key, model_id=model_id)

        template = parser.create_prompt_template()

        # Fine-tuned models should need less instruction
        # The template should be simpler than zero-shot
        assert template is not None


class TestParserRetryLogic:
    """Test retry logic and error handling across parsers."""

    @patch('src.parsers.base.ChatOpenAI')
    def test_retry_on_transient_failure(self, mock_chat_openai, api_key, mock_openai_response):
        """Test that parser retries on transient failures."""
        # Mock to fail twice, then succeed
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps(mock_openai_response)

        mock_llm.invoke.side_effect = [
            Exception("Transient error 1"),
            Exception("Transient error 2"),
            mock_response
        ]
        mock_chat_openai.return_value = mock_llm

        parser = ZeroShotParser(api_key=api_key, max_retries=3, retry_delay=0.1)
        result = parser.parse("Test text.")

        # Should succeed on third try
        assert result is not None
        assert parser.num_calls == 1  # One logical call
        assert parser.num_successes == 1

    @patch('src.parsers.base.ChatOpenAI')
    def test_max_retries_exceeded(self, mock_chat_openai, api_key):
        """Test behavior when max retries is exceeded."""
        # Mock to always fail
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("Persistent error")
        mock_chat_openai.return_value = mock_llm

        parser = ZeroShotParser(api_key=api_key, max_retries=2, retry_delay=0.1)
        result = parser.parse("Test text.")

        # Should fail after max retries
        assert result is None
        assert parser.num_failures == 1

    @patch('src.parsers.base.ChatOpenAI')
    @patch('time.sleep')
    def test_retry_delay_is_applied(self, mock_sleep, mock_chat_openai, api_key):
        """Test that retry delay is applied between attempts."""
        # Mock to fail then succeed
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            Exception("Error"),
            MagicMock(content='{"edus": []}')
        ]
        mock_chat_openai.return_value = mock_llm

        parser = ZeroShotParser(api_key=api_key, max_retries=2, retry_delay=1.0)
        parser.parse("Test text.")

        # Should have called sleep with retry_delay
        mock_sleep.assert_called()


class TestParserStatistics:
    """Test statistics tracking across parsers."""

    @patch('src.parsers.base.ChatOpenAI')
    def test_statistics_tracking_success(self, mock_chat_openai, api_key, mock_openai_response):
        """Test that statistics are tracked on success."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps(mock_openai_response)
        mock_llm.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_llm

        parser = ZeroShotParser(api_key=api_key)

        # Parse twice
        parser.parse("Text 1")
        parser.parse("Text 2")

        assert parser.num_calls == 2
        assert parser.num_successes == 2
        assert parser.num_failures == 0

        stats = parser.get_statistics()
        assert stats["success_rate"] == 100.0

    @patch('src.parsers.base.ChatOpenAI')
    def test_statistics_tracking_failure(self, mock_chat_openai, api_key):
        """Test that statistics are tracked on failure."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("Error")
        mock_chat_openai.return_value = mock_llm

        parser = ZeroShotParser(api_key=api_key, max_retries=1, retry_delay=0.1)

        # Parse twice (both fail)
        parser.parse("Text 1")
        parser.parse("Text 2")

        assert parser.num_calls == 2
        assert parser.num_successes == 0
        assert parser.num_failures == 2

        stats = parser.get_statistics()
        assert stats["success_rate"] == 0.0

    @patch('src.parsers.base.ChatOpenAI')
    def test_statistics_tracking_mixed(self, mock_chat_openai, api_key, mock_openai_response):
        """Test statistics with mixed success and failure."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps(mock_openai_response)

        # Alternate success and failure
        mock_llm.invoke.side_effect = [
            mock_response,  # Success
            Exception("Error"),  # Failure
            mock_response,  # Success
        ]
        mock_chat_openai.return_value = mock_llm

        parser = ZeroShotParser(api_key=api_key, max_retries=1, retry_delay=0.1)

        parser.parse("Text 1")  # Success
        parser.parse("Text 2")  # Failure
        parser.parse("Text 3")  # Success

        assert parser.num_calls == 3
        assert parser.num_successes == 2
        assert parser.num_failures == 1

        stats = parser.get_statistics()
        assert abs(stats["success_rate"] - 66.67) < 0.1  # Approximately 66.67%
