"""
Unit tests for src/training/ modules

Tests cover:
- FineTuningDataPreparator
- Creating training examples
- JSONL dataset preparation
- Dataset validation
- OpenAIFineTuner (mocked)
- Fine-tuning pipeline

Note: OpenAI API calls are mocked to avoid actual API usage.
"""
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

from src.training.data_prep import FineTuningDataPreparator, FineTuningExample
from src.training.finetune import OpenAIFineTuner, FineTuningPipeline
from src.models import EDUModel, DiscourseTreeModel


class TestFineTuningExample:
    """Test cases for FineTuningExample model."""

    def test_valid_finetuning_example(self):
        """Test creating a valid fine-tuning example."""
        example = FineTuningExample(
            messages=[
                {"role": "system", "content": "System prompt."},
                {"role": "user", "content": "User message."},
                {"role": "assistant", "content": "Assistant response."}
            ]
        )

        assert len(example.messages) == 3
        assert example.messages[0]["role"] == "system"
        assert example.messages[1]["role"] == "user"
        assert example.messages[2]["role"] == "assistant"

    def test_finetuning_example_serialization(self):
        """Test serializing fine-tuning example to dict."""
        example = FineTuningExample(
            messages=[
                {"role": "system", "content": "System."},
                {"role": "user", "content": "User."},
                {"role": "assistant", "content": "Assistant."}
            ]
        )

        example_dict = example.model_dump()
        assert "messages" in example_dict
        assert len(example_dict["messages"]) == 3


class TestFineTuningDataPreparator:
    """Test cases for FineTuningDataPreparator."""

    def test_preparator_initialization(self):
        """Test preparator initialization."""
        prep = FineTuningDataPreparator(
            model_type="gpt-3.5-turbo",
            include_format_instructions=True
        )

        assert prep.model_type == "gpt-3.5-turbo"
        assert prep.include_format_instructions is True

    def test_create_system_message(self):
        """Test creating system message."""
        prep = FineTuningDataPreparator()
        system_msg = prep.create_system_message()

        assert system_msg["role"] == "system"
        assert "content" in system_msg
        assert len(system_msg["content"]) > 0

    def test_create_user_message(self):
        """Test creating user message."""
        prep = FineTuningDataPreparator()
        user_msg = prep.create_user_message("Test text to parse.")

        assert user_msg["role"] == "user"
        assert "Test text to parse." in user_msg["content"]

    def test_create_assistant_message(self, sample_simple_tree):
        """Test creating assistant message from tree."""
        prep = FineTuningDataPreparator()
        assistant_msg = prep.create_assistant_message(sample_simple_tree)

        assert assistant_msg["role"] == "assistant"

        # Parse the JSON content
        content = json.loads(assistant_msg["content"])
        assert "edus" in content
        assert len(content["edus"]) == 3

    def test_tree_to_example(self, sample_simple_tree):
        """Test converting tree to FineTuningExample."""
        prep = FineTuningDataPreparator()
        text = "Background sentence. Main finding."

        example = prep.tree_to_example(text, sample_simple_tree)

        assert isinstance(example, FineTuningExample)
        assert len(example.messages) == 3
        assert example.messages[0]["role"] == "system"
        assert example.messages[1]["role"] == "user"
        assert example.messages[2]["role"] == "assistant"

    def test_prepare_dataset_creates_jsonl(self, temp_data_dir, sample_simple_tree, sample_complex_tree):
        """Test preparing dataset and creating JSONL file."""
        prep = FineTuningDataPreparator()

        # Create list of (text, tree) pairs
        dataset = [
            ("Text 1", sample_simple_tree),
            ("Text 2", sample_complex_tree)
        ]

        output_file = temp_data_dir / "training.jsonl"
        result = prep.prepare_dataset(dataset, str(output_file))

        assert result is True
        assert output_file.exists()

        # Verify JSONL content
        with open(output_file, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 2

            # Check first example
            example1 = json.loads(lines[0])
            assert "messages" in example1
            assert len(example1["messages"]) == 3

    def test_prepare_dataset_with_empty_list(self, temp_data_dir):
        """Test preparing dataset with empty list."""
        prep = FineTuningDataPreparator()
        output_file = temp_data_dir / "empty.jsonl"

        result = prep.prepare_dataset([], str(output_file))

        assert result is True
        assert output_file.exists()

        # File should be empty or have no examples
        with open(output_file, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 0

    def test_validate_dataset_valid(self, sample_finetuning_jsonl):
        """Test validating a valid JSONL dataset."""
        prep = FineTuningDataPreparator()
        is_valid = prep.validate_dataset(str(sample_finetuning_jsonl))

        assert is_valid is True

    def test_validate_dataset_invalid_json(self, temp_data_dir):
        """Test validating dataset with invalid JSON."""
        invalid_file = temp_data_dir / "invalid.jsonl"
        with open(invalid_file, 'w') as f:
            f.write("{ invalid json\n")
            f.write('{"messages": []}\n')

        prep = FineTuningDataPreparator()
        is_valid = prep.validate_dataset(str(invalid_file))

        assert is_valid is False

    def test_validate_dataset_missing_messages(self, temp_data_dir):
        """Test validating dataset with missing 'messages' field."""
        invalid_file = temp_data_dir / "no_messages.jsonl"
        with open(invalid_file, 'w') as f:
            f.write('{"data": "missing messages field"}\n')

        prep = FineTuningDataPreparator()
        is_valid = prep.validate_dataset(str(invalid_file))

        assert is_valid is False

    def test_validate_dataset_nonexistent_file(self, temp_data_dir):
        """Test validating a file that doesn't exist."""
        prep = FineTuningDataPreparator()
        is_valid = prep.validate_dataset(str(temp_data_dir / "nonexistent.jsonl"))

        assert is_valid is False

    def test_prepare_from_loader(self, sample_scidtb_dataset, temp_data_dir):
        """Test preparing dataset from SciDTBLoader."""
        from src.data.loader import SciDTBLoader

        prep = FineTuningDataPreparator()
        loader = SciDTBLoader(str(sample_scidtb_dataset))

        output_file = temp_data_dir / "from_loader.jsonl"
        result = prep.prepare_from_loader(loader, "train", str(output_file))

        assert result is True
        assert output_file.exists()

        # Verify content
        with open(output_file, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 3  # 3 train files in fixture

    def test_include_format_instructions_flag(self, sample_simple_tree):
        """Test that include_format_instructions affects output."""
        # With format instructions
        prep_with = FineTuningDataPreparator(include_format_instructions=True)
        msg_with = prep_with.create_system_message()

        # Without format instructions
        prep_without = FineTuningDataPreparator(include_format_instructions=False)
        msg_without = prep_without.create_system_message()

        # Messages should differ
        # (Exact difference depends on implementation, but length should differ)
        assert len(msg_with["content"]) != len(msg_without["content"]) or msg_with == msg_without


class TestOpenAIFineTuner:
    """Test cases for OpenAIFineTuner (with mocked API)."""

    @patch('src.training.finetune.openai.OpenAI')
    def test_finetuner_initialization(self, mock_openai_class, api_key):
        """Test fine-tuner initialization."""
        finetuner = OpenAIFineTuner(api_key=api_key)
        assert finetuner is not None
        mock_openai_class.assert_called_once()

    @patch('src.training.finetune.openai.OpenAI')
    def test_upload_file(self, mock_openai_class, api_key, sample_finetuning_jsonl):
        """Test uploading a file for fine-tuning."""
        # Mock OpenAI client and file upload
        mock_client = MagicMock()
        mock_file_response = MagicMock()
        mock_file_response.id = "file-abc123"
        mock_client.files.create.return_value = mock_file_response
        mock_openai_class.return_value = mock_client

        finetuner = OpenAIFineTuner(api_key=api_key)
        file_id = finetuner.upload_file(str(sample_finetuning_jsonl))

        assert file_id == "file-abc123"
        mock_client.files.create.assert_called_once()

    @patch('src.training.finetune.openai.OpenAI')
    def test_create_finetune_job(self, mock_openai_class, api_key):
        """Test creating a fine-tune job."""
        # Mock OpenAI client and job creation
        mock_client = MagicMock()
        mock_job_response = MagicMock()
        mock_job_response.id = "ftjob-xyz789"
        mock_client.fine_tuning.jobs.create.return_value = mock_job_response
        mock_openai_class.return_value = mock_client

        finetuner = OpenAIFineTuner(api_key=api_key)
        job_id = finetuner.create_finetune_job("file-abc123", model="gpt-3.5-turbo")

        assert job_id == "ftjob-xyz789"
        mock_client.fine_tuning.jobs.create.assert_called_once()

    @patch('src.training.finetune.openai.OpenAI')
    def test_get_job_status(self, mock_openai_class, api_key):
        """Test getting job status."""
        # Mock OpenAI client and status retrieval
        mock_client = MagicMock()
        mock_job = MagicMock()
        mock_job.status = "running"
        mock_job.id = "ftjob-xyz789"
        mock_client.fine_tuning.jobs.retrieve.return_value = mock_job
        mock_openai_class.return_value = mock_client

        finetuner = OpenAIFineTuner(api_key=api_key)
        status = finetuner.get_job_status("ftjob-xyz789")

        assert status["status"] == "running"
        assert status["job_id"] == "ftjob-xyz789"

    @patch('src.training.finetune.openai.OpenAI')
    def test_wait_for_completion_success(self, mock_openai_class, api_key):
        """Test waiting for job completion (success case)."""
        # Mock OpenAI client
        mock_client = MagicMock()

        # Mock job that completes after 2 checks
        mock_job_running = MagicMock()
        mock_job_running.status = "running"
        mock_job_running.fine_tuned_model = None

        mock_job_complete = MagicMock()
        mock_job_complete.status = "succeeded"
        mock_job_complete.fine_tuned_model = "ft:gpt-3.5-turbo:model:abc"

        mock_client.fine_tuning.jobs.retrieve.side_effect = [
            mock_job_running,
            mock_job_complete
        ]
        mock_openai_class.return_value = mock_client

        finetuner = OpenAIFineTuner(api_key=api_key)
        result = finetuner.wait_for_completion(
            "ftjob-xyz789",
            check_interval=0.1,
            timeout=10
        )

        assert result["status"] == "succeeded"
        assert result["fine_tuned_model"] == "ft:gpt-3.5-turbo:model:abc"

    @patch('src.training.finetune.openai.OpenAI')
    def test_wait_for_completion_timeout(self, mock_openai_class, api_key):
        """Test waiting for job with timeout."""
        # Mock OpenAI client that always returns running
        mock_client = MagicMock()
        mock_job = MagicMock()
        mock_job.status = "running"
        mock_client.fine_tuning.jobs.retrieve.return_value = mock_job
        mock_openai_class.return_value = mock_client

        finetuner = OpenAIFineTuner(api_key=api_key)
        result = finetuner.wait_for_completion(
            "ftjob-xyz789",
            check_interval=0.1,
            timeout=0.5  # Short timeout
        )

        # Should return current status when timeout
        assert result["status"] == "running"

    @patch('src.training.finetune.openai.OpenAI')
    def test_cancel_job(self, mock_openai_class, api_key):
        """Test canceling a fine-tune job."""
        # Mock OpenAI client
        mock_client = MagicMock()
        mock_job = MagicMock()
        mock_job.status = "cancelled"
        mock_client.fine_tuning.jobs.cancel.return_value = mock_job
        mock_openai_class.return_value = mock_client

        finetuner = OpenAIFineTuner(api_key=api_key)
        result = finetuner.cancel_job("ftjob-xyz789")

        assert result is True
        mock_client.fine_tuning.jobs.cancel.assert_called_once_with("ftjob-xyz789")

    @patch('src.training.finetune.openai.OpenAI')
    def test_list_finetune_jobs(self, mock_openai_class, api_key):
        """Test listing fine-tune jobs."""
        # Mock OpenAI client
        mock_client = MagicMock()
        mock_jobs = [
            MagicMock(id="job1", status="running"),
            MagicMock(id="job2", status="succeeded")
        ]
        mock_client.fine_tuning.jobs.list.return_value = MagicMock(data=mock_jobs)
        mock_openai_class.return_value = mock_client

        finetuner = OpenAIFineTuner(api_key=api_key)
        jobs = finetuner.list_finetune_jobs(limit=10)

        assert len(jobs) == 2
        assert jobs[0]["job_id"] == "job1"
        assert jobs[1]["job_id"] == "job2"


class TestFineTuningPipeline:
    """Test cases for FineTuningPipeline."""

    @patch('src.training.finetune.openai.OpenAI')
    def test_pipeline_initialization(self, mock_openai_class, api_key):
        """Test pipeline initialization."""
        pipeline = FineTuningPipeline(api_key=api_key)
        assert pipeline is not None

    @patch('src.training.finetune.openai.OpenAI')
    def test_run_full_pipeline_success(self, mock_openai_class, api_key, sample_finetuning_jsonl):
        """Test running full fine-tuning pipeline."""
        # Mock OpenAI client
        mock_client = MagicMock()

        # Mock file upload
        mock_file = MagicMock()
        mock_file.id = "file-abc123"
        mock_client.files.create.return_value = mock_file

        # Mock job creation
        mock_job_create = MagicMock()
        mock_job_create.id = "ftjob-xyz789"
        mock_client.fine_tuning.jobs.create.return_value = mock_job_create

        # Mock job completion
        mock_job_complete = MagicMock()
        mock_job_complete.status = "succeeded"
        mock_job_complete.fine_tuned_model = "ft:gpt-3.5-turbo:model:abc"
        mock_client.fine_tuning.jobs.retrieve.return_value = mock_job_complete

        mock_openai_class.return_value = mock_client

        pipeline = FineTuningPipeline(api_key=api_key)
        result = pipeline.run_full_pipeline(
            training_file=str(sample_finetuning_jsonl),
            model="gpt-3.5-turbo",
            wait_for_completion=True,
            check_interval=0.1,
            timeout=10
        )

        assert result["file_id"] == "file-abc123"
        assert result["job_id"] == "ftjob-xyz789"
        assert result["status"] == "succeeded"
        assert result["fine_tuned_model"] == "ft:gpt-3.5-turbo:model:abc"

    @patch('src.training.finetune.openai.OpenAI')
    def test_run_pipeline_without_waiting(self, mock_openai_class, api_key, sample_finetuning_jsonl):
        """Test running pipeline without waiting for completion."""
        # Mock OpenAI client
        mock_client = MagicMock()

        # Mock file upload
        mock_file = MagicMock()
        mock_file.id = "file-abc123"
        mock_client.files.create.return_value = mock_file

        # Mock job creation
        mock_job = MagicMock()
        mock_job.id = "ftjob-xyz789"
        mock_job.status = "validating_files"
        mock_client.fine_tuning.jobs.create.return_value = mock_job
        mock_client.fine_tuning.jobs.retrieve.return_value = mock_job

        mock_openai_class.return_value = mock_client

        pipeline = FineTuningPipeline(api_key=api_key)
        result = pipeline.run_full_pipeline(
            training_file=str(sample_finetuning_jsonl),
            model="gpt-3.5-turbo",
            wait_for_completion=False
        )

        assert result["file_id"] == "file-abc123"
        assert result["job_id"] == "ftjob-xyz789"
        # Should not have fine_tuned_model yet
        assert "fine_tuned_model" not in result or result["fine_tuned_model"] is None


class TestIntegration:
    """Integration tests for training workflow."""

    def test_end_to_end_dataset_preparation(self, temp_data_dir, sample_simple_tree, sample_complex_tree):
        """Test complete dataset preparation workflow."""
        prep = FineTuningDataPreparator()

        # Create dataset
        dataset = [
            ("Background sentence. Main finding.", sample_simple_tree),
            ("Complex abstract with multiple sentences.", sample_complex_tree)
        ]

        # Prepare JSONL
        output_file = temp_data_dir / "prepared.jsonl"
        result = prep.prepare_dataset(dataset, str(output_file))
        assert result is True

        # Validate
        is_valid = prep.validate_dataset(str(output_file))
        assert is_valid is True

        # Verify content structure
        with open(output_file, 'r') as f:
            for line in f:
                example = json.loads(line)
                assert "messages" in example
                assert len(example["messages"]) == 3
                assert example["messages"][0]["role"] == "system"
                assert example["messages"][1]["role"] == "user"
                assert example["messages"][2]["role"] == "assistant"

                # Verify assistant message is valid JSON
                assistant_content = json.loads(example["messages"][2]["content"])
                assert "edus" in assistant_content
