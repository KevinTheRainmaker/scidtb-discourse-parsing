"""
Fine-tuning utilities for OpenAI models.
"""
import time
import json
from pathlib import Path
from typing import Optional, Dict, List
import openai
from ..utils.logger import get_logger

logger = get_logger(__name__)

class OpenAIFineTuner:
    """Manage OpenAI fine-tuning jobs."""
    
    def __init__(self, api_key: str):
        """
        Initialize fine-tuner.
        
        Args:
            api_key: OpenAI API key
        """
        openai.api_key = api_key
        self.api_key = api_key
    
    def upload_file(self, file_path: Path) -> str:
        """
        Upload training file to OpenAI.
        
        Args:
            file_path: Path to JSONL training file
            
        Returns:
            File ID
        """
        logger.info(f"Uploading file: {file_path}")
        
        with open(file_path, 'rb') as f:
            response = openai.File.create(
                file=f,
                purpose='fine-tune'
            )
        
        file_id = response['id']
        logger.info(f"✓ File uploaded successfully: {file_id}")
        
        return file_id
    
    def create_finetune_job(
        self,
        training_file_id: str,
        model: str = "gpt-3.5-turbo-1106",
        n_epochs: int = 3,
        suffix: Optional[str] = None,
        validation_file_id: Optional[str] = None
    ) -> str:
        """
        Create a fine-tuning job.
        
        Args:
            training_file_id: ID of uploaded training file
            model: Base model to fine-tune
            n_epochs: Number of training epochs
            suffix: Suffix for fine-tuned model name
            validation_file_id: Optional validation file ID
            
        Returns:
            Job ID
        """
        logger.info(f"Creating fine-tuning job for model: {model}")
        
        params = {
            "training_file": training_file_id,
            "model": model,
            "hyperparameters": {
                "n_epochs": n_epochs
            }
        }
        
        if suffix:
            params["suffix"] = suffix
        
        if validation_file_id:
            params["validation_file"] = validation_file_id
        
        response = openai.FineTuningJob.create(**params)
        
        job_id = response['id']
        logger.info(f"✓ Fine-tuning job created: {job_id}")
        logger.info(f"  Monitor at: https://platform.openai.com/finetune/{job_id}")
        
        return job_id
    
    def get_job_status(self, job_id: str) -> Dict:
        """
        Get status of a fine-tuning job.
        
        Args:
            job_id: Job ID
            
        Returns:
            Job status dictionary
        """
        response = openai.FineTuningJob.retrieve(job_id)
        
        return {
            "id": response['id'],
            "status": response['status'],
            "model": response.get('fine_tuned_model'),
            "created_at": response['created_at'],
            "finished_at": response.get('finished_at'),
            "trained_tokens": response.get('trained_tokens'),
            "error": response.get('error')
        }
    
    def wait_for_completion(
        self,
        job_id: str,
        check_interval: int = 60,
        timeout: int = 7200
    ) -> str:
        """
        Wait for fine-tuning job to complete.
        
        Args:
            job_id: Job ID
            check_interval: Seconds between status checks
            timeout: Maximum seconds to wait
            
        Returns:
            Fine-tuned model ID
            
        Raises:
            TimeoutError: If job doesn't complete within timeout
            RuntimeError: If job fails
        """
        logger.info(f"Waiting for job {job_id} to complete...")
        
        start_time = time.time()
        
        while True:
            elapsed = time.time() - start_time
            
            if elapsed > timeout:
                raise TimeoutError(
                    f"Fine-tuning job did not complete within {timeout} seconds"
                )
            
            status = self.get_job_status(job_id)
            
            if status['status'] == 'succeeded':
                model_id = status['model']
                logger.info(f"✓ Fine-tuning completed successfully!")
                logger.info(f"  Model ID: {model_id}")
                return model_id
            
            elif status['status'] == 'failed':
                error = status.get('error', 'Unknown error')
                raise RuntimeError(f"Fine-tuning job failed: {error}")
            
            elif status['status'] in ['cancelled', 'expired']:
                raise RuntimeError(f"Fine-tuning job {status['status']}")
            
            else:
                logger.info(
                    f"  Status: {status['status']} "
                    f"(elapsed: {elapsed:.0f}s)"
                )
                time.sleep(check_interval)
    
    def list_finetune_jobs(self, limit: int = 10) -> List[Dict]:
        """
        List recent fine-tuning jobs.
        
        Args:
            limit: Maximum number of jobs to return
            
        Returns:
            List of job information dictionaries
        """
        response = openai.FineTuningJob.list(limit=limit)
        
        jobs = []
        for job in response['data']:
            jobs.append({
                "id": job['id'],
                "status": job['status'],
                "model": job.get('fine_tuned_model'),
                "created_at": job['created_at']
            })
        
        return jobs
    
    def cancel_job(self, job_id: str) -> Dict:
        """
        Cancel a fine-tuning job.
        
        Args:
            job_id: Job ID
            
        Returns:
            Updated job status
        """
        logger.info(f"Cancelling job: {job_id}")
        response = openai.FineTuningJob.cancel(job_id)
        logger.info(f"✓ Job cancelled")
        return response

class FineTuningPipeline:
    """End-to-end fine-tuning pipeline."""
    
    def __init__(self, api_key: str, output_dir: Path):
        """
        Initialize pipeline.
        
        Args:
            api_key: OpenAI API key
            output_dir: Directory for outputs
        """
        self.finetuner = OpenAIFineTuner(api_key)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run_full_pipeline(
        self,
        training_file: Path,
        model: str = "gpt-3.5-turbo-1106",
        n_epochs: int = 3,
        suffix: Optional[str] = "scidtb",
        validation_file: Optional[Path] = None,
        wait_for_completion: bool = True
    ) -> Dict[str, str]:
        """
        Run complete fine-tuning pipeline.
        
        Args:
            training_file: Path to training JSONL file
            model: Base model
            n_epochs: Training epochs
            suffix: Model name suffix
            validation_file: Optional validation file
            wait_for_completion: Whether to wait for job completion
            
        Returns:
            Dictionary with job_id and model_id (if completed)
        """
        logger.info("="*60)
        logger.info("Starting Fine-tuning Pipeline")
        logger.info("="*60)
        
        # Upload training file
        training_file_id = self.finetuner.upload_file(training_file)
        
        # Upload validation file if provided
        validation_file_id = None
        if validation_file:
            validation_file_id = self.finetuner.upload_file(validation_file)
        
        # Create fine-tuning job
        job_id = self.finetuner.create_finetune_job(
            training_file_id=training_file_id,
            model=model,
            n_epochs=n_epochs,
            suffix=suffix,
            validation_file_id=validation_file_id
        )
        
        result = {
            "job_id": job_id,
            "training_file_id": training_file_id,
            "validation_file_id": validation_file_id,
            "model_id": None
        }
        
        # Save job info
        job_info_file = self.output_dir / f"job_{job_id}.json"
        with open(job_info_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Job info saved to: {job_info_file}")
        
        # Wait for completion if requested
        if wait_for_completion:
            try:
                model_id = self.finetuner.wait_for_completion(job_id)
                result["model_id"] = model_id
                
                # Update job info file
                with open(job_info_file, 'w') as f:
                    json.dump(result, f, indent=2)
                
            except Exception as e:
                logger.error(f"Error waiting for completion: {e}")
        
        logger.info("="*60)
        logger.info("Pipeline Complete")
        logger.info("="*60)
        
        return result