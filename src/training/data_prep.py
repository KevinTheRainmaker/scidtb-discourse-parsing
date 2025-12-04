"""
Data preparation for fine-tuning.
"""
import json
from pathlib import Path
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser

from ..models import DiscourseTreeModel
from ..data.loader import SciDTBLoader
from ..utils.logger import get_logger

logger = get_logger(__name__)

class FineTuningExample(BaseModel):
    """Single fine-tuning example in OpenAI format."""
    messages: List[Dict[str, str]] = Field(
        description="List of messages in the conversation"
    )

class FineTuningDataPreparator:
    """Prepare data for OpenAI fine-tuning."""
    
    def __init__(
        self,
        model_type: str = "gpt-3.5-turbo-1106",
        include_format_instructions: bool = True
    ):
        """
        Initialize preparator.
        
        Args:
            model_type: Target model type for fine-tuning
            include_format_instructions: Whether to include Pydantic format instructions
        """
        self.model_type = model_type
        self.include_format_instructions = include_format_instructions
        self.parser = PydanticOutputParser(pydantic_object=DiscourseTreeModel)
    
    def create_system_message(self) -> str:
        """Create system message for fine-tuning."""
        base_message = """You are an expert discourse structure analyzer for scientific abstracts.

Your task is to parse scientific abstracts into Elementary Discourse Units (EDUs) and identify discourse dependencies between them.

Output a valid JSON structure representing the discourse dependency tree."""
        
        if self.include_format_instructions:
            format_instructions = self.parser.get_format_instructions()
            base_message += f"\n\n{format_instructions}"
        
        return base_message
    
    def create_user_message(self, text: str) -> str:
        """Create user message for a given text."""
        return f"Parse the following scientific abstract:\n\n{text}"
    
    def create_assistant_message(self, tree: DiscourseTreeModel) -> str:
        """Create assistant message from a discourse tree."""
        return json.dumps(tree.to_dict(), ensure_ascii=False)
    
    def tree_to_example(
        self,
        tree: DiscourseTreeModel,
        text: str
    ) -> FineTuningExample:
        """
        Convert a discourse tree to a fine-tuning example.
        
        Args:
            tree: Discourse tree
            text: Input text
            
        Returns:
            FineTuningExample in OpenAI format
        """
        messages = [
            {
                "role": "system",
                "content": self.create_system_message()
            },
            {
                "role": "user",
                "content": self.create_user_message(text)
            },
            {
                "role": "assistant",
                "content": self.create_assistant_message(tree)
            }
        ]
        
        return FineTuningExample(messages=messages)
    
    def prepare_dataset(
        self,
        trees: List[DiscourseTreeModel],
        texts: List[str],
        output_file: Path,
        validate: bool = True
    ) -> Path:
        """
        Prepare fine-tuning dataset.
        
        Args:
            trees: List of discourse trees
            texts: List of corresponding texts
            output_file: Output JSONL file path
            validate: Whether to validate each example
            
        Returns:
            Path to the created file
        """
        if len(trees) != len(texts):
            raise ValueError(
                f"Number of trees ({len(trees)}) must match number of texts ({len(texts)})"
            )
        
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        examples = []
        failed = 0
        
        for i, (tree, text) in enumerate(zip(trees, texts)):
            try:
                example = self.tree_to_example(tree, text)
                
                if validate:
                    # Validate that assistant response is valid JSON and valid tree
                    assistant_content = example.messages[2]["content"]
                    parsed_json = json.loads(assistant_content)
                    DiscourseTreeModel.from_dict(parsed_json)
                
                examples.append(example)
                
            except Exception as e:
                logger.error(f"Failed to create example {i}: {e}")
                failed += 1
        
        # Write to JSONL
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(example.model_dump_json() + '\n')
        
        logger.info(
            f"Created fine-tuning dataset: {len(examples)} examples, "
            f"{failed} failed, saved to {output_file}"
        )
        
        return output_file
    
    def validate_dataset(self, dataset_file: Path) -> Dict[str, any]:
        """
        Validate a fine-tuning dataset.
        
        Args:
            dataset_file: Path to JSONL file
            
        Returns:
            Validation report
        """
        errors = []
        valid_count = 0
        
        with open(dataset_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines, 1):
            try:
                # Parse line as JSON
                data = json.loads(line)
                
                # Validate structure
                if "messages" not in data:
                    errors.append(f"Line {i}: Missing 'messages' field")
                    continue
                
                messages = data["messages"]
                
                # Check message roles
                expected_roles = ["system", "user", "assistant"]
                actual_roles = [msg.get("role") for msg in messages]
                
                if actual_roles != expected_roles:
                    errors.append(
                        f"Line {i}: Expected roles {expected_roles}, got {actual_roles}"
                    )
                    continue
                
                # Validate assistant response
                assistant_msg = messages[2]["content"]
                try:
                    parsed_json = json.loads(assistant_msg)
                    DiscourseTreeModel.from_dict(parsed_json)
                    valid_count += 1
                except Exception as e:
                    errors.append(f"Line {i}: Invalid tree structure - {e}")
                
            except json.JSONDecodeError as e:
                errors.append(f"Line {i}: Invalid JSON - {e}")
        
        report = {
            "total_examples": len(lines),
            "valid_examples": valid_count,
            "invalid_examples": len(errors),
            "errors": errors[:10]  # Show first 10 errors
        }
        
        if errors:
            logger.warning(f"Validation found {len(errors)} errors")
            for error in errors[:5]:
                logger.warning(f"  {error}")
        else:
            logger.info("✓ All examples are valid")
        
        return report
    
    @staticmethod
    def prepare_from_loader(
        loader: SciDTBLoader,
        split: str,
        output_file: Path,
        validate: bool = True
    ) -> Path:
        """
        Convenience method to prepare dataset from loader.
        
        Args:
            loader: SciDTBLoader instance
            split: Data split to load
            output_file: Output file path
            validate: Whether to validate examples
            
        Returns:
            Path to created file
        """
        trees = loader.load_split(split)
        texts = [SciDTBLoader.extract_text(tree) for tree in trees]
        
        preparator = FineTuningDataPreparator()
        return preparator.prepare_dataset(trees, texts, output_file, validate)