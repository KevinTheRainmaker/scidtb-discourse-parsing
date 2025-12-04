"""
Few-shot discourse dependency parser.
"""
from typing import Optional, List
import json
from langchain.prompts import PromptTemplate

from .base import BaseParser
from ..models import DiscourseTreeModel
from ..utils.logger import get_logger

logger = get_logger(__name__)


class FewShotParser(BaseParser):
    """
    Few-shot parser for discourse dependency parsing.
    
    This parser uses a few example annotations to guide the model.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4-1106-preview",
        temperature: float = 0.0,
        max_retries: int = 3,
        n_shots: int = 3
    ):
        """
        Initialize few-shot parser.
        
        Args:
            api_key: OpenAI API key
            model: Model name
            temperature: Sampling temperature
            max_retries: Maximum retry attempts
            n_shots: Number of examples to use
        """
        super().__init__(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_retries=max_retries
        )
        
        self.n_shots = n_shots
        self.examples: List[tuple[str, DiscourseTreeModel]] = []
        
        logger.info(
            f"Initialized FewShotParser with model={model}, n_shots={n_shots}"
        )
    
    def add_examples(
        self,
        examples: List[tuple[str, DiscourseTreeModel]]
    ):
        """
        Add examples for few-shot learning.
        
        Args:
            examples: List of (text, tree) tuples
        """
        self.examples = examples[:self.n_shots]
        logger.info(f"Added {len(self.examples)} examples for few-shot learning")
    
    def add_example(self, text: str, tree: DiscourseTreeModel):
        """
        Add a single example.
        
        Args:
            text: Example input text
            tree: Example discourse tree
        """
        if len(self.examples) < self.n_shots:
            self.examples.append((text, tree))
            logger.debug(f"Added example {len(self.examples)}/{self.n_shots}")
    
    def format_examples(self) -> str:
        """
        Format examples for the prompt.
        
        Returns:
            Formatted examples string
        """
        if not self.examples:
            return ""
        
        formatted = []
        for i, (text, tree) in enumerate(self.examples, 1):
            tree_json = json.dumps(tree.to_dict(), indent=2, ensure_ascii=False)
            
            formatted.append(f"""Example {i}:
Input Text:
{text}

Output Structure:
{tree_json}

---""")
        
        return "\n\n".join(formatted)
    
    def create_prompt_template(self) -> PromptTemplate:
        """
        Create few-shot prompt template.
        
        Returns:
            PromptTemplate for few-shot parsing
        """
        
        examples_text = self.format_examples()
        
        # TODO: Fill in the prompt template content
        template = f"""You are an expert discourse structure analyzer for scientific abstracts.

Learn from the following examples and apply the same analysis pattern to new text.

{examples_text}

[TODO: Add additional instructions here]

Now analyze the following new text:

Input Text:
{{text}}

{{format_instructions}}

Output the discourse structure as valid JSON:"""

        return PromptTemplate(
            template=template,
            input_variables=["text"],
            partial_variables={
                "format_instructions": self.get_format_instructions()
            }
        )
    
    def parse(self, text: str) -> Optional[DiscourseTreeModel]:
        """
        Parse text using few-shot approach.
        
        Args:
            text: Input text (scientific abstract)
            
        Returns:
            DiscourseTreeModel or None if parsing failed
        """
        if not self.examples:
            logger.warning(
                "No examples added. Few-shot parser behaves like zero-shot. "
                "Call add_examples() first."
            )
        
        logger.debug(
            f"Parsing text (few-shot, {len(self.examples)} examples): {text[:100]}..."
        )
        
        return self.parse_with_retry(text)