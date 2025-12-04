"""
Zero-shot discourse dependency parser.
"""
from typing import Optional
from langchain.prompts import PromptTemplate

from .base import BaseParser
from ..models import DiscourseTreeModel
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ZeroShotParser(BaseParser):
    """
    Zero-shot parser for discourse dependency parsing.
    
    This parser uses no examples and relies solely on instructions
    and the model's pre-trained knowledge.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4-1106-preview",
        temperature: float = 0.0,
        max_retries: int = 3
    ):
        """
        Initialize zero-shot parser.
        
        Args:
            api_key: OpenAI API key
            model: Model name
            temperature: Sampling temperature
            max_retries: Maximum retry attempts
        """
        super().__init__(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_retries=max_retries
        )
        
        logger.info(f"Initialized ZeroShotParser with model={model}")
    
    def create_prompt_template(self) -> PromptTemplate:
        """
        Create zero-shot prompt template.
        
        Returns:
            PromptTemplate for zero-shot parsing
        """
        
        # TODO: Fill in the prompt template content
        template = """You are an expert discourse structure analyzer for scientific abstracts.

Your task is to parse the given scientific abstract into Elementary Discourse Units (EDUs) and identify discourse dependencies between them.

[TODO: Add detailed instructions here]

Input Text:
{text}

{format_instructions}

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
        Parse text using zero-shot approach.
        
        Args:
            text: Input text (scientific abstract)
            
        Returns:
            DiscourseTreeModel or None if parsing failed
        """
        logger.debug(f"Parsing text (zero-shot): {text[:100]}...")
        
        return self.parse_with_retry(text)