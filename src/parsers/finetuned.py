"""
Fine-tuned discourse dependency parser.
"""
from typing import Optional
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage

from .base import BaseParser
from ..models import DiscourseTreeModel
from ..utils.logger import get_logger

logger = get_logger(__name__)


class FineTunedParser(BaseParser):
    """
    Parser using a fine-tuned model.
    
    This parser uses a model that has been fine-tuned on SciDTB data,
    so it requires minimal prompting.
    """
    
    def __init__(
        self,
        api_key: str,
        model_id: str,
        temperature: float = 0.0,
        max_retries: int = 3
    ):
        """
        Initialize fine-tuned parser.
        
        Args:
            api_key: OpenAI API key
            model_id: Fine-tuned model ID (e.g., "ft:gpt-3.5-turbo:...")
            temperature: Sampling temperature
            max_retries: Maximum retry attempts
        """
        super().__init__(
            api_key=api_key,
            model=model_id,
            temperature=temperature,
            max_retries=max_retries
        )
        
        self.model_id = model_id
        
        logger.info(f"Initialized FineTunedParser with model_id={model_id}")
    
    def create_prompt_template(self) -> PromptTemplate:
        """
        Create prompt template for fine-tuned model.
        
        Fine-tuned models need minimal prompting since they were
        trained with specific format instructions.
        
        Returns:
            PromptTemplate for fine-tuned parsing
        """
        
        # TODO: Fill in the prompt template content
        # Fine-tuned models typically need simpler prompts
        template = """Parse the following scientific abstract:

{text}"""

        return PromptTemplate(
            template=template,
            input_variables=["text"]
        )
    
    def _parse_single(
        self,
        text: str,
        **kwargs
    ) -> Optional[DiscourseTreeModel]:
        """
        Override parsing for fine-tuned model.
        
        Fine-tuned models may benefit from using the messages API directly
        with system/user messages instead of a single prompt.
        
        Args:
            text: Input text to parse
            **kwargs: Additional arguments
            
        Returns:
            DiscourseTreeModel or None
        """
        self.num_calls += 1
        
        # Use system message (matching fine-tuning format)
        system_message = "You are an expert discourse structure analyzer for scientific abstracts."
        
        # Create user message
        user_message = f"Parse the following scientific abstract:\n\n{text}"
        
        # Call LLM with messages
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=user_message)
        ]
        
        response = self.llm.predict_messages(messages)
        
        # Parse response with Pydantic
        parsed = self.pydantic_parser.parse(response.content)
        
        return parsed
    
    def parse(self, text: str) -> Optional[DiscourseTreeModel]:
        """
        Parse text using fine-tuned model.
        
        Args:
            text: Input text (scientific abstract)
            
        Returns:
            DiscourseTreeModel or None if parsing failed
        """
        logger.debug(f"Parsing text (fine-tuned): {text[:100]}...")
        
        return self.parse_with_retry(text)
    
    def get_statistics(self) -> dict:
        """
        Get parser statistics including model ID.
        
        Returns:
            Dictionary with statistics
        """
        stats = super().get_statistics()
        stats["model_id"] = self.model_id
        return stats