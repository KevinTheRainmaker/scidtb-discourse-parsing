"""
Base parser class for discourse dependency parsing.
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from pathlib import Path
import time

from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate

from ..models import DiscourseTreeModel
from ..utils.logger import get_logger

logger = get_logger(__name__)


class BaseParser(ABC):
    """
    Abstract base class for discourse dependency parsers.
    
    All parsers should inherit from this class and implement the parse() method.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4-1106-preview",
        temperature: float = 0.0,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize base parser.
        
        Args:
            api_key: OpenAI API key
            model: Model name (must support JSON mode)
            temperature: Sampling temperature (0.0 = deterministic)
            max_retries: Maximum number of retry attempts on failure
            retry_delay: Delay between retries in seconds
        """
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Initialize LLM with JSON mode
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            openai_api_key=api_key,
            model_kwargs={"response_format": {"type": "json_object"}}
        )
        
        # Initialize Pydantic parser
        self.pydantic_parser = PydanticOutputParser(
            pydantic_object=DiscourseTreeModel
        )
        
        # Statistics
        self.num_calls = 0
        self.num_successes = 0
        self.num_failures = 0
        self.total_tokens = 0
    
    @abstractmethod
    def create_prompt_template(self) -> PromptTemplate:
        """
        Create the prompt template for this parser.
        
        Must be implemented by subclasses.
        
        Returns:
            PromptTemplate instance
        """
        pass
    
    def get_format_instructions(self) -> str:
        """
        Get Pydantic format instructions.
        
        Returns:
            Format instructions string
        """
        return self.pydantic_parser.get_format_instructions()
    
    def parse_with_retry(
        self,
        text: str,
        **kwargs
    ) -> Optional[DiscourseTreeModel]:
        """
        Parse text with retry logic.
        
        Args:
            text: Input text to parse
            **kwargs: Additional arguments for parsing
            
        Returns:
            DiscourseTreeModel or None if all retries failed
        """
        for attempt in range(self.max_retries):
            try:
                result = self._parse_single(text, **kwargs)
                
                if result is not None:
                    self.num_successes += 1
                    return result
                
            except Exception as e:
                logger.warning(
                    f"Parse attempt {attempt + 1}/{self.max_retries} failed: {e}"
                )
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        
        # All retries failed
        self.num_failures += 1
        logger.error(f"All {self.max_retries} parse attempts failed for text: {text[:100]}...")
        return None
    
    def _parse_single(
        self,
        text: str,
        **kwargs
    ) -> Optional[DiscourseTreeModel]:
        """
        Single parse attempt (internal method).
        
        Args:
            text: Input text to parse
            **kwargs: Additional arguments
            
        Returns:
            DiscourseTreeModel or None
        """
        self.num_calls += 1
        
        # Create prompt
        prompt_template = self.create_prompt_template()
        prompt = prompt_template.format(text=text, **kwargs)
        
        # Call LLM
        response = self.llm.predict(prompt)
        
        # Parse response with Pydantic
        parsed = self.pydantic_parser.parse(response)
        
        return parsed
    
    @abstractmethod
    def parse(self, text: str) -> Optional[DiscourseTreeModel]:
        """
        Parse input text into discourse dependency tree.
        
        Must be implemented by subclasses.
        
        Args:
            text: Input text (scientific abstract)
            
        Returns:
            DiscourseTreeModel or None if parsing failed
        """
        pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get parser statistics.
        
        Returns:
            Dictionary with statistics
        """
        success_rate = (
            self.num_successes / self.num_calls * 100
            if self.num_calls > 0 else 0.0
        )
        
        return {
            "num_calls": self.num_calls,
            "num_successes": self.num_successes,
            "num_failures": self.num_failures,
            "success_rate": round(success_rate, 2),
            "model": self.model,
            "temperature": self.temperature
        }
    
    def reset_statistics(self):
        """Reset statistics counters."""
        self.num_calls = 0
        self.num_successes = 0
        self.num_failures = 0
        self.total_tokens = 0
    
    def print_statistics(self):
        """Print parser statistics."""
        stats = self.get_statistics()
        
        print(f"\n{'='*60}")
        print(f"{'Parser Statistics':^60}")
        print(f"{'='*60}")
        print(f"Model:           {stats['model']}")
        print(f"Temperature:     {stats['temperature']}")
        print(f"Total calls:     {stats['num_calls']}")
        print(f"Successes:       {stats['num_successes']}")
        print(f"Failures:        {stats['num_failures']}")
        print(f"Success rate:    {stats['success_rate']:.2f}%")
        print(f"{'='*60}\n")