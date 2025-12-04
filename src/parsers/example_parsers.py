"""
Example usage of discourse parsers.
"""
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parsers import ZeroShotParser, FewShotParser, FineTunedParser
from src.data.loader import SciDTBLoader
from config.settings import settings


def example_zero_shot():
    """Example: Zero-shot parsing"""
    print("\n" + "="*60)
    print("Zero-shot Parsing Example")
    print("="*60)
    
    # Initialize parser
    parser = ZeroShotParser(
        api_key=settings.openai_api_key,
        model="gpt-4-1106-preview"
    )
    
    # Example text
    text = """We propose a neural network approach to benefit from the non-linearity 
    of corpus-wide statistics for part-of-speech tagging. We investigated several 
    types of corpus-wide information for the words."""
    
    # Parse
    print(f"\nInput: {text[:100]}...")
    result = parser.parse(text)
    
    if result:
        print("\n✓ Parsing successful!")
        print(f"Number of EDUs: {len(result.edus)}")
        print("\nFirst 3 EDUs:")
        for edu in result.edus[:3]:
            print(f"  EDU {edu.id}: {edu.text[:50]}... (parent={edu.parent}, rel={edu.relation})")
    else:
        print("\n✗ Parsing failed")
    
    # Statistics
    parser.print_statistics()


def example_few_shot():
    """Example: Few-shot parsing"""
    print("\n" + "="*60)
    print("Few-shot Parsing Example")
    print("="*60)
    
    # Load training examples
    loader = SciDTBLoader(settings.train_dir.parent)
    train_trees = loader.load_split('train')
    
    if not train_trees:
        print("No training data found for few-shot examples")
        return
    
    # Initialize parser
    parser = FewShotParser(
        api_key=settings.openai_api_key,
        model="gpt-4-1106-preview",
        n_shots=3
    )
    
    # Add examples
    examples = []
    for tree in train_trees[:3]:
        text = SciDTBLoader.extract_text(tree)
        examples.append((text, tree))
    
    parser.add_examples(examples)
    print(f"\n✓ Added {len(parser.examples)} examples")
    
    # Parse new text
    text = """We propose a neural network approach to benefit from the non-linearity 
    of corpus-wide statistics for part-of-speech tagging."""
    
    print(f"\nInput: {text[:100]}...")
    result = parser.parse(text)
    
    if result:
        print("\n✓ Parsing successful!")
        print(f"Number of EDUs: {len(result.edus)}")
    else:
        print("\n✗ Parsing failed")
    
    # Statistics
    parser.print_statistics()


def example_finetuned():
    """Example: Fine-tuned model parsing"""
    print("\n" + "="*60)
    print("Fine-tuned Model Parsing Example")
    print("="*60)
    
    # Replace with your actual fine-tuned model ID
    model_id = "ft:gpt-3.5-turbo-1106:..."
    
    print(f"\nNote: Replace model_id with your actual fine-tuned model ID")
    print(f"Current model_id: {model_id}")
    
    if "..." in model_id:
        print("\n⚠ Skipping fine-tuned example (no model ID provided)")
        return
    
    # Initialize parser
    parser = FineTunedParser(
        api_key=settings.openai_api_key,
        model_id=model_id
    )
    
    # Parse
    text = """We propose a neural network approach to benefit from the non-linearity 
    of corpus-wide statistics for part-of-speech tagging."""
    
    print(f"\nInput: {text[:100]}...")
    result = parser.parse(text)
    
    if result:
        print("\n✓ Parsing successful!")
        print(f"Number of EDUs: {len(result.edus)}")
    else:
        print("\n✗ Parsing failed")
    
    # Statistics
    parser.print_statistics()


def main():
    """Run all examples"""
    
    if not settings.validate_api_key():
        print("Error: OpenAI API key not found. Set OPENAI_API_KEY in .env")
        return 1
    
    # Run examples
    try:
        example_zero_shot()
    except Exception as e:
        print(f"\nZero-shot example failed: {e}")
    
    try:
        example_few_shot()
    except Exception as e:
        print(f"\nFew-shot example failed: {e}")
    
    try:
        example_finetuned()
    except Exception as e:
        print(f"\nFine-tuned example failed: {e}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())