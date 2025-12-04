# Parsers Usage Guide

## Overview

The `src/parsers` module provides three types of discourse dependency parsers:

1. **ZeroShotParser** - Uses model knowledge without examples
2. **FewShotParser** - Uses a few annotated examples for guidance
3. **FineTunedParser** - Uses a model fine-tuned on SciDTB data

All parsers:
- Inherit from `BaseParser`
- Use **LangChain PromptTemplate** for structured prompting
- Use **Pydantic OutputParser** for guaranteed JSON structure
- Support automatic retry on failures
- Track statistics (success rate, number of calls, etc.)

---

## Parser Classes

### 1. BaseParser (Abstract)

Base class that all parsers inherit from.

**Key Features:**
- Automatic retry logic with exponential backoff
- Pydantic-based output validation
- Statistics tracking
- LangChain integration

**Methods:**
- `create_prompt_template()` - Must be implemented by subclasses
- `parse(text)` - Must be implemented by subclasses
- `parse_with_retry(text)` - Handles retries automatically
- `get_statistics()` - Returns parsing statistics
- `print_statistics()` - Prints statistics to console

---

### 2. ZeroShotParser

Parser that uses no training examples.

**Usage:**

```python
from src.parsers import ZeroShotParser

# Initialize
parser = ZeroShotParser(
    api_key="your-api-key",
    model="gpt-4-1106-preview",
    temperature=0.0,
    max_retries=3
)

# Parse text
text = "We propose a neural network approach..."
result = parser.parse(text)

if result:
    print(f"Parsed {len(result.edus)} EDUs")
    for edu in result.edus:
        print(f"EDU {edu.id}: parent={edu.parent}, relation={edu.relation}")
```

**Prompt Template:**

The prompt template is defined in `create_prompt_template()`. Current template has `[TODO]` markers where you should add:
- Detailed task instructions
- EDU segmentation guidelines
- Discourse relation definitions
- Output format requirements

**Example Prompt Structure:**

```
System: You are an expert discourse analyzer...

Instructions:
[TODO: Add instructions]
1. Segment text into EDUs
2. Identify parent for each EDU
3. Classify relation types

Input Text:
{text}

Format Instructions:
{format_instructions}

Output:
```

---

### 3. FewShotParser

Parser that learns from examples.

**Usage:**

```python
from src.parsers import FewShotParser
from src.data.loader import SciDTBLoader

# Initialize
parser = FewShotParser(
    api_key="your-api-key",
    model="gpt-4-1106-preview",
    n_shots=3
)

# Load examples
loader = SciDTBLoader("./data/raw")
train_trees = loader.load_split('train')

# Add examples
examples = []
for tree in train_trees[:3]:
    text = SciDTBLoader.extract_text(tree)
    examples.append((text, tree))

parser.add_examples(examples)

# Parse new text
result = parser.parse("Your text here...")
```

**Methods:**

- `add_examples(examples)` - Add list of (text, tree) tuples
- `add_example(text, tree)` - Add single example
- `format_examples()` - Format examples for prompt

**Prompt Template:**

Examples are automatically formatted and inserted into the prompt:

```
You are an expert discourse analyzer...

Example 1:
Input: [text]
Output: [tree structure]

Example 2:
Input: [text]
Output: [tree structure]

[TODO: Add additional instructions]

Now analyze this text:
{text}
```

---

### 4. FineTunedParser

Parser using a fine-tuned model.

**Usage:**

```python
from src.parsers import FineTunedParser

# Initialize with fine-tuned model ID
parser = FineTunedParser(
    api_key="your-api-key",
    model_id="ft:gpt-3.5-turbo-1106:org:model:id"
)

# Parse (simpler prompts for fine-tuned models)
result = parser.parse("Your text here...")
```

**Key Differences:**

- Uses simpler prompts (model already trained on format)
- Uses system/user message format matching training
- Requires fine-tuned model ID

**Prompt Template:**

```
System: You are an expert discourse structure analyzer.

User: Parse the following scientific abstract:
{text}
```

---

## Output Format

All parsers return a `DiscourseTreeModel` with Pydantic validation:

```python
{
    "edus": [
        {
            "id": 0,
            "text": "ROOT",
            "parent": -1,
            "relation": "null"
        },
        {
            "id": 1,
            "text": "We propose a neural network approach",
            "parent": 0,
            "relation": "ROOT"
        },
        {
            "id": 2,
            "text": "to benefit from non-linearity",
            "parent": 1,
            "relation": "enablement"
        }
    ]
}
```

**Validation Rules:**

✓ Exactly one ROOT (id=0, parent=-1)
✓ Consecutive IDs starting from 0
✓ parent < current_id (no forward references)
✓ No circular dependencies
✓ Valid relation types

---

## Statistics Tracking

All parsers track usage statistics:

```python
# Get statistics
stats = parser.get_statistics()
print(stats)
# {
#     "num_calls": 10,
#     "num_successes": 9,
#     "num_failures": 1,
#     "success_rate": 90.0,
#     "model": "gpt-4-1106-preview",
#     "temperature": 0.0
# }

# Print formatted statistics
parser.print_statistics()

# Reset counters
parser.reset_statistics()
```

---

## Prompt Engineering Tips

### TODO Items in Templates

Each parser has `[TODO]` markers in the prompt template where you should add:

**ZeroShotParser:**
- Task description
- EDU segmentation rules
- Relation type definitions with examples
- Tree structure requirements

**FewShotParser:**
- Additional guidance beyond examples
- Pattern recognition instructions
- Edge case handling

**FineTunedParser:**
- Minimal additional instructions
- Format reminders (if needed)

### Recommended Prompt Structure

```
1. Role Definition
   "You are an expert discourse structure analyzer..."

2. Task Description
   "Your task is to parse scientific abstracts into EDUs..."

3. Instructions (Numbered)
   "1. Segment into clauses (EDUs)
    2. Identify parent for each EDU
    3. Classify relation type"

4. Relation Types (with brief examples)
   "- ROOT: Main discourse unit
    - Elaboration: Provides additional detail
    - Cause: Expresses causation"

5. Constraints
   "- EDU IDs must be consecutive
    - parent < current_id
    - Exactly one ROOT"

6. Examples (for few-shot)
   [Auto-generated from add_examples()]

7. Input
   "Input Text: {text}"

8. Format Instructions
   {format_instructions}  # Pydantic schema

9. Output Instruction
   "Output the discourse structure as valid JSON:"
```

---

## Error Handling

All parsers handle errors gracefully:

```python
result = parser.parse(text)

if result is None:
    # Parsing failed after all retries
    print("Failed to parse text")
    stats = parser.get_statistics()
    print(f"Success rate: {stats['success_rate']}%")
else:
    # Successful parse
    print(f"Parsed {len(result.edus)} EDUs")
```

**Common Errors:**

- Invalid JSON from model → Automatic retry
- Pydantic validation failure → Automatic retry
- API timeout → Automatic retry with delay
- All retries exhausted → Returns None

---

## Best Practices

### 1. Model Selection

- **Zero-shot**: Use GPT-4 for better instruction following
- **Few-shot**: GPT-4 or GPT-3.5-turbo (3-5 examples)
- **Fine-tuned**: GPT-3.5-turbo (better cost/performance after training)

### 2. Temperature Settings

- Use `temperature=0.0` for consistent, deterministic outputs
- Higher temperature (0.3-0.7) for more creative parsing (not recommended)

### 3. Retry Configuration

```python
parser = ZeroShotParser(
    api_key=api_key,
    max_retries=3,        # 3 attempts total
    retry_delay=1.0       # 1 second between retries
)
```

### 4. Few-shot Example Selection

- Choose diverse examples (different structures, lengths)
- Include edge cases (complex nested structures)
- 3-5 examples typically optimal (more doesn't always help)

### 5. Batch Processing

```python
from tqdm import tqdm

texts = [...]  # Your texts
results = []

for text in tqdm(texts):
    result = parser.parse(text)
    if result:
        results.append(result)

# Check success rate
parser.print_statistics()
```

---

## Complete Example

```python
#!/usr/bin/env python3
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parsers import ZeroShotParser, FewShotParser
from src.data.loader import SciDTBLoader
from src.evaluation.metrics import DiscourseEvaluator
from config.settings import settings

def main():
    # 1. Load test data
    loader = SciDTBLoader(settings.data_dir)
    test_trees = loader.load_split('test/gold')[:10]  # First 10
    
    # 2. Initialize parser
    parser = ZeroShotParser(
        api_key=settings.openai_api_key,
        model="gpt-4-1106-preview"
    )
    
    # 3. Parse all texts
    predictions = []
    for gold_tree in test_trees:
        text = SciDTBLoader.extract_text(gold_tree)
        pred_tree = parser.parse(text)
        if pred_tree:
            predictions.append(pred_tree)
    
    # 4. Evaluate
    evaluator = DiscourseEvaluator()
    metrics = evaluator.evaluate_batch(
        gold_trees=test_trees[:len(predictions)],
        pred_trees=predictions
    )
    
    # 5. Display results
    metrics.print_summary()
    parser.print_statistics()

if __name__ == "__main__":
    main()
```

---

## Next Steps

1. **Fill in TODO prompts** in each parser class
2. **Test with sample data** using `example_parsers.py`
3. **Tune prompt templates** based on initial results
4. **Run evaluation** to compare parser performance
5. **Fine-tune model** for best results

See `experiments/` directory for full evaluation scripts.