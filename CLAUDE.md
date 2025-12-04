# SciDTB Discourse Parsing with LLMs

## What This Does

Parses scientific abstracts into discourse dependency trees using GPT-4 or fine-tuned GPT-3.5-turbo. Takes text and produces a tree showing which sentences depend on which, with labeled relations (Addition, Elaboration, Cause-effect, etc.).

## Why This Architecture

**Problem:** Scientific text has hierarchical discourse structure. Traditional RST constituency parsing with LLMs achieves only ~60% accuracy.

**Solution:** Dependency trees (SciDTB format):
- Simpler: Each clause has exactly one parent
- More accurate: ~85% with LLMs  
- Flexible: Supports non-projective structures

**Three approaches:**
1. Zero-shot: Instructions only
2. Few-shot: Add 3-5 examples
3. Fine-tuned: Train GPT-3.5-turbo (best results)

## Structure

```
src/models/          Pydantic models with validation
src/data/loader.py   Loads .edu.txt.dep files (UTF-8 BOM handling)
src/parsers/         Three parser types
src/evaluation/      UAS, LAS, F1 metrics
src/training/        OpenAI fine-tuning

experiments/         Automated pipelines
scripts/             CLI tools
docs/                Detailed guides
```

## Data Format

Input files (`.edu.txt.dep`):
```json
{"root": [
  {"id": 0, "text": "ROOT", "parent": -1, "relation": "null"},
  {"id": 1, "text": "Sentence", "parent": 0, "relation": "Background"}
]}
```

**Critical:** Files have UTF-8 BOM. Use `encoding='utf-8-sig'`.

**Validation** (enforced by Pydantic):
- Exactly one ROOT (id=0, parent=-1)
- Consecutive IDs starting from 0
- parent < id (no forward references)
- Valid relation type (26 options)

## How To Use

**Load data:**
```python
from src.data.loader import SciDTBLoader

loader = SciDTBLoader("data/raw")
trees = loader.load_split("train")  # or "test/gold"
```

**Run experiments:**
```bash
python experiments/run_zero_shot.py --max-samples 5    # Quick test
python experiments/run_few_shot.py --n-shots 3         # With examples
python experiments/run_finetuned.py --model-id ft:...  # Fine-tuned
```

**Parse custom text:**
```python
from src.parsers import ZeroShotParser

parser = ZeroShotParser(api_key="...")
tree = parser.parse("Your scientific abstract here")
```

## Key Details

**All parsers:**
- Auto-retry (max_retries=3)
- Pydantic validation (guaranteed valid output)
- Return `Optional[DiscourseTreeModel]`

**Metrics:**
- UAS: Correct parent (ignores label)
- LAS: Correct parent AND relation
- F1: Harmonic mean
All as percentages.

**Prompt templates** have `[TODO]` markers for:
- Task description
- EDU segmentation rules
- Relation definitions
Fill these for better accuracy.

**JSON mode:** All parsers use `model_kwargs={"response_format": {"type": "json_object"}}`.

**Relation distribution:**
- Top 3: Addition (17%), Elaboration (15%), Enablement (11%)
- 26 types total, long tail

## Common Tasks

**Quick test:**
```bash
python experiments/run_zero_shot.py --max-samples 5
```

**Full evaluation:**
```bash
python experiments/run_zero_shot.py
```

**Fine-tune:**
```bash
python scripts/train.py --upload-and-train --wait
```

**Add parser:**
1. Inherit from `BaseParser` in `src/parsers/base.py`
2. Implement `create_prompt_template()` and `parse()`
3. Use `PydanticOutputParser` for validation

## Need More Detail

Read docs/ for specific tasks:
- `DATA_README.md`: Data format
- `PARSERS_USAGE.md`: Parser customization
- `EXPERIMENTS_GUIDE.md`: Full workflows

## Environment

Python 3.8+. Key dependencies:
- `pydantic>=2.0`: Validation
- `langchain>=0.1`: Prompts
- `openai>=1.0`: API

Set `OPENAI_API_KEY` in `.env` file.
