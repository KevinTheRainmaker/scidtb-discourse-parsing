# SciDTB Scripts Usage Guide

## Overview

This directory contains three main scripts for working with SciDTB data:

1. **prepare_data.py** - Data loading, validation, and preprocessing
2. **train.py** - Fine-tuning preparation and execution
3. **evaluate.py** - Model evaluation and comparison

## Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

## Data Structure

Your data directory should be organized as follows:

```
data/raw/
├── train/
│   ├── D14-1101.edu.txt.dep
│   ├── D14-1102.edu.txt.dep
│   └── ...
└── test/
    ├── gold/
    │   ├── D14-1001.edu.txt.dep
    │   ├── D14-1002.edu.txt.dep
    │   └── ...
    ├── second_annotate/
    │   └── ...
    └── edu/
        └── ...
```

---

## 1. prepare_data.py

### Purpose
- Load and validate SciDTB data
- Compute dataset statistics
- Save processed data for training/evaluation

### Basic Usage

```bash
# Validate data and show statistics
python scripts/prepare_data.py --data-dir ./data/raw

# Show statistics only (no saving)
python scripts/prepare_data.py --data-dir ./data/raw --stats-only

# Save processed data
python scripts/prepare_data.py \
    --data-dir ./data/raw \
    --output-dir ./data/processed \
    --save-processed

# Filter by minimum EDUs
python scripts/prepare_data.py \
    --data-dir ./data/raw \
    --min-edus 5 \
    --save-processed
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--data-dir` | `./data/raw` | Root directory with SciDTB data |
| `--output-dir` | `./data/processed` | Output directory for processed data |
| `--min-edus` | `3` | Minimum EDUs per tree |
| `--save-processed` | `False` | Save processed trees to JSON |
| `--stats-only` | `False` | Only compute statistics |
| `--log-level` | `INFO` | Logging level |

### Output

```
data/processed/
├── train.json           # Processed training data
├── test.json           # Processed test data
└── statistics.json     # Dataset statistics
```

### Example Output

```
==============================================================
                      TRAIN Statistics                       
==============================================================
Number of trees:             492
Total EDUs:                 5234
Average EDUs per tree:     10.64
Maximum tree depth:           8
Average tree depth:        2.34
Unique relation types:       26
==============================================================

Top 10 Relation Types:
--------------------------------------------------------------
           Elaboration-Addition:    987 (18.86%)
                      ROOT:        492 ( 9.40%)
              elab-addition:        456 ( 8.71%)
                Background:        398 ( 7.61%)
                     Cause:        312 ( 5.96%)
...
```

---

## 2. train.py

### Purpose
- Prepare fine-tuning dataset in OpenAI format
- Upload data and create fine-tuning job
- Monitor training progress

### Basic Usage

#### Step 1: Prepare Data Only

```bash
python scripts/train.py \
    --data-dir ./data/raw \
    --output-dir ./data/processed \
    --prepare-only
```

#### Step 2: Upload and Start Training

```bash
python scripts/train.py \
    --data-dir ./data/raw \
    --output-dir ./data/processed \
    --upload-and-train \
    --suffix scidtb-v1
```

#### Combined: Prepare and Train

```bash
python scripts/train.py \
    --data-dir ./data/raw \
    --output-dir ./data/processed \
    --upload-and-train \
    --model gpt-3.5-turbo-1106 \
    --n-epochs 3 \
    --suffix scidtb-v1
```

#### Wait for Completion

```bash
python scripts/train.py \
    --data-dir ./data/raw \
    --output-dir ./data/processed \
    --upload-and-train \
    --wait \
    --suffix scidtb-v1
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--data-dir` | `./data/raw` | Root data directory |
| `--output-dir` | `./data/processed` | Output directory |
| `--train-split` | `train` | Training split name |
| `--val-split` | `None` | Validation split (optional) |
| `--model` | `gpt-3.5-turbo-1106` | Base model to fine-tune |
| `--n-epochs` | `3` | Number of training epochs |
| `--suffix` | `scidtb` | Model name suffix |
| `--api-key` | from `.env` | OpenAI API key |
| `--prepare-only` | `False` | Only prepare data |
| `--upload-and-train` | `False` | Upload and start training |
| `--wait` | `False` | Wait for completion |
| `--no-validate` | `False` | Skip validation |

### Output

```
data/processed/
├── finetune_train_20240101_120000.jsonl  # Training data
├── finetune_val_20240101_120000.jsonl    # Validation data (if specified)
└── job_ftjob-xxxxx.json                  # Job information
```

### Example Output

```
==============================================================
              Starting Fine-tuning Job
==============================================================
Base model:      gpt-3.5-turbo-1106
Training file:   ./data/processed/finetune_train_20240101.jsonl
Validation file: None
Epochs:          3
Suffix:          scidtb-v1
==============================================================

✓ File uploaded successfully: file-xxxxx
✓ Fine-tuning job created: ftjob-xxxxx
  Monitor at: https://platform.openai.com/finetune/ftjob-xxxxx

==============================================================
                   Fine-tuning Job Created
==============================================================
Job ID:          ftjob-xxxxx
Status:          In progress

Monitor at: https://platform.openai.com/finetune/ftjob-xxxxx
==============================================================
```

---

## 3. evaluate.py

### Purpose
- Evaluate parsing predictions against gold standard
- Compute UAS, LAS, F1 metrics
- Generate detailed reports
- Compare multiple models

### Basic Usage

#### Single Model Evaluation

```bash
python scripts/evaluate.py \
    --data-dir ./data/raw \
    --test-split test/gold \
    --predictions ./results/predictions.json \
    --output-dir ./data/results \
    --save-detailed
```

#### With Alignments

```bash
python scripts/evaluate.py \
    --data-dir ./data/raw \
    --test-split test/gold \
    --predictions ./results/predictions.json \
    --save-detailed \
    --save-alignments
```

#### Compare Multiple Models

```bash
python scripts/evaluate.py \
    --data-dir ./data/raw \
    --test-split test/gold \
    --compare \
    --prediction-files \
        ./results/zero_shot.json \
        ./results/few_shot.json \
        ./results/finetuned.json \
    --model-names "Zero-shot" "Few-shot" "Fine-tuned" \
    --output-dir ./data/results
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--data-dir` | `./data/raw` | Root data directory |
| `--test-split` | `test/gold` | Test split name |
| `--predictions` | Required | Path to predictions file |
| `--output-dir` | `./data/results` | Output directory |
| `--max-samples` | `None` | Max samples to evaluate |
| `--save-detailed` | `False` | Save detailed results |
| `--save-alignments` | `False` | Save EDU alignments |
| `--no-relations` | `False` | Skip relation evaluation |
| `--compare` | `False` | Compare multiple models |
| `--prediction-files` | `None` | Multiple prediction files |
| `--model-names` | `None` | Model names for comparison |

### Predictions Format

Your predictions file should be a JSON array of discourse trees:

```json
[
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
      ...
    ]
  },
  ...
]
```

### Output

```
data/results/
├── evaluation_results_20240101_120000.json  # Detailed results
└── model_comparison_20240101_120000.json    # Comparison results
```

### Example Output

```
==============================================================
                    Evaluation Results
==============================================================
Total EDUs:                152
Matched EDUs:              152
--------------------------------------------------------------
UAS (Unlabeled):         78.29% (119/152)
LAS (Labeled):           65.13% (99/152)
F1 Score:                67.89%
--------------------------------------------------------------
Precision:               65.13%
Recall:                  65.13%
==============================================================

==============================================================
              Detailed EDU Alignments
==============================================================
 EDU | Gold       | Pred       | Gold Rel        | Pred Rel        | UAS | LAS
--------------------------------------------------------------------------------
   1 |          0 |          0 | ROOT            | ROOT            |  ✔  |  ✔
   2 |          1 |          1 | enablement      | enablement      |  ✔  |  ✔
   3 |          1 |          1 | elab-aspect     | elaboration     |  ✔  |  ✘
   4 |          5 |          5 | cause           | cause           |  ✔  |  ✔
   5 |          3 |          3 | elab-addition   | elab-addition   |  ✔  |  ✔
...
==============================================================

==============================================================
           Relation Classification Report
==============================================================
                 Relation |  Precision |     Recall |         F1 |  Support
--------------------------------------------------------------------------------
                     ROOT |      95.00% |      95.00% |      95.00% |       20
        Elaboration-Addition |      72.34% |      68.00% |      70.10% |       50
               Background |      65.22% |      60.00% |      62.50% |       30
...
--------------------------------------------------------------------------------
          Weighted Average |             |             |      68.45% |      152
==============================================================
```

---

## Complete Workflow Example

```bash
# 1. Prepare and validate data
python scripts/prepare_data.py \
    --data-dir ./data/raw \
    --output-dir ./data/processed \
    --save-processed

# 2. Prepare fine-tuning data
python scripts/train.py \
    --data-dir ./data/raw \
    --output-dir ./data/processed \
    --prepare-only

# 3. Start fine-tuning
python scripts/train.py \
    --data-dir ./data/raw \
    --output-dir ./data/processed \
    --upload-and-train \
    --suffix scidtb-v1 \
    --wait

# 4. Generate predictions (using your parser)
# ... run your parser to create predictions.json ...

# 5. Evaluate predictions
python scripts/evaluate.py \
    --data-dir ./data/raw \
    --test-split test/gold \
    --predictions ./results/predictions.json \
    --output-dir ./data/results \
    --save-detailed \
    --save-alignments
```

---

## Troubleshooting

### Error: "Data directory does not exist"
**Solution**: Check that your `--data-dir` points to the correct location with train/ and test/ subdirectories.

### Error: "No .edu.txt.dep files found"
**Solution**: Ensure your data files have the `.edu.txt.dep` extension and are in the correct directories.

### Error: "OpenAI API key not found"
**Solution**: Set `OPENAI_API_KEY` in your `.env` file or use `--api-key` argument.

### Error: "Invalid tree structure"
**Solution**: Check that your prediction files match the expected JSON format with valid discourse trees.

### Warning: "Number of gold trees != predicted trees"
**Solution**: The evaluator will use the minimum length. Ensure your predictions cover all test samples.

---

## Tips

1. **Always validate first**: Run `prepare_data.py` before training to catch data issues early.

2. **Use validation split**: If you have a dev set, use `--val-split dev` for better training monitoring.

3. **Save alignments for debugging**: Use `--save-alignments` to understand where your model makes mistakes.

4. **Compare multiple approaches**: Use `--compare` mode to evaluate zero-shot, few-shot, and fine-tuned models together.

5. **Monitor training**: Use the OpenAI dashboard URL provided to monitor training progress and metrics.