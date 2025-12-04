# Experiments Usage Guide

## Overview

The `experiments/` directory contains scripts for running complete discourse parsing experiments:

1. **run_zero_shot.py** - Zero-shot parsing experiment
2. **run_few_shot.py** - Few-shot parsing experiment (with examples)
3. **run_finetuned.py** - Fine-tuned model parsing experiment

All experiments:
- Load test data automatically
- Parse all samples with progress tracking
- Compute evaluation metrics (UAS, LAS, F1)
- Save predictions and results
- Generate detailed reports

---

## Quick Start

### 1. Zero-shot Experiment

```bash
# Basic usage (5 samples for testing)
python experiments/run_zero_shot.py \
    --data-dir ./data/raw \
    --max-samples 5

# Full test set
python experiments/run_zero_shot.py \
    --data-dir ./data/raw \
    --output-dir ./data/results/zero_shot

# With different model
python experiments/run_zero_shot.py \
    --data-dir ./data/raw \
    --model gpt-4-turbo-preview
```

### 2. Few-shot Experiment

```bash
# 3-shot (default)
python experiments/run_few_shot.py \
    --data-dir ./data/raw \
    --n-shots 3 \
    --max-samples 5

# 5-shot
python experiments/run_few_shot.py \
    --data-dir ./data/raw \
    --n-shots 5

# With custom train split
python experiments/run_few_shot.py \
    --data-dir ./data/raw \
    --train-split train \
    --n-shots 3
```

### 3. Fine-tuned Experiment

```bash
# Basic usage
python experiments/run_finetuned.py \
    --data-dir ./data/raw \
    --model-id ft:gpt-3.5-turbo-1106:org:model:id \
    --max-samples 5

# Full test set
python experiments/run_finetuned.py \
    --data-dir ./data/raw \
    --model-id ft:gpt-3.5-turbo-1106:org:model:id \
    --output-dir ./data/results/finetuned

# Compare with baselines
python experiments/run_finetuned.py \
    --data-dir ./data/raw \
    --model-id ft:gpt-3.5-turbo-1106:org:model:id \
    --compare \
    --baseline-files \
        ./data/results/zero_shot/zero_shot_results_*.json \
        ./data/results/few_shot/few_shot_3shot_results_*.json
```

---

## Command-Line Options

### Common Options (All Experiments)

| Option | Default | Description |
|--------|---------|-------------|
| `--data-dir` | `./data/raw` | Root data directory |
| `--output-dir` | (varies) | Output directory for results |
| `--test-split` | `test/gold` | Test split to evaluate on |
| `--max-samples` | `None` | Limit number of test samples |
| `--temperature` | `0.0` | Sampling temperature |
| `--api-key` | from `.env` | OpenAI API key |
| `--no-save-predictions` | `False` | Skip saving predictions |
| `--log-level` | `INFO` | Logging level |

### Zero-shot Specific

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `gpt-4-1106-preview` | Model to use |

### Few-shot Specific

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `gpt-4-1106-preview` | Model to use |
| `--n-shots` | `3` | Number of examples |
| `--train-split` | `train` | Split for examples |
| `--seed` | `42` | Random seed for examples |

### Fine-tuned Specific

| Option | Default | Description |
|--------|---------|-------------|
| `--model-id` | **Required** | Fine-tuned model ID |
| `--compare` | `False` | Compare with baselines |
| `--baseline-files` | `None` | Baseline result files |

---

## Output Structure

Each experiment creates the following outputs:

```
data/results/{experiment_name}/
├── {experiment}_predictions_{timestamp}.json   # Predicted trees
└── {experiment}_results_{timestamp}.json       # Evaluation results
```

### Predictions File Format

```json
[
  {
    "edus": [
      {"id": 0, "text": "ROOT", "parent": -1, "relation": "null"},
      {"id": 1, "text": "...", "parent": 0, "relation": "ROOT"},
      ...
    ]
  },
  null,  // Failed parse
  ...
]
```

### Results File Format

```json
{
  "experiment": "zero_shot",
  "model": "gpt-4-1106-preview",
  "temperature": 0.0,
  "test_split": "test/gold",
  "timestamp": "20240101_120000",
  "summary": {
    "total_edus": 152,
    "matched_edus": 152,
    "uas_correct": 119,
    "las_correct": 99,
    "uas": 78.29,
    "las": 65.13,
    "f1": 67.89,
    "precision": 65.13,
    "recall": 65.13
  },
  "relation_metrics": {
    "ROOT": {"precision": 95.0, "recall": 95.0, "f1": 95.0, "support": 20},
    ...
  },
  "parser_stats": {
    "num_calls": 10,
    "num_successes": 9,
    "num_failures": 1,
    "success_rate": 90.0
  }
}
```

---

## Example Output

### Console Output

```
======================================================================
              Zero-shot Discourse Parsing Experiment
======================================================================
Model: gpt-4-1106-preview
Temperature: 0.0
Test split: test/gold
Max samples: 5
======================================================================

[1] Loading test data...
✓ Loaded 152 gold trees
Limited to 5 samples

[2] Initializing zero-shot parser...
✓ Parser initialized

[3] Parsing test set...
Parsing: 100%|████████████████████| 5/5 [00:25<00:00,  5.12s/it]
✓ Parsing complete: 5/5 successful

============================================================
                    Parser Statistics
============================================================
Model:           gpt-4-1106-preview
Temperature:     0.0
Total calls:     5
Successes:       5
Failures:        0
Success rate:    100.00%
============================================================

[4] Evaluating predictions...

============================================================
                    Zero-shot Results
============================================================
Total EDUs:                 52
Matched EDUs:               52
------------------------------------------------------------
UAS (Unlabeled):         80.77% (42/52)
LAS (Labeled):           67.31% (35/52)
F1 Score:                69.23%
------------------------------------------------------------
Precision:               67.31%
Recall:                  67.31%
============================================================

[5] Saving results...
✓ Predictions saved to ./data/results/zero_shot/zero_shot_predictions_20240101_120000.json
✓ Results saved to ./data/results/zero_shot/zero_shot_results_20240101_120000.json

======================================================================
                Zero-shot Experiment Complete
======================================================================
Total samples:       5
Successful parses:   5
Failed parses:       0
UAS:                 80.77%
LAS:                 67.31%
F1:                  69.23%
======================================================================
```

---

## Complete Workflow

### Step 1: Prepare Data

```bash
python scripts/prepare_data.py \
    --data-dir ./data/raw \
    --save-processed
```

### Step 2: Run Zero-shot (Baseline)

```bash
python experiments/run_zero_shot.py \
    --data-dir ./data/raw \
    --output-dir ./data/results/zero_shot \
    --max-samples 10
```

### Step 3: Run Few-shot

```bash
# Try different n-shots
for n in 1 3 5; do
    python experiments/run_few_shot.py \
        --data-dir ./data/raw \
        --output-dir ./data/results/few_shot \
        --n-shots $n \
        --max-samples 10
done
```

### Step 4: Fine-tune Model

```bash
python scripts/train.py \
    --data-dir ./data/raw \
    --upload-and-train \
    --wait
```

### Step 5: Evaluate Fine-tuned Model

```bash
python experiments/run_finetuned.py \
    --data-dir ./data/raw \
    --model-id ft:gpt-3.5-turbo-1106:org:model:id \
    --output-dir ./data/results/finetuned \
    --compare \
    --baseline-files \
        ./data/results/zero_shot/zero_shot_results_*.json \
        ./data/results/few_shot/few_shot_3shot_results_*.json
```

---

## Comparing Results

### Manual Comparison

```bash
# View all results
cat data/results/zero_shot/zero_shot_results_*.json | jq '.summary'
cat data/results/few_shot/few_shot_3shot_results_*.json | jq '.summary'
cat data/results/finetuned/finetuned_results_*.json | jq '.summary'
```

### Using evaluate.py

```bash
python scripts/evaluate.py \
    --data-dir ./data/raw \
    --test-split test/gold \
    --compare \
    --prediction-files \
        ./data/results/zero_shot/zero_shot_predictions_*.json \
        ./data/results/few_shot/few_shot_3shot_predictions_*.json \
        ./data/results/finetuned/finetuned_predictions_*.json \
    --model-names "Zero-shot" "Few-shot (3)" "Fine-tuned"
```

### Using Fine-tuned Compare

```bash
python experiments/run_finetuned.py \
    --data-dir ./data/raw \
    --model-id ft:gpt-3.5-turbo:... \
    --compare \
    --baseline-files \
        data/results/zero_shot/zero_shot_results_*.json \
        data/results/few_shot/few_shot_*shot_results_*.json
```

**Output:**

```
================================================================================
                              Model Comparison
================================================================================
            Approach |        UAS |        LAS |         F1 |  Samples
--------------------------------------------------------------------------------
          Zero-shot |     78.29% |     65.13% |     67.89% |      152
       Few-shot (3) |     82.45% |     69.74% |     72.15% |      152
         Fine-tuned |     87.50% |     76.97% |     79.82% |      152
================================================================================
```

---

## Tips and Best Practices

### 1. Start Small

Always test with `--max-samples 5` first to:
- Verify setup is correct
- Estimate cost and time
- Debug any issues

### 2. Monitor Progress

Watch for:
- Success rate (should be >90%)
- Parse time per sample
- API errors or rate limits

### 3. Compare Approaches

Run experiments in order:
1. Zero-shot (baseline, no training needed)
2. Few-shot (test if examples help)
3. Fine-tuned (best results, requires training)

### 4. Optimize Costs

**Zero-shot/Few-shot:**
- Use GPT-4 for best quality
- Use `--max-samples` for initial testing
- Consider GPT-3.5-turbo for large-scale evaluation

**Fine-tuned:**
- Fine-tune GPT-3.5-turbo (cheaper than GPT-4)
- Results often match or exceed GPT-4 zero-shot

### 5. Save Everything

Always keep:
- Predictions files (for later analysis)
- Results files (for comparison)
- Model IDs (for reproducibility)

### 6. Check Logs

If something fails:
```bash
# Check recent logs
tail -f logs/scidtb.log

# Search for errors
grep ERROR logs/scidtb.log
```

---

## Troubleshooting

### "OpenAI API key not found"
**Solution**: Set `OPENAI_API_KEY` in `.env` or use `--api-key` argument

### "No test data found"
**Solution**: Verify `--data-dir` points to correct location with `test/gold/` subdirectory

### "Success rate <90%"
**Solution**: 
- Check prompt templates in parser classes
- Verify model supports JSON mode
- Increase `--max-retries` in parser

### "Rate limit exceeded"
**Solution**:
- Add delays between requests
- Use smaller `--max-samples`
- Upgrade API tier

### "Out of memory"
**Solution**:
- Process in smaller batches
- Clear predictions list periodically
- Use `--no-save-predictions` for testing

---

## Next Steps

After running experiments:

1. **Analyze Results**: Compare UAS, LAS, F1 across approaches
2. **Error Analysis**: Check `print_detailed_alignments()` output
3. **Improve Prompts**: Update TODO sections in parser templates
4. **Fine-tune**: If zero/few-shot work, fine-tuning will likely improve
5. **Scale Up**: Run on full test set once satisfied with quality