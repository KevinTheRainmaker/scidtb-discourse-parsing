# SciDTB Discourse Dependency Parsing with LLMs

A comprehensive framework for evaluating Large Language Models on discourse dependency parsing using the SciDTB dataset.

## Features

- ✅ **Pydantic-based structured outputs** for guaranteed JSON validity
- 🔍 **Multiple parsing approaches**: Zero-shot, Few-shot, and Fine-tuned
- 📊 **Comprehensive evaluation**: UAS, LAS metrics with detailed analysis
- 🎯 **Scientific domain focus**: Optimized for scientific abstract parsing
- 🛠️ **Easy experimentation**: Simple CLI interface for running experiments

## Installation
```bash
# Clone the repository
git clone <repository-url>
cd scidtb-discourse-parsing

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your OpenAI API key
```

## Quick Start

### 1. Prepare Data
```bash
python scripts/prepare_data.py --data-dir ./dataset
```

### 2. Run Zero-shot Experiment
```bash
python experiments/run_zero_shot.py --max-samples 5
```

### 3. Run Few-shot Experiment
```bash
python experiments/run_few_shot.py --n-shots 3 --max-samples 5
```

### 4. Prepare Fine-tuning Data
```bash
python scripts/train.py --prepare-only
```

### 5. Fine-tune Model
```bash
python scripts/train.py --upload-and-train
```

### 6. Evaluate Fine-tuned Model
```bash
python experiments/run_finetuned.py --model-id ft:gpt-3.5-turbo:...
```

## Project Structure
```
scidtb-discourse-parsing/
├── src/              # Source code
│   ├── models/       # Pydantic models
│   ├── parsers/      # Parser implementations
│   ├── evaluation/   # Evaluation metrics
│   └── training/     # Fine-tuning utilities
├── experiments/      # Experiment scripts
├── scripts/          # Utility scripts
└── tests/           # Unit tests
```

## Usage Examples

See detailed examples in the `notebooks/` directory.

## Citation

If you use this code, please cite:
```bibtex
@article{yang2018scidtb,
  title={SciDTB: Discourse Dependency TreeBank for Scientific Abstracts},
  author={Yang, An and Li, Sujian},
  journal={arXiv preprint arXiv:1806.03653},
  year={2018}
}
```

## License

MIT License