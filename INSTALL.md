# Installation Guide

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Setup

### 1. Clone or download this repository

```bash
git clone <your-repo-url>
cd SCRIPTS
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API keys

Copy the example environment file and fill in your API keys:

```bash
cp ENV_EXAMPLE.txt .env
# Edit .env with your favorite editor and add your API keys
```

### 4. Prepare your datasets

Place your dataset files in the `dataset/` directory:

```bash
# Example:
cp /path/to/your/english_combined.csv dataset/
cp /path/to/your/korean_combined.csv dataset/
```

See `dataset/README.md` for more information about dataset format.

### 5. Verify installation

```bash
cd experiment
python -c "from models import ChatModel; print('Installation successful!')"
```

## Directory Structure After Installation

```
SCRIPTS/
├── experiment/           # Main code
├── dataset/             # Your datasets
├── results/             # Generated automatically
├── .env                 # Your API keys (not in git)
├── .gitignore
└── README.md
```

## Running Experiments

### Calculate accuracy for English relations

```bash
cd experiment
python analysis/calculate_acc_en_rel.py
```

### Calculate accuracy for Korean relations

```bash
cd experiment
python analysis/calculate_acc_ko_rel.py
```

### Use the scoring system

```bash
cd experiment
python -m scoring.scoring \
    --input_folder ../results/english \
    --answer_sheet ../dataset/english_combined.csv \
    --output_folder ../results/english/scores \
    --task_name my_task
```

## Troubleshooting

### ModuleNotFoundError

Make sure you're in the right directory and have added the experiment path:

```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
```

### API Key Errors

Check that your `.env` file exists and contains valid API keys:

```bash
cat .env  # Should show your API keys
```

### Dataset Not Found

Verify your dataset files are in the correct location:

```bash
ls -l dataset/
```

## GPU Support (Optional)

For local model inference with GPU support:

```bash
# Install CUDA toolkit (if not already installed)
# Then install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install vLLM for efficient inference
pip install vllm
```

## Development

To contribute to this project:

1. Create a new branch for your feature
2. Make your changes
3. Test your changes
4. Submit a pull request

For adding new analysis scripts, remember to:
- Import paths from `experiment.config`
- Use relative imports for experiment modules
- Add documentation
