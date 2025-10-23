## Social Reasoning Analysis Scripts

Refactored and organized social reasoning experiment scripts.

### Directory Structure

```
SCRIPTS/
├── experiment/        # Main experiment code
│   ├── __init__.py
│   ├── config.py         # Configuration for paths
│   ├── core/             # Core evaluation and prompts
│   │   ├── evaluate.py
│   │   ├── prompts.py
│   │   └── utils.py
│   ├── scoring/          # Scoring and parsing
│   │   ├── scoring.py
│   │   └── parsers.py
│   ├── analysis/         # Analysis scripts
│   │   ├── calculate_acc*.py
│   │   └── eval_*.py
│   └── models/           # Model wrappers
│       ├── models.py
│       └── __init__.py
├── dataset/           # Dataset files (place your data here)
│   ├── english_combined.csv
│   └── korean_combined.csv
├── results/           # Results output directory (created automatically)
│   ├── english/
│   └── korean/
├── README.md          # This file
├── requirements.txt   # Python dependencies
└── ENV_EXAMPLE.txt    # Example environment variables
```

### Quick Start

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Set up API keys:**
Create a `.env` file with your API keys (see `ENV_EXAMPLE.txt`)

3. **Place your datasets:**
Put your dataset files in the `dataset/` directory

4. **Run experiments:**
```bash
cd experiment
python analysis/calculate_acc_en_rel.py
```

### Core Module

The `core` module provides fundamental building blocks:

- **evaluate.py**: Scoring functions for age, gender, relations, and sub-relations
- **prompts.py**: Generate prompts for various social reasoning tasks (English/Korean)
- **utils.py**: Utility functions for JSON parsing, file operations, accuracy calculation

### Scoring Module

The `scoring` module contains evaluation scripts:

- **scoring.py**: Universal scoring system that works with any task
- **parsers.py**: Parser utilities for extracting age/gender and sub-relation information

### Analysis Module

The `analysis` module contains accuracy calculation scripts:

- Various `calculate_acc*.py` scripts for different evaluation scenarios
- Evaluation scripts for running experiments
- Result generation and aggregation scripts

### Models Module

The `models` module provides unified interfaces for various LLM providers:

- **OpenAI** (GPT models)
- **Google** (Gemini models)
- **Anthropic** (Claude models)
- **HuggingFace** models (via vLLM)
- **OpenRouter** (unified API)

#### API Configuration

All API keys must be set as environment variables. Create a `.env` file in the SCRIPTS directory:

```bash
# See ENV_EXAMPLE.txt for the complete list
OPENAI_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
OPENROUTER_API_KEY=your_key_here
```

### Configuration

The `config.py` file manages all path configurations:

```python
from experiment.config import DATASET_DIR, RESULT_DIR, ENGLISH_DATASET

# All paths are relative to SCRIPTS root
# Modify config.py to match your directory structure
```

### Usage Examples

#### Using Core Modules

```python
import sys
sys.path.insert(0, 'experiment')

from core import extract_json, score_relation
from core import get_factor_prompt_ko, RELATIONSHIP_PROMPT

# Extract JSON from text
data = extract_json(generated_text)

# Generate prompts
prompt = get_factor_prompt_ko('intimacy', dialogue)

# Evaluate predictions
score = score_relation(predicted, ground_truth)
```

#### Using Models

```python
import sys
sys.path.insert(0, 'experiment')

from models import ChatModel

# Create a model instance (requires API key in environment)
model = ChatModel.create_model('gpt-4o', temperature=0.7, max_tokens=2048)

# Generate response
response = model.invoke("Hello, how are you?")
print(response.content)

# Use local HuggingFace models
model = ChatModel.create_model('llama-3.1-8b', tensor_parallel_size=2)
response = model.invoke("Your prompt here")
```

#### Using Scoring System

```bash
cd experiment
python -m scoring.scoring \
    --input_folder ../results/english \
    --answer_sheet ../dataset/english_combined.csv \
    --output_folder ../results/english/scores \
    --task_name english_relation
```

#### Using Parsers

```python
import sys
sys.path.insert(0, 'experiment')

from scoring import parse_age_gender_json, parse_sub_relation_json

# Parse age and gender information
age_gender = parse_age_gender_json(model_output)
print(age_gender)  # {"A": {"Age": "...", "Gender": "..."}, "B": {...}}

# Parse sub-relation information
sub_relation = parse_sub_relation_json(model_output)
print(sub_relation)  # {"intimacy": "...", "formality": "...", "hierarchy": "..."}
```

#### Running Analysis Scripts

```bash
cd experiment
python analysis/calculate_acc_en_rel.py
python analysis/calculate_acc_ko_rel.py
```

### Supported Models

**API-based models:**
- GPT-4o, GPT-4.1, GPT-4-turbo, GPT-4o-mini
- Gemini 1.5 Pro, Gemini 2.0 Flash, Gemini 2.5 Flash
- Claude 3.7 Sonnet
- Gemma 9B, Gemma 27B

**Local HuggingFace models:**
- Korean models: EXAONE, Kanana, Midm, A.X-4.0
- Llama 3.1 (8B, 70B)
- Qwen 2.5, Qwen 3
- Mistral, Mixtral
- DeepSeek Math/Coder

### Git Repository Structure

This repository is designed to be shared via Git:

- All hardcoded paths have been converted to relative paths
- Configuration is centralized in `config.py`
- API keys are managed via environment variables (not in repo)
- Dataset and results directories are created automatically
- Original experiment directory remains intact for reference at:
  ```
  /mnt/nas3/eunsu/social_reasoning_quiche_bu/experiment
  ```

### Notes

- All comments and docstrings are in English
- Shell scripts (`.sh`) are not included
- Duplicate and redundant files have been removed
- API keys must be set as environment variables (no hardcoded keys)
- Paths are configured via `config.py` for easy customization

### Contributing

When adding new analysis scripts:
1. Import paths from `config.py`
2. Use relative imports for modules within `experiment/`
3. Add documentation to this README

### Future Work

- Add unit tests for core modules
- Create comprehensive examples and tutorials
- Add support for more models
- Improve documentation with more usage examples
