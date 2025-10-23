# Experiment Module - Technical Documentation

Complete documentation for the social reasoning experiment framework.

## Quick Start

### 1. Installation

```bash
# From SCRIPTS directory
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
# Copy example and edit
cp ENV_EXAMPLE.txt .env

# Edit .env and add your API keys
nano .env
```

Required API keys:
- `OPENAI_API_KEY` - For GPT models
- `GEMINI_API_KEY` - For Gemini models (optional)
- `OPENROUTER_API_KEY` - For OpenRouter API (optional)

### 3. Prepare Dataset

Place your dataset files in the `dataset/` directory:
- `korean_combined.csv` - Korean dialogue data
- `english_combined.csv` - English dialogue data

See Dataset Format section below for required columns.

### 4. Run Experiments

```bash
cd experiment

# Generate predictions
python analysis/eval_llm.py \
    --model_name gpt-4o \
    --input_path ../dataset/korean_combined.csv \
    --output_path ../results/korean/gpt4o.csv \
    --type relation \
    --mode plain-ko

# Calculate accuracy
python analysis/calculate_acc_ko_rel.py
```

---

## Detailed Technical Documentation

Detailed technical documentation for the experiment framework.

## Architecture Overview

The experiment module is organized into 4 main components:

1. **core** - Fundamental evaluation and prompt generation
2. **scoring** - Scoring systems and parsers
3. **analysis** - Analysis scripts for running experiments
4. **models** - LLM model wrappers and interfaces

## Module Reference

### Core Module (`core/`)

#### evaluate.py

Scoring functions for different prediction tasks.

**Functions:**

```python
score_age(pred: str, gt: str) -> tuple[int|str, int|str]
```
Score age predictions for persons A and B.
- Returns: `(score_A, score_B)` where score is 0/1 or "None-Human"/"None-Model"

```python
score_age_diff(pred: str, gt: str) -> int|str
```
Score age difference comparison (`>`, `<`, `=`).

```python
score_gender(pred: str, gt: str) -> tuple[int|str, int|str]
```
Score gender predictions for persons A and B.

```python
score_gender_diff(pred: str, gt: str) -> int|str
```
Score gender difference comparison.

```python
score_relation(pred: str, gt: str) -> int
```
Score relation classification (simple exact match).

```python
score_sub_relation(pred: str, gt: pd.Series) -> dict
```
Score sub-relation predictions (intimacy, formality, hierarchy).
- Returns: `{'intimacy': score, 'formality': score, 'hierarchy': score}`

#### prompts.py

Prompt template generation.

**Functions:**

```python
get_factor_prompt(factor: str, dialogue: list) -> str
```
Generate English prompt for single factor (intimacy/formality/hierarchy).

```python
get_factor_prompt_ko(factor: str, dialogue: list) -> str
```
Generate Korean prompt for single factor.

```python
get_factor_prompt_all(dialogue: list) -> str
get_factor_prompt_all_ko(dialogue: list) -> str
```
Generate prompts for all three factors at once.

**Constants:**
- `RELATIONSHIP_PROMPT` - Relation classification prompt template
- `INTIMACY_DEFINITION`, `FORMALITY_DEFINITION`, `HIERARCHY_DEFINITION`
- Korean versions with `_KO` suffix

#### utils.py

Utility functions for data processing.

**Functions:**

```python
extract_json(text: str) -> dict|str
```
Extract and parse JSON from text (handles markdown code blocks, malformed JSON).

```python
extract_brace_content(text: str) -> str|None
```
Extract content within braces `{...}`.

```python
accuracy(val1: list, val2: list) -> float
```
Calculate accuracy between two lists.

```python
write_csv_row(values: list|dict, filename: str)
```
Append row to CSV file (handles both list and dict formats).

```python
find_csv_files(directory: str, ends: list) -> list
```
Find CSV files matching pattern.

### Scoring Module (`scoring/`)

#### scoring.py

Universal scoring system for any task.

**Main Function:**

```python
process_scoring_task(
    input_folder: str,
    answer_sheet_path: str,
    output_folder: str,
    task_name: str = None
) -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]
```

Automatically scores all CSV files in a folder.

- Returns: `(combined_results, summary_stats, file_stats, model_stats)`
- Generates: CSV files and Markdown summary

**Command Line:**

```bash
python -m scoring.scoring \
    --input_folder ../results/korean \
    --answer_sheet ../dataset/korean_combined.csv \
    --output_folder ../results/korean/scores \
    --task_name korean_relation
```

#### parsers.py

Parse structured information from model outputs.

**Functions:**

```python
parse_age_gender_json(text: str) -> dict|None
```
Parse age/gender from text.
- Returns: `{"A": {"Age": "...", "Gender": "..."}, "B": {...}}`

```python
parse_sub_relation_json(text: str) -> dict|None
```
Parse sub-relations from text.
- Returns: `{"intimacy": "...", "formality": "...", "hierarchy": "..."}`

```python
clean_json_string(json_str: str) -> str
```
Clean and normalize JSON strings for parsing.

### Models Module (`models/`)

#### models.py

Unified interface for multiple LLM providers.

**Main Class:**

```python
ChatModel.create_model(
    name: str,
    temp: float = 0.7,
    max_tokens: int = 2048,
    tensor_parallel_size: int = None,
    reasoning_enabled: bool = False,
    reasoning_budget: int = None,
    reasoning_effort: str = None,
    thinking_budget: int = None
) -> Model
```

Create a model instance.

**Examples:**

```python
# OpenAI GPT
model = ChatModel.create_model('gpt-4o')

# Gemini with reasoning
model = ChatModel.create_model(
    'gemini-2.5-flash',
    reasoning_enabled=True,
    reasoning_effort='high'
)

# Local HuggingFace model with 2 GPUs
model = ChatModel.create_model(
    'llama-3.1-8b',
    tensor_parallel_size=2
)

# Generate response
response = model.invoke("Your prompt")
print(response.content)  # AIMessage with .content attribute
```

**Model Classes:**
- `HuggModel` - For local HuggingFace models (uses vLLM)
- `GeminiModel` - For Google Gemini (native API)
- `OpenrouterModel` - For OpenRouter unified API
- `ChatModel` - Factory class (use this)

### Config Module (`config.py`)

Path configuration for the framework.

**Variables:**

```python
from config import (
    SCRIPTS_ROOT,           # SCRIPTS directory root
    DATASET_DIR,            # dataset/ directory
    RESULT_DIR,             # results/ directory
    KOREAN_DATASET,         # dataset/korean_combined.csv
    ENGLISH_DATASET,        # dataset/english_combined.csv
    KOREAN_RESULT_DIR,      # results/korean/
    ENGLISH_RESULT_DIR,     # results/english/
    KOREAN_SCORE_DIR,       # results/korean/scores/
    ENGLISH_SCORE_DIR,      # results/english/scores/
)
```

All paths are:
- Automatically resolved relative to SCRIPTS root
- Converted to strings for compatibility
- Created automatically if they don't exist

## Analysis Scripts

### eval_llm.py

Run LLM inference on dialogue data.

**Usage:**

```bash
python analysis/eval_llm.py \
    --model_name MODEL \
    --input_path INPUT.csv \
    --output_path OUTPUT.csv \
    --type TYPE \
    --mode MODE \
    [--cot] \
    [--tensor_parallel_size N]
```

**Task Types:**
- `relation` - Relation classification
- `sub-relation` - Intimacy/formality/hierarchy
- `age` - Age prediction
- `gender` - Gender prediction
- `dual-relation` - Multiple simultaneous relations

**Modes:**
- `plain-ko` - Korean without reasoning
- `cot-en` - English with chain-of-thought
- `input_gender` - Provide gender/age hints
- `input_sub_relation` - Provide sub-relation hints

**Features:**
- Incremental saving (resume interrupted runs)
- Skip already processed dialogues
- Support for dict prompts (multi-prompt tasks)

### calculate_acc_*.py

Calculate accuracy for various tasks.

**Scripts:**
- `calculate_acc_en_rel.py` - English relations
- `calculate_acc_ko_rel.py` - Korean relations
- `calculate_acc_en_sub.py` - English sub-relations
- `calculate_acc_ko_sub.py` - Korean sub-relations
- `calculate_acc_judge.py` - LLM judge evaluation

**Common Pattern:**

All scripts:
1. Load ground truth from `config.DATASET`
2. Find result files in `config.RESULT_DIR`
3. Calculate accuracy
4. Save to `config.SCORE_DIR` with timestamp

**Example:**

```bash
cd experiment/analysis
python calculate_acc_ko_rel.py
# Output: results/korean/scores/accuracy_summary_YYYYMMDD_HHMMSS.csv
```

## Best Practices

### Import Pattern

Always use this pattern in analysis scripts:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from config import DATASET_DIR
from core import extract_json
from models import ChatModel
```

### Error Handling

Handle missing files gracefully:

```python
if not os.path.exists(dataset_file):
    print(f"Dataset not found: {dataset_file}")
    print("Please place your dataset in the dataset/ directory")
    return
```

### Saving Results

Always use timestamps:

```python
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"results_{timestamp}.csv"
```

### API Key Management

Never hardcode API keys:

```python
# ❌ Bad
api_key = "sk-..."

# ✅ Good
from models import ChatModel  # Handles API keys automatically
model = ChatModel.create_model('gpt-4o')
```

## Advanced Usage

### Custom Scoring Functions

Add to `core/evaluate.py`:

```python
def score_custom_task(pred, gt):
    """
    Score custom task
    
    Args:
        pred: Prediction
        gt: Ground truth
        
    Returns:
        float: Score
    """
    # Your scoring logic
    return score
```

Then export in `core/__init__.py`.

### Custom Prompts

Add to `core/prompts.py`:

```python
CUSTOM_PROMPT = """
Your prompt template here: {Variable}
"""

def get_custom_prompt(variable, dialogue):
    return CUSTOM_PROMPT.format(Variable=variable, Dialogue=dialogue)
```

### Batch Processing

For processing multiple models:

```bash
for model in gpt-4o gpt-4o-mini gemini-2.5-flash; do
    python analysis/eval_llm.py \
        --model_name $model \
        --input_path ../dataset/korean_combined.csv \
        --output_path ../results/korean/${model}.csv \
        --type relation
done
```

## Performance Tips

### Multi-GPU Inference

For local models:

```bash
python analysis/eval_llm.py \
    --model_name llama-3.1-8b \
    --tensor_parallel_size 4 \
    ...
```

### Batch Size

For vLLM models, adjust in `models/models.py`:

```python
self.model = LLM(
    model=model_path,
    tensor_parallel_size=tensor_parallel_size,
    max_model_len=8192  # Adjust as needed
)
```

## API Reference

See inline documentation in each module file for complete API details.

## Support

For issues or questions, refer to:
- Module docstrings
- Function documentation
- Example scripts in `analysis/`


def get_custom_prompt(variable, dialogue):
    return CUSTOM_PROMPT.format(Variable=variable, Dialogue=dialogue)
```

### Batch Processing

For processing multiple models:

```bash
for model in gpt-4o gpt-4o-mini gemini-2.5-flash; do
    python analysis/eval_llm.py \
        --model_name $model \
        --input_path ../dataset/korean_combined.csv \
        --output_path ../results/korean/${model}.csv \
        --type relation
done
```

## Performance Tips

### Multi-GPU Inference

For local models:

```bash
python analysis/eval_llm.py \
    --model_name llama-3.1-8b \
    --tensor_parallel_size 4 \
    ...
```

### Batch Size

For vLLM models, adjust in `models/models.py`:

```python
self.model = LLM(
    model=model_path,
    tensor_parallel_size=tensor_parallel_size,
    max_model_len=8192  # Adjust as needed
)
```

## API Reference

See inline documentation in each module file for complete API details.

## Support

For issues or questions, refer to:
- Module docstrings
- Function documentation
- Example scripts in `analysis/`
