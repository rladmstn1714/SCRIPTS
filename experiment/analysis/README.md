# Analysis Scripts

This directory contains scripts for analyzing model outputs and calculating accuracy.

## Main Scripts

### Model Evaluation

- **eval_llm.py** - Run LLM inference on dialogue data to generate predictions
  ```bash
  python eval_llm.py --model_name gpt-4o --input_path ../dataset/korean_combined.csv --output_path ../results/korean/gpt4o.csv --type relation
  ```

### Relation Classification

- **calculate_acc_en_rel.py** - Calculate accuracy for English relation classification
- **calculate_acc_ko_rel.py** - Calculate accuracy for Korean relation classification

### Sub-Relation Classification

- **calculate_acc_en_sub.py** - Calculate accuracy for English sub-relation (intimacy, formality, hierarchy)
- **calculate_acc_ko_sub.py** - Calculate accuracy for Korean sub-relation

### LLM Judge Evaluation

- **calculate_acc_judge.py** - Use LLM judge to evaluate relation similarity (more sophisticated than string matching)

## Usage

All scripts should be run from the `experiment/` directory:

```bash
cd experiment

# Run model inference
python analysis/eval_llm.py \
    --model_name gpt-4o \
    --input_path ../dataset/korean_combined.csv \
    --output_path ../results/korean/gpt4o.csv \
    --type relation \
    --mode plain-ko

# Calculate accuracy
python analysis/calculate_acc_en_rel.py
python analysis/calculate_acc_ko_rel.py

# Use LLM judge for evaluation
python analysis/calculate_acc_judge.py
```

## Configuration

Scripts use paths from `../config.py`:
- `DATASET_DIR` - Location of dataset files
- `RESULT_DIR` - Location to save results
- `ENGLISH_DATASET`, `KOREAN_DATASET` - Dataset files

## eval_llm.py Options

### Task Types
- `relation` - Classify relationship between speakers
- `sub-relation` - Classify intimacy, formality, hierarchy
- `age` - Predict age of speakers
- `gender` - Predict gender of speakers
- `dual-relation` - Identify multiple simultaneous relations

### Modes
- `plain-ko` - Korean prompt without reasoning
- `cot-en` - English prompt with chain-of-thought
- `input_gender` - Provide gender/age hints
- `input_sub_relation` - Provide sub-relation hints
- `input_relation` - Provide relation hints

### Example Commands

```bash
# Korean relation classification
python analysis/eval_llm.py \
    --model_name gpt-4o \
    --input_path ../dataset/korean_combined.csv \
    --output_path ../results/korean/gpt4o_relation.csv \
    --type relation \
    --mode plain-ko

# English relation with chain-of-thought
python analysis/eval_llm.py \
    --model_name gpt-4o \
    --input_path ../dataset/english_combined.csv \
    --output_path ../results/english/gpt4o_cot.csv \
    --type relation \
    --mode cot-en \
    --cot

# Sub-relation classification
python analysis/eval_llm.py \
    --model_name gpt-4o \
    --input_path ../dataset/korean_combined.csv \
    --output_path ../results/korean/gpt4o_subrel.csv \
    --type sub-relation

# Use local model with multi-GPU
python analysis/eval_llm.py \
    --model_name llama-3.1-8b \
    --input_path ../dataset/korean_combined.csv \
    --output_path ../results/korean/llama_relation.csv \
    --type relation \
    --tensor_parallel_size 2
```

## Adding New Scripts

When creating new analysis scripts:

1. Import paths from config:
```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import DATASET_DIR, RESULT_DIR
from core import extract_json, score_relation
```

2. Use English for all comments and docstrings

3. Handle missing files gracefully:
```python
if not os.path.exists(dataset_file):
    print(f"Dataset not found: {dataset_file}")
    print("Please place your dataset in the dataset/ directory")
    return
```

4. Save results with timestamps:
```python
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"results_{timestamp}.csv"
```

## Workflow

Typical analysis workflow:

1. **Generate predictions** using `eval_llm.py`
2. **Calculate accuracy** using `calculate_acc_*.py` scripts
3. **Optional:** Use `calculate_acc_judge.py` for more sophisticated evaluation

```bash
# Step 1: Generate predictions
python analysis/eval_llm.py --model_name gpt-4o --input_path ../dataset/korean_combined.csv --output_path ../results/korean/gpt4o.csv

# Step 2: Calculate accuracy
python analysis/calculate_acc_ko_rel.py

# Step 3: (Optional) Use LLM judge
python analysis/calculate_acc_judge.py
```
