# Dataset Directory

Place your dataset CSV files here for social reasoning experiments.

## Required Files

- `english_combined.csv` - English language dialogue dataset
- `korean_combined.csv` - Korean language dialogue dataset

---

## Column Specifications

Both datasets use the **same standardized column format** with lowercase and hyphens for consistency.

### Core Columns

| Column Name | Type | Description |
|------------|------|-------------|
| `scene_id` | string | Unique identifier for each dialogue scene |
| `dialogue` | string | Full conversation text with speaker markers `[A]:` and `[B]:` |

### Ground Truth Annotations (with `_gold` suffix)

These are the gold standard human annotations used for evaluation:

#### Relation Classification

| Column Name | Type | Description |
|------------|------|-------------|
| `relation_high_probable_gold` | string | Ground truth high-probability relation (e.g., "Friends", "Parent-Child") |
| `relation_impossible_gold` | string | Ground truth impossible/unlikely relation |
| `high_probable_agreement` | float | Inter-annotator agreement score for high-probable relation |

#### Sub-Relation Classification

| Column Name | Type | Values | Description |
|------------|------|--------|-------------|
| `intimacy_gold` | string | intimate, not intimate, neutral, unknown | Intimacy level between speakers |
| `formality_gold` | string | formal, informal, neutral, unknown | Task vs. pleasure orientation |
| `hierarchy_gold` | string | equal, hierarchical, unknown | Power dynamics (A>B, A<B, or A=B) |
| `intimacy_agreement` | float | 0.0-1.0 | Inter-annotator agreement for intimacy |
| `formality_agreement` | float | 0.0-1.0 | Inter-annotator agreement for formality |
| `hierarchy_agreement` | float | 0.0-1.0 | Inter-annotator agreement for hierarchy |

#### Age and Gender Annotations

| Column Name | Type | Values | Description |
|------------|------|--------|-------------|
| `age-a_gold` | string | Child, Teen, Young adult, Middle-aged, Senior, Unknown | Age category for speaker A |
| `age-b_gold` | string | Child, Teen, Young adult, Middle-aged, Senior, Unknown | Age category for speaker B |
| `age_diff_gold` | string | A>B, A<B, A=B, Unknown | Age comparison between speakers |
| `gender-a_gold` | string | Male, Female, Other, Unknown | Gender for speaker A |
| `gender-b_gold` | string | Male, Female, Other, Unknown | Gender for speaker B |
| `gender_diff_gold` | string | Same, Different, Unknown | Gender comparison |

---

## Data Format Example

```csv
scene_id,dialogue,intimacy_gold,formality_gold,hierarchy_gold,relation_high_probable_gold,age-a_gold,age-b_gold,gender-a_gold,gender-b_gold
S001,"[A]: Hey, what's up?
[B]: Not much, just working.",intimate,informal,equal,Friends,Young adult,Young adult,Male,Male
S002,"[A]: Good morning, sir.
[B]: Good morning. Please sit down.",not intimate,formal,hierarchical,Professional colleagues,Young adult,Middle-aged,Female,Male
```

---

## Notes

### Encoding
- **UTF-8** encoding required
- Dialogues may contain multiple lines (properly escaped in CSV)

### Column Naming Convention
- **Standard format**: lowercase with hyphens for attributes (e.g., `age-a_gold`, `gender-b_gold`)
- **Gold suffix**: `_gold` for ground truth annotations
- **Agreement suffix**: `_agreement` for inter-annotator agreement scores

### Automatic Standardization
The `experiment/utils/column_mapping.py` module handles legacy column name formats:

| Legacy Format | Standard Format |
|--------------|-----------------|
| `Age_A`, `Age_B` | `age-a_gold`, `age-b_gold` |
| `Gender_A`, `Gender_B` | `gender-a_gold`, `gender-b_gold` |
| `Age_compare` | `age_diff_gold` |
| `Gender_compare` | `gender_diff_gold` |
| `Hierarchy`, `Intimacy`, `Pleasure` | `hierarchy_gold`, `intimacy_gold`, `formality_gold` |
| `relation_1`, `relation_best` | `relation_high_probable_gold` |

When loading data with `standardize_column_names()`, all formats are automatically converted to the standard format.

### Missing Values
- Use `Unknown` or empty string for uncertain annotations
- Agreement scores range from 0.0 (no agreement) to 1.0 (full agreement)

### Git Ignore
This directory is listed in `.gitignore` by default to avoid committing large dataset files.

