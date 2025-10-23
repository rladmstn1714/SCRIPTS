# Dataset Directory

Place your dataset CSV files here.

## Required Files

- `english_combined.csv` - English language dataset
- `korean_combined.csv` - Korean language dataset

## Dataset Format

### For Relation Classification

Required columns:
- `dialogue` - The conversation text (with speaker markers like [A]:, [B]:)
- `relation_high_probable_gold` - Ground truth relation (English dataset)
- `relation_high_probable_gold` - Ground truth relation (Korean dataset)

### For Sub-Relation Classification

Additional columns:
- `intimacy_gold` - Intimacy level (intimate, not intimate, neutral)
- `formality_gold` - Formality level (formal, informal, neutral)
- `hierarchy_gold` - Hierarchy level (equal, hierarchical)

### For Age/Gender Prediction

Additional columns:
- `age_A`, `age_B` - Age annotations for speakers A and B
- `gender_A`, `gender_B` - Gender annotations for speakers A and B

## Example Format

```csv
dialogue,relation_best,intimacy_gold,formality_gold,hierarchy_gold
"[A]: Hello, how are you?
[B]: I'm fine, thanks!",Friends,intimate,pleasure-oriented,equal
```

## Notes

- Dialogue should include speaker markers: `[A]:` and `[B]:`
- Multi-line dialogues are supported
- CSV encoding should be UTF-8
- This directory is gitignored by default to avoid committing large files

