"""
Configuration file for experiment paths

All paths are relative to the SCRIPTS directory root.
Modify these paths according to your local setup.
"""
import os
from pathlib import Path

# Get the SCRIPTS root directory
try:
    from utils.paths import SCRIPTS_ROOT
except ImportError:
    # Fallback if utils not available
    SCRIPTS_ROOT = Path(__file__).parent.parent.resolve()

# Dataset paths
DATASET_DIR = SCRIPTS_ROOT / "dataset"

# Result directories (you can change these to your preferred locations)
RESULT_DIR = SCRIPTS_ROOT / "results"
KOREAN_RESULT_DIR = RESULT_DIR / "korean"
ENGLISH_RESULT_DIR = RESULT_DIR / "english"

# Create result directories if they don't exist
for dir_path in [RESULT_DIR, KOREAN_RESULT_DIR, ENGLISH_RESULT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Dataset files (these should be placed in the dataset directory)
KOREAN_DATASET = DATASET_DIR / "korean_combined.csv"
ENGLISH_DATASET = DATASET_DIR / "english_combined.csv"

# Score output directories
KOREAN_SCORE_DIR = KOREAN_RESULT_DIR / "scores"
ENGLISH_SCORE_DIR = ENGLISH_RESULT_DIR / "scores"

for dir_path in [KOREAN_SCORE_DIR, ENGLISH_SCORE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Convert Path objects to strings for backward compatibility
DATASET_DIR = str(DATASET_DIR)
RESULT_DIR = str(RESULT_DIR)
KOREAN_RESULT_DIR = str(KOREAN_RESULT_DIR)
ENGLISH_RESULT_DIR = str(ENGLISH_RESULT_DIR)
KOREAN_DATASET = str(KOREAN_DATASET)
ENGLISH_DATASET = str(ENGLISH_DATASET)
KOREAN_SCORE_DIR = str(KOREAN_SCORE_DIR)
ENGLISH_SCORE_DIR = str(ENGLISH_SCORE_DIR)

