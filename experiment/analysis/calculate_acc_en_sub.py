"""
Calculate accuracy for English sub-relation classification
Sub-relations: intimacy, formality, hierarchy
"""
import os
import sys
import pandas as pd
from datetime import datetime
import re

# Add parent directory to path to import from experiment modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.evaluate import (
    extract_brace_content, score_age, score_age_diff,
    score_gender, score_gender_diff, score_relation, score_sub_relation
)
from utils import standardize_column_names
from config import ENGLISH_RESULT_DIR, ENGLISH_DATASET, ENGLISH_SCORE_DIR

# Path configuration
RESULT_DIR = ENGLISH_RESULT_DIR
GT_FILE = ENGLISH_DATASET
SCORE_DIR = ENGLISH_SCORE_DIR


def save_score_summary(output_dir, model, type_, cot, scores, save_df_column):
    """
    Save score summary to CSV file
    
    Args:
        output_dir: Output directory path
        model: Model name
        type_: Evaluation type
        cot: Chain of thought flag
        scores: Dictionary of scores
        save_df_column: DataFrame column name for saving
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model}_{type_}{'_cot' if cot else ''}_{timestamp}.csv"
    output_path = os.path.join(output_dir, filename)
    
    # Create summary dataframe
    summary_data = {
        'model': model,
        'type': type_,
        'cot': cot,
        **scores
    }
    
    df = pd.DataFrame([summary_data])
    df.to_csv(output_path, index=False)
    print(f"Summary saved to: {output_path}")
    
    return output_path


def calculate_sub_relation_accuracy(csv_path, gt_df):
    """
    Calculate sub-relation accuracy for a single result file
    
    Args:
        csv_path: Path to result CSV file
        gt_df: Ground truth dataframe with sub-relation annotations
        
    Returns:
        dict: Accuracy statistics for each sub-relation dimension
    """
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return None
    
    print(f"\nProcessing: {csv_path}")
    
    try:
        result_df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None
    
    if 'generated' not in result_df.columns or 'dialogue' not in result_df.columns:
        print(f"Missing required columns in {csv_path}")
        return None
    
    # Initialize counters for each dimension
    dimensions = ['intimacy', 'formality', 'hierarchy']
    scores = {dim: {'correct': 0, 'total': 0} for dim in dimensions}
    
    for idx, row in result_df.iterrows():
        dialogue = row['dialogue']
        generated = row.get('generated', '')
        
        # Find matching ground truth
        gt_rows = gt_df[gt_df['dialogue'] == dialogue]
        if gt_rows.empty:
            continue

        gt_row = gt_rows.iloc[0]
        
        # Try to score sub-relations
        try:
            dimension_scores = score_sub_relation(generated, gt_row)
            
            for dim in dimensions:
                if dim in dimension_scores:
                    score = dimension_scores[dim]
                    if score == 1:
                        scores[dim]['correct'] += 1
                        scores[dim]['total'] += 1
                    elif score == 0:
                        scores[dim]['total'] += 1
                    # Skip "None-Human" and "None-Model" cases
        except Exception as e:
            print(f"Error scoring sub-relation at row {idx}: {e}")
            continue

    # Calculate accuracies
    result = {
        'file': os.path.basename(csv_path)
    }
    
    for dim in dimensions:
        if scores[dim]['total'] > 0:
            accuracy = scores[dim]['correct'] / scores[dim]['total']
            result[f'{dim}_accuracy'] = accuracy
            result[f'{dim}_correct'] = scores[dim]['correct']
            result[f'{dim}_total'] = scores[dim]['total']
        else:
            result[f'{dim}_accuracy'] = 0.0
            result[f'{dim}_correct'] = 0
            result[f'{dim}_total'] = 0
    
    return result


def main():
    """Main function to process all result files"""
    # Load ground truth
    if not os.path.exists(GT_FILE):
        print(f"Ground truth file not found: {GT_FILE}")
        print("Please place your dataset in the dataset directory")
        return
    
    print(f"Loading ground truth from: {GT_FILE}")
    gt_df = pd.read_csv(GT_FILE)
    gt_df = standardize_column_names(gt_df)  # Standardize column names
    
    # Check if sub-relation columns exist
    required_cols = ['intimacy_gold', 'formality_gold', 'hierarchy_gold']
    missing_cols = [col for col in required_cols if col not in gt_df.columns]
    
    if missing_cols:
        print(f"Warning: Missing ground truth columns: {missing_cols}")
        print("Sub-relation evaluation requires these columns in the dataset")
        return
    
    # Find all result CSV files
    if not os.path.exists(RESULT_DIR):
        print(f"Result directory not found: {RESULT_DIR}")
        return
    
    result_files = [f for f in os.listdir(RESULT_DIR) if f.endswith('.csv')]
    
    if not result_files:
        print(f"No CSV files found in {RESULT_DIR}")
        return
    
    print(f"Found {len(result_files)} result files")
    
    # Process each file
    all_results = []
    for filename in result_files:
        file_path = os.path.join(RESULT_DIR, filename)
        result = calculate_sub_relation_accuracy(file_path, gt_df)
        if result:
            all_results.append(result)
    
    # Save summary
    if all_results:
        summary_df = pd.DataFrame(all_results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(SCORE_DIR, f"sub_relation_accuracy_{timestamp}.csv")
        summary_df.to_csv(output_path, index=False)
        print(f"\nSummary saved to: {output_path}")
        print("\nSub-Relation Accuracy Summary:")
        print(summary_df.to_string(index=False))
    else:
        print("No results to save")


if __name__ == "__main__":
    main()
