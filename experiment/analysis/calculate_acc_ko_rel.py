"""
Calculate accuracy for Korean relation classification
"""
import os
import sys
import pandas as pd
from datetime import datetime
import re
import json

# Add parent directory to path to import from experiment modules
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from core.evaluate import extract_brace_content
from models import ChatModel
from config import KOREAN_RESULT_DIR, KOREAN_DATASET, KOREAN_SCORE_DIR

# Initialize judge model (requires API key)
try:
    judge = ChatModel.create_model("gpt-4o-mini")
except Exception as e:
    print(f"Warning: Could not initialize judge model: {e}")
    print("Scoring without judge model...")
    judge = None

# Path configuration
RESULT_DIR = KOREAN_RESULT_DIR
GT_FILE = KOREAN_DATASET
SCORE_DIR = KOREAN_SCORE_DIR


def extract_brace_content_or_raw(text):
    """
    Extract content within braces { ... } if present
    Otherwise return original text
    """
    if pd.isna(text):
        return ""
    text = str(text).strip()
    match = re.findall(r"\{(.*?)\}", text)
    if match:
        return match[0].strip()
    return text


def make_prompt(gt_list, model_outputs):
    """Create prompt for judge model to evaluate relation similarity"""
    prompt = """
    당신은 모델 출력물의 관계 분류 정확도를 평가하는 심사위원입니다.

    모델의 응답에서 최종 답변을 추출하고, 정답(GT) 관계와 비교하여 동일한 의미를 전달하는지 판단하세요.
    모델 출력이 정답 관계 중 하나와 같은 관계를 표현한다면 (단어는 다르지만 의미가 동등한 경우, 예: 동의어 사용) "1"로 표시하세요.
    정답 관계 중 어느 것과도 의미가 일치하지 않으면 "0"으로 표시하세요.

    설명 없이 숫자로만 답변하세요.

    ## 형식:
    GT: {GT_relation_1}, {GT_relation_2}, ...

    ## JSON 형식으로 출력:
    { "similarity": 0 or 1}
    """
    prompt += f"\n정답 관계: {gt_list}\n"
    prompt += f"\n모델 응답: {model_outputs}"
    prompt += "\n## 출력:\n"
    return prompt


def extract_json_from_text(text):
    """Extract JSON from text with various formats"""
    try:
        return json.loads(text)
    except:
        pass
    
    # Try to find JSON in text
    match = re.search(r'\{[^{}]*\}', text)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            pass
    
    return {"similarity": 0}


def calculate_accuracy_for_file(csv_path, gt_df):
    """
    Calculate accuracy for a single result file
    
    Args:
        csv_path: Path to result CSV file
        gt_df: Ground truth dataframe
        
    Returns:
        dict: Accuracy statistics
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
    
    correct = 0
    total = 0
    judge_errors = 0
    
    for idx, row in result_df.iterrows():
        dialogue = row['dialogue']
        generated = row.get('generated', '')
        
        # Find matching ground truth
        gt_rows = gt_df[gt_df['dialogue'] == dialogue]
        if gt_rows.empty:
            continue
        
        gt_relation = gt_rows.iloc[0].get('relation_best', '')
        
        # Extract prediction
        pred_relation = extract_brace_content_or_raw(generated)
        
        # Simple string matching
        if str(gt_relation).lower() in str(pred_relation).lower():
            correct += 1
        elif judge is not None:
            # Use judge model for more sophisticated matching
            try:
                prompt = make_prompt(gt_relation, pred_relation)
                response = judge.invoke(prompt)
                result = extract_json_from_text(response.content)
                if result.get('similarity', 0) == 1:
                    correct += 1
            except Exception as e:
                print(f"Judge error: {e}")
                judge_errors += 1
        
        total += 1
    
    accuracy = correct / total if total > 0 else 0
    
    return {
        'file': os.path.basename(csv_path),
        'total': total,
        'correct': correct,
        'accuracy': accuracy,
        'judge_errors': judge_errors
    }


def main():
    """Main function to process all result files"""
    # Load ground truth
    if not os.path.exists(GT_FILE):
        print(f"Ground truth file not found: {GT_FILE}")
        print("Please place your dataset in the dataset directory")
        return
    
    print(f"Loading ground truth from: {GT_FILE}")
    gt_df = pd.read_csv(GT_FILE)
    
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
        result = calculate_accuracy_for_file(file_path, gt_df)
        if result:
            all_results.append(result)
    
    # Save summary
    if all_results:
        summary_df = pd.DataFrame(all_results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(SCORE_DIR, f"accuracy_summary_{timestamp}.csv")
        summary_df.to_csv(output_path, index=False)
        print(f"\nSummary saved to: {output_path}")
        print("\nAccuracy Summary:")
        print(summary_df.to_string(index=False))
    else:
        print("No results to save")


if __name__ == "__main__":
    main()
