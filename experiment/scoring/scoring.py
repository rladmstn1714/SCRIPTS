"""
Universal scoring system for social reasoning tasks
"""
import os
import pandas as pd
import json
import re
from datetime import datetime
import argparse


def parse_json_string(json_str):
    """Parse JSON string"""
    try:
        if pd.isna(json_str) or json_str is None or json_str == '':
            return {}
    except (ValueError, TypeError):
        return {}
    
    try:
        # Try extracting from JSON block
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', json_str, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
        
        # Try general JSON parsing
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return {}


def extract_relation_from_generated(generated_text):
    """Extract relation from generated column (improved version)"""
    if pd.isna(generated_text) or generated_text is None:
        return None
    
    # 1. Try extracting from JSON block
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', generated_text, re.DOTALL)
    if json_match:
        try:
            parsed_dict = json.loads(json_match.group(1))
            if parsed_dict and 'relation' in parsed_dict:
                return parsed_dict['relation']
        except (json.JSONDecodeError, TypeError):
            pass
    
    # 2. Try general JSON parsing
    try:
        parsed_dict = json.loads(generated_text)
        if parsed_dict and 'relation' in parsed_dict:
            return parsed_dict['relation']
    except (json.JSONDecodeError, TypeError):
        pass
    
    # 3. Find all JSON patterns in text
    json_patterns = [
        r'\{[^}]*"relation"[^}]*\}',  # Single line JSON
        r'\{[^}]*"relation"[^}]*"[^}]*\}',  # More complex JSON
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, generated_text, re.DOTALL)
        for match in matches:
            try:
                # Clean JSON string
                cleaned_match = match.strip()
                if not cleaned_match.endswith('}'):
                    cleaned_match += '}'
                
                parsed_dict = json.loads(cleaned_match)
                if parsed_dict and 'relation' in parsed_dict:
                    return parsed_dict['relation']
            except (json.JSONDecodeError, TypeError):
                continue
    
    # 4. Find "relation": "value" pattern
    relation_match = re.search(r'"relation"\s*:\s*"([^"]+)"', generated_text)
    if relation_match:
        return relation_match.group(1)
    
    # 5. Last attempt: relation: value pattern
    relation_match = re.search(r'relation["\s]*:["\s]*([^"\s,}]+)', generated_text, re.IGNORECASE)
    if relation_match:
        return relation_match.group(1).strip()
    
    return None


def find_relation_match(extracted_relation, s_relation_keys):
    """Match extracted relation with s_relation keys"""
    if not extracted_relation:
        return None
    
    relation_str = str(extracted_relation).strip()
    
    # 1. Exact match
    if relation_str in s_relation_keys:
        return relation_str
    
    # 2. Check inclusion relationship
    for key in s_relation_keys:
        if relation_str in key or key in relation_str:
            return key
    
    return None


def calculate_scores_for_file(file_path, standard_df):
    """Perform scoring for a single file"""
    print(f"Processing: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None
    
    # Check required columns
    required_cols = ['dialogue', 'generated']
    if not all(col in df.columns for col in required_cols):
        print(f"Missing required columns in {file_path}")
        return None
    
    results = []
    
    for idx, row in df.iterrows():
        dialogue = row['dialogue']
        generated = row['generated']
        
        # Find corresponding s_relation in standard_df based on dialogue
        matching_rows = standard_df[standard_df['dialogue'] == dialogue]
        if matching_rows.empty:
            continue
        
        s_relation = matching_rows.iloc[0]['s_relation']
        s_relation_dict = parse_json_string(s_relation)
        if not s_relation_dict:
            continue
        
        s_relation_keys = list(s_relation_dict.keys())
        
        # Extract relation from generated
        extracted_relation = extract_relation_from_generated(generated)
        
        # Attempt matching
        matched_key = find_relation_match(extracted_relation, s_relation_keys)
        
        if matched_key:
            # Match successful
            score_info = s_relation_dict[matched_key]
            is_in_possible = score_info.get('is_in_possible', 0)
            is_in_impossible = score_info.get('is_in_impossible', 0)
            has_judge_error = False
        else:
            # Match failed
            is_in_possible = 0
            is_in_impossible = 0
            has_judge_error = True
        
        results.append({
            'row_index': idx,
            'dialogue': dialogue,
            'extracted_relation': extracted_relation,
            'matched_relation': matched_key,
            'is_in_possible': is_in_possible,
            'is_in_impossible': is_in_impossible,
            'judge_error': has_judge_error
        })
    
    return pd.DataFrame(results)


def process_scoring_task(input_folder, answer_sheet_path, output_folder, task_name=None):
    """Universal scoring system - automatic scoring with input folder, answer sheet, and output folder"""
    
    # Extract task name from folder name if not provided
    if task_name is None:
        task_name = os.path.basename(input_folder)
    
    print(f"Starting scoring task: {task_name}")
    print(f"Input folder: {input_folder}")
    print(f"Answer sheet: {answer_sheet_path}")
    print(f"Output folder: {output_folder}")
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Load answer sheet
    try:
        standard_df = pd.read_csv(answer_sheet_path)
        print(f"Loaded answer sheet: {answer_sheet_path}")
        print(f"Answer sheet contains {len(standard_df)} rows")
    except Exception as e:
        print(f"Error loading answer sheet {answer_sheet_path}: {e}")
        return None, None, None, None
    
    all_results = []
    
    # Find all CSV files in folder (excluding aggregated files)
    csv_files = [f for f in os.listdir(input_folder) 
                if f.endswith('.csv') and 'aggregated' not in f.lower()]
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    for csv_file in csv_files:
        file_path = os.path.join(input_folder, csv_file)
        
        # Process file
        result_df = calculate_scores_for_file(file_path, standard_df)
        
        if result_df is not None and not result_df.empty:
            # Add file information
            result_df['file'] = csv_file
            all_results.append(result_df)
            
            # Save individual file results
            output_file = os.path.join(input_folder, f"scored_{csv_file}")
            result_df.to_csv(output_file, index=False)
            print(f"Saved scored results: {output_file}")
    
    # Combine all results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Calculate overall statistics
        summary_stats = {
            'total_rows': len(combined_df),
            'rows_with_judge_error': combined_df['judge_error'].sum(),
            'judge_error_rate': combined_df['judge_error'].mean(),
            'overall_possible_avg': combined_df['is_in_possible'].mean(),
            'overall_impossible_avg': combined_df['is_in_impossible'].mean(),
        }
        
        # File-wise statistics
        file_stats = combined_df.groupby('file').agg({
            'is_in_possible': 'mean',
            'is_in_impossible': 'mean',
            'judge_error': 'mean',
            'row_index': 'count'
        }).rename(columns={'row_index': 'total_rows'}).reset_index()
        
        # Extract model name (from file name)
        file_stats['model'] = file_stats['file'].apply(lambda x: x.split('_')[0] if '_' in x else x.replace('.csv', ''))
        
        # Model-wise statistics
        model_stats = file_stats.groupby('model').agg({
            'is_in_possible': 'mean',
            'is_in_impossible': 'mean',
            'judge_error': 'mean',
            'total_rows': 'sum'
        }).reset_index()
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save all results
        combined_df.to_csv(os.path.join(output_folder, f"all_{task_name}_scores_{timestamp}.csv"), index=False)
        
        # Save file-wise statistics
        file_stats.to_csv(os.path.join(output_folder, f"{task_name}_file_stats_{timestamp}.csv"), index=False)
        
        # Save model-wise statistics
        model_stats.to_csv(os.path.join(output_folder, f"{task_name}_model_stats_{timestamp}.csv"), index=False)
        
        # Save summary statistics as Markdown
        md_content = f"""# {task_name} Scoring Results

Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Task Information

- **Input Folder**: {input_folder}
- **Answer Sheet**: {answer_sheet_path}
- **Output Folder**: {output_folder}

## Overall Statistics

- Total rows processed: {summary_stats['total_rows']}
- Rows with judge errors: {summary_stats['rows_with_judge_error']} ({summary_stats['judge_error_rate']:.2%})
- Overall possible average: {summary_stats['overall_possible_avg']:.4f}
- Overall impossible average: {summary_stats['overall_impossible_avg']:.4f}

## Model-wise Statistics

| Model | Total Rows | Judge Error Rate | Possible Avg | Impossible Avg |
|-------|------------|------------------|--------------|----------------|
"""
        
        for _, row in model_stats.iterrows():
            md_content += f"| {row['model']} | {row['total_rows']} | {row['judge_error']:.2%} | {row['is_in_possible']:.4f} | {row['is_in_impossible']:.4f} |\n"
        
        md_content += f"""
## File-wise Statistics

| File | Model | Total Rows | Judge Error Rate | Possible Avg | Impossible Avg |
|------|-------|------------|------------------|--------------|----------------|
"""
        
        for _, row in file_stats.iterrows():
            md_content += f"| {row['file']} | {row['model']} | {row['total_rows']} | {row['judge_error']:.2%} | {row['is_in_possible']:.4f} | {row['is_in_impossible']:.4f} |\n"
        
        # Save Markdown file
        with open(os.path.join(output_folder, f"{task_name}_scoring_summary_{timestamp}.md"), 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        print(f"\n{task_name} scoring results saved to: {output_folder}")
        print(f"Summary: {os.path.join(output_folder, f'{task_name}_scoring_summary_{timestamp}.md')}")
        print(f"File stats: {os.path.join(output_folder, f'{task_name}_file_stats_{timestamp}.csv')}")
        print(f"Model stats: {os.path.join(output_folder, f'{task_name}_model_stats_{timestamp}.csv')}")
        
        return combined_df, summary_stats, file_stats, model_stats
    
    return None, None, None, None


def main():
    """Main function - handle command line arguments"""
    parser = argparse.ArgumentParser(description='Universal Social Reasoning Scoring System')
    parser.add_argument('--input_folder', required=True, help='Input folder containing CSV files to score')
    parser.add_argument('--answer_sheet', required=True, help='Path to the answer sheet CSV file')
    parser.add_argument('--output_folder', required=True, help='Output folder for results')
    parser.add_argument('--task_name', help='Optional task name (default: folder name)')
    
    args = parser.parse_args()
    
    # Check input folder exists
    if not os.path.exists(args.input_folder):
        print(f"Error: Input folder does not exist: {args.input_folder}")
        return
    
    # Check answer sheet exists
    if not os.path.exists(args.answer_sheet):
        print(f"Error: Answer sheet does not exist: {args.answer_sheet}")
        return
    
    # Execute scoring
    combined_df, summary_stats, file_stats, model_stats = process_scoring_task(
        args.input_folder, 
        args.answer_sheet, 
        args.output_folder, 
        args.task_name
    )
    
    if combined_df is not None:
        print(f"\n✅ Scoring completed successfully!")
        print(f"📊 Processed {summary_stats['total_rows']} rows")
        print(f"❌ Judge error rate: {summary_stats['judge_error_rate']:.2%}")
        print(f"✅ Overall possible average: {summary_stats['overall_possible_avg']:.4f}")
    else:
        print("❌ Scoring failed!")


if __name__ == "__main__":
    main()
