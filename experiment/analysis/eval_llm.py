"""
Evaluate LLM models on social reasoning tasks

This script runs LLM inference on dialogue data and generates predictions for:
- Relation classification
- Sub-relation classification (intimacy, formality, hierarchy)
- Age and gender prediction

Usage:
    python eval_llm.py --model_name gpt-4o --input_path ../dataset/korean_combined.csv --output_path ../results/korean/gpt4o.csv --type relation --mode plain-ko
"""
import pandas as pd
import argparse
import sys
import os

# Add experiment directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import ChatModel
from core.prompts import (
    get_factor_prompt_ko, RELATIONSHIP_PROMPT,
    INTIMACY_DEFINITION_KO, INTIMACY_INSTRUCTION_KO,
    FORMALITY_DEFINITION_KO, FORMALITY_INSTRUCTION_KO,
    HIERARCHY_DEFINITION_KO, HIERARCHY_INSTRUCTION_KO
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate LLM on social reasoning tasks")
    parser.add_argument('--model_name', type=str, required=True, 
                       help='Model name (e.g., gpt-4o, llama-3.1-8b)')
    parser.add_argument('--input_path', type=str, required=True, 
                       help='Input data path (CSV file)')
    parser.add_argument('--output_path', type=str, required=True, 
                       help='Output data path (CSV file)')
    parser.add_argument('--mode', type=str, default='plain-ko', 
                       help='Evaluation mode (plain-ko, cot-en, etc.)')
    parser.add_argument('--type', type=str, default='relation', 
                       help='Evaluation type (relation, sub-relation, age, gender)')
    parser.add_argument('--cot', action='store_true', 
                       help='Use chain-of-thought prompting')
    parser.add_argument('--tensor_parallel_size', type=int, default=None, 
                       help='Tensor parallel size for multi-GPU (e.g., 2, 4, 8)')
    return parser.parse_args()


# Korean relation examples
EXAMPLE_RELATIONS_KO = """
부모-자식, 형제,자매.남매, 조부모-손주, 사촌, 삼촌,이모,고모-조카
단짝 친구, 친구, 지인, 이웃, 모르는 사이
썸, 연애, 부부, 약혼관계, Friends with benefits, 불륜관계, 전애인 관계
동료, 상관-부하직원 관계
멘토-멘티, 선생-제자, 변호사-고객, 의사-환자, 집주인-세입자
경쟁관계, 라이벌관계, 숙적
"""

# English relation examples
EXAMPLE_RELATIONS_EN = """
Parent-Children, Brothers/Sisters, Grandparent-Grandchildren, Cousins, Uncle/Aunt-Niece
Friends, Acquaintances, Neighbors, Strangers
Romantic Interest, Dating, Married, Engaged, Friends with benefits, Affair, Ex-relationship
Coworkers, Professional colleagues, Supervisor-Subordinate relationship
Mentor-Mentee, Teacher-Student, Lawyer-Client, Doctor-Patient, Landlord-Tenant
Competitive relationship, Rivalry, Arch-enemies
"""


def prompt_generate(row, type='relation', mode="plain-ko"):
    """
    Generate prompt for LLM evaluation
    
    Args:
        row: DataFrame row containing dialogue data
        type: Evaluation type ('relation', 'sub-relation', 'age', 'gender', 'dual-relation')
        mode: Prompt mode ('plain-ko', 'cot-en', 'input_gender', etc.)
        
    Returns:
        str or dict: Generated prompt(s)
    """
    dialogue = row.get('final_diag', row.get('dialogue', ''))
    
    # Try to get additional information if available
    try:
        gender = row.get('gender', '')
        age = row.get('age', '')
        sub_relation = row.get('sub_relation', '')
        relation = row.get('relation_high_probable_gold', '')
    except:
        gender = age = sub_relation = relation = ''
    
    example_relations = EXAMPLE_RELATIONS_KO if 'ko' in mode else EXAMPLE_RELATIONS_EN
    
    # Relation classification
    if type == 'relation':
        if mode == "plain-ko":
            prompt = (
                "다음 대화를 읽고, 등장인물 A와 B의 관계를 추측하시오. "
                "관계를 추측할 때는 다음의 관계 예시를 참고하시오:\n"
                f"{example_relations}\n\n"
                "만약 위의 예시에 해당하는 관계라면 그대로 사용하고, "
                "예시에 없는 관계라고 판단되면 해당 관계를 직접 서술하시오.\n\n"
                "관계에 대한 최종 답은 반드시 JSON 형식으로 답변하시오:\n"
                "{\n"
                '  "relation": ""\n'
                "}\n\n"
                "대화:\n"
                f"{dialogue}\n\n"
            )
        
        elif mode == "cot-en":
            prompt = (
                "Read the following conversation and infer the relationship between A and B. "
                "When inferring the relationship, refer to these examples:\n"
                f"{example_relations}\n\n"
                "If it matches one of the examples, use it. Otherwise, describe the relationship yourself.\n\n"
                "First provide a rationale for the relationship, then give the final relationship.\n"
                "Your answer must be in JSON format:\n"
                "{\n"
                '  "rationale": "",\n'
                '  "relation": ""\n'
                "}\n\n"
                "Conversation:\n"
                f"{dialogue}\n\n"
                "Output (JSON):"
            )
        
        elif mode == "input_gender":
            prompt = (
                "다음 대화를 읽고, 등장하는 A와 B의 관계를 추측하시오. "
                "두 사람의 가능한 성별과 나이, 상대적 크기는 다음과 같다.\n"
                f"{gender}\n"
                f"{age}\n"
                "관계를 추측할 때는 다음의 관계 예시를 참고하시오:\n"
                f"{example_relations}\n\n"
                "만약 위의 예시에 해당하는 관계라면 그대로 사용하고, "
                "예시에 없는 관계라고 판단되면 해당 관계를 직접 서술하시오.\n\n"
                "출력은 반드시 JSON 형식으로만 답변하시오:\n"
                "{\n"
                '  "relation": ""\n'
                "}\n\n"
                "대화:\n"
                f"{dialogue}\n\n"
                "출력(JSON):"
            )
        
        elif mode == "input_sub_relation":
            sub_relation_def = (
                f"Intimacy: {INTIMACY_DEFINITION_KO}\n"
                f"Formality: {FORMALITY_DEFINITION_KO}\n"
                f"Hierarchy: {HIERARCHY_DEFINITION_KO}"
            )
            
            prompt = (
                "다음 대화를 읽고, 등장인물 A와 B의 관계를 추측하시오. "
                "관계를 추측할 때는 다음의 관계 예시를 참고하여 "
                "만약 위의 예시에 해당하는 관계라면 그대로 사용하고, "
                "예시에 없는 관계라고 판단되면 해당 관계를 직접 서술하시오.\n\n"
                f"{example_relations}\n\n"
                "관계에 대한 최종 답은 반드시 JSON 형식으로 답변하시오:\n"
                "{\n"
                '  "relation": ""\n'
                "}\n\n"
                "대화속 A와 B의 Intimacy level, Formality level, Hierarchy level은 다음과 같고 이를 관계 유추에 참고하시오.\n"
                f"{sub_relation}\n"
                "Intimacy level, Formality level, Hierarchy level의 정의는 다음을 참고하시오.\n"
                f"{sub_relation_def}\n\n"
                "대화:\n"
                f"{dialogue}\n\n"
            )
        
        else:  # default mode
            prompt = (
                "다음 대화를 읽고, 등장인물 A와 B의 관계를 추측하시오. "
                "관계를 추측할 때는 다음의 관계 예시를 참고하시오:\n"
                f"{example_relations}\n\n"
                "만약 위의 예시에 해당하는 관계라면 그대로 사용하고, "
                "예시에 없는 관계라고 판단되면 해당 관계를 직접 서술하시오.\n\n"
                "관계에 대한 최종 답은 반드시 JSON 형식으로 답변하시오:\n"
                "{\n"
                '  "relation": ""\n'
                "}\n\n"
                "대화:\n"
                f"{dialogue}\n\n"
                "출력(JSON):"
            )
    
    # Sub-relation classification
    elif type == 'sub-relation':
        sub_relation_def = (
            f"Intimacy: {INTIMACY_DEFINITION_KO}\n"
            f"Formality: {FORMALITY_DEFINITION_KO}\n"
            f"Hierarchy: {HIERARCHY_DEFINITION_KO}"
        )
        sub_relation_instruction = (
            f"Intimacy: {INTIMACY_INSTRUCTION_KO}\n"
            f"Formality: {FORMALITY_INSTRUCTION_KO}\n"
            f"Hierarchy: {HIERARCHY_INSTRUCTION_KO}"
        )
        
        prompt_intimacy = get_factor_prompt_ko('intimacy', dialogue)
        prompt_formality = get_factor_prompt_ko('formality', dialogue)
        prompt_hierarchy = get_factor_prompt_ko('hierarchy', dialogue)
        
        if mode == 'input_relation':
            relational_prompt = f"이때 A와 B의 가능한 사회적 관계는 {relation} 입니다. 이를 추론에 사용하시오."
            prompt_intimacy += relational_prompt
            prompt_formality += relational_prompt
            prompt_hierarchy += relational_prompt
        
        prompt = {
            'intimacy': prompt_intimacy,
            'formality': prompt_formality,
            'hierarchy': prompt_hierarchy
        }
    
    # Age prediction
    elif type == "age":
        prompt = (
            "다음 대화를 읽고, 등장인물 A와 B의 나이를 추측하고 나이의 크기를 비교하시오. "
            "가능한 나이 범주를 모두 말하시오.\n\n"
            "나이에 대한 최종 답은 반드시 JSON 형식으로 답변하시오:\n"
            "{\n"
            '  "A": "",\n'
            '  "B": ""\n'            
            "}\n\n"
            "대화:\n"
            f"{dialogue}\n\n"
        )
    
    # Gender prediction
    elif type == "gender":
        prompt = (
            "다음 대화를 읽고, 등장인물 A와 B의 성별을 추측하시오. "
            "성별은 '남', '여', '그 외' 로 대답하시오.\n"
            "가능한 성별을 모두 말하시오.\n\n"
            "성별에 대한 최종 답은 반드시 JSON 형식으로 답변하시오:\n"
            "{\n"
            '  "A": "",\n'
            '  "B": ""\n'
            "}\n\n"
            "대화:\n"
            f"{dialogue}\n\n"
        )
    
    # Dual relation (multiple relations)
    elif type == 'dual-relation':
        prompt = (
            "다음 대화를 읽고, 등장인물 A와 B의 관계를 추측하시오. "
            "두 사람은 동시에 두 개 이상의 관계를 동시에 가집니다. 두 사람이 속한 관계를 모두 말하시오. "
            "예를 들어, 두 사람이 친구이자 직장동료이면 '친구, 동료' 라고 서술하시오.\n"
            "관계를 추측할 때는 다음의 관계 예시를 참고하시오:\n"
            f"{example_relations}\n\n"
            "만약 위의 예시에 해당하는 관계라면 그대로 사용하고, "
            "예시에 없는 관계라고 판단되면 해당 관계를 직접 서술하시오.\n\n"
            "관계에 대한 최종 답은 반드시 JSON 형식으로 답변하시오:\n"
            "{\n"
            '  "relation": ""\n'
            "}\n\n"
            "대화:\n"
            f"{dialogue}\n\n"
        )
    
    else:
        raise ValueError(f"Invalid type: {type}. Choose from 'relation', 'sub-relation', 'age', 'gender', 'dual-relation'.")
    
    return prompt


def main():
    """Main function"""
    args = parse_args()
    
    # Load data
    print(f"Loading data from: {args.input_path}")
    if not os.path.exists(args.input_path):
        print(f"Error: Input file not found: {args.input_path}")
        return
    
    data = pd.read_csv(args.input_path)
    print(f"Loaded {len(data)} rows")
    
    # Create model
    print(f"Creating model: {args.model_name}")
    try:
        if args.cot:
            model = ChatModel.create_model(
                args.model_name, 
                max_tokens=700, 
                tensor_parallel_size=args.tensor_parallel_size
            )
        else:
            model = ChatModel.create_model(
                args.model_name, 
                max_tokens=300, 
                tensor_parallel_size=args.tensor_parallel_size
            )
        print("Model created successfully")
    except Exception as e:
        print(f"Error creating model: {e}")
        return
    
    # Load existing results if output file exists
    if os.path.exists(args.output_path):
        existing_data = pd.read_csv(args.output_path)
        print(f"Loaded existing results: {len(existing_data)} rows")
    else:
        existing_data = pd.DataFrame(columns=["dialogue", "generated"])
        print("Starting fresh (no existing results)")
    
    # Check if already finished
    if len(existing_data) == len(data):
        print("All rows already processed!")
        return
    
    # Process data
    print(f"\nProcessing {len(data)} rows...")
    for idx, row in data.iterrows():
        dialogue = row.get("final_diag", row.get("dialogue", ""))
        
        # Skip if already processed
        if dialogue in existing_data["dialogue"].values:
            print(f"Skipping already processed row {idx + 1}/{len(data)}...")
            continue
        
        # Generate prompt
        prompt = prompt_generate(row, type=args.type, mode=args.mode)
        
        # Get response
        if isinstance(prompt, dict):
            # Multi-prompt case (sub-relations)
            response = {}
            for p in prompt:
                try:
                    if args.cot:
                        prompt[p] = prompt[p] + "\nstep by step으로 생각하여 rationale을 먼저 생성하고, 최종 답을 JSON 형식으로 답변하시오. 답변:\n"
                    
                    response[p] = model.invoke(prompt[p]).content
                except Exception as e:
                    print(f"Error occurred while processing row {idx + 1}/{len(data)}: {e}")
                    response[p] = "Error"
        else:
            # Single prompt case
            try:
                if args.cot:
                    prompt = prompt + "\nstep by step으로 생각하여 rationale을 먼저 생성하고, 최종 답을 JSON 형식으로 답변하시오. 답변:\n"
                else:
                    prompt = prompt + "\n답변:\n"
                
                response = model.invoke(prompt).content
            except Exception as e:
                print(f"Error occurred while processing row {idx + 1}/{len(data)}: {e}")
                response = "Error"
        
        print(f"Processed {idx + 1}/{len(data)}: {str(response)[:100]}...")
        
        # Add new result
        new_result = pd.DataFrame([{"dialogue": dialogue, "generated": response}])
        existing_data = pd.concat([existing_data, new_result], ignore_index=True)
        
        # Save immediately (incremental save)
        existing_data.to_csv(args.output_path, index=False, encoding='utf-8')
    
    print(f"\nResults saved to: {args.output_path}")


if __name__ == "__main__":
    main()

