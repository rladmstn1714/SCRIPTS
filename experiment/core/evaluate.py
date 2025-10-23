"""
Evaluation functions module
"""
import ast
import json
import re
from json_repair import repair_json
from .utils import extract_brace_content, decode_unicode_escape


def score_age(pred, gt):
    """
    Calculate age prediction score
    
    Args:
        pred: Prediction
        gt: Ground truth
    
    Returns:
        tuple: (score_A, score_B)
    """
    pred_text = extract_brace_content(pred)
    
    try:
        gt_eval_A = ast.literal_eval(gt)['A']
        gt_eval_B = ast.literal_eval(gt)['B']
    except Exception:
        return ("None-Human", "None-Human")
    
    score_A = 0
    score_B = 0
    
    # Ground truth validation
    if ("남" in gt_eval_A) and ("여" in gt_eval_A):
        score_A = "None-Human"
    elif ("알수없음" in gt_eval_A):
        score_A = "None-Human"
    
    if ("남" in gt_eval_B) and ("여" in gt_eval_B):
        score_B = "None-Human"
    elif ("알수없음" in gt_eval_B):
        score_B = "None-Human"
    
    if pred_text is None:
        if score_A == 0:
            score_A = "None-Model"
        if score_B == 0:
            score_B = "None-Model"
        return (score_A, score_B)
    
    # JSON repair and parsing
    pred_json = repair_json(pred_text)
    
    if pred_json is None:
        if score_A == 0:
            score_A = "None-Model"
        if score_B == 0:
            score_B = "None-Model"
        return (score_A, score_B)
    
    try:
        pred_json = ast.literal_eval(pred_json)
    except Exception:
        if score_A == 0:
            score_A = "None-Model"
        if score_B == 0:
            score_B = "None-Model"
        return (score_A, score_B)
    
    if not isinstance(pred_json, dict):
        if score_A == 0:
            score_A = "None-Model"
        if score_B == 0:
            score_B = "None-Model"
        return (score_A, score_B)
    
    pred_A = pred_json.get("A", None)
    pred_B = pred_json.get("B", None)
    
    if pred_A is None:
        if score_A == 0:
            score_A = "None-Model"
    if pred_B is None:
        if score_B == 0:
            score_B = "None-Model"
    
    # Validate predictions
    if score_A == 0 and pred_A:
        if ("남" in pred_A) and ("여" in pred_A):
            score_A = "None-Model"
        if ("알수없음" in pred_A):
            score_A = "None-Model"
    
    if score_B == 0 and pred_B:
        if ("남" in pred_B) and ("여" in pred_B):
            score_B = "None-Model"
        if ("알수없음" in pred_B):
            score_B = "None-Model"
    
    # Score matching
    if score_A == 0:
        if "남" in gt_eval_A and "남" in pred_A:
            score_A = 1
        if "여" in gt_eval_A and "여" in pred_A:
            score_A = 1
    
    if score_B == 0:
        if "남" in gt_eval_B and "남" in pred_B:
            score_B = 1
        if "여" in gt_eval_B and "여" in pred_B:
            score_B = 1
    
    return (score_A, score_B)


def score_age_diff(pred, gt):
    """
    Calculate age difference comparison score
    
    Args:
        pred: Prediction
        gt: Ground truth
    
    Returns:
        int or str: Score (1/0) or "None-Human"/"None-Model"
    """
    pred_text = extract_brace_content(pred)
    
    try:
        gt_eval = ast.literal_eval(gt)['compare']
    except Exception:
        return "None-Human"
    
    if "Unknown" in gt_eval:
        return "None-Human"
    
    if pred_text is None:
        return "None-Model"
    
    pred_text = f"{{{pred_text}}}"
    pred_json = repair_json(pred_text)
    
    if pred_json is None:
        return 0
    
    # Compare operators
    pred_value_str = str(pred_json).strip()
    gt_value_str = str(gt_eval).strip()
    operators = [">", "<", "="]
    
    for op in operators:
        if op in gt_value_str and op in pred_value_str:
            return 1
    
    return 0


def score_gender(pred, gt):
    """
    Calculate gender prediction score
    
    Args:
        pred: Prediction
        gt: Ground truth
    
    Returns:
        tuple: (score_A, score_B)
    """
    pred_text = extract_brace_content(pred)
    
    try:
        gt_eval_A = ast.literal_eval(gt)['A']
        gt_eval_B = ast.literal_eval(gt)['B']
    except Exception:
        return ("None-Human", "None-Human")
    
    score_A = 0
    score_B = 0
    
    # Ground truth validation
    if ("남" in gt_eval_A) and ("여" in gt_eval_A):
        score_A = "None-Human"
    elif ("알수없음" in gt_eval_A):
        score_A = "None-Human"
    
    if ("남" in gt_eval_B) and ("여" in gt_eval_B):
        score_B = "None-Human"
    elif ("알수없음" in gt_eval_B):
        score_B = "None-Human"
    
    if pred_text is None:
        if score_A == 0:
            score_A = "None-Model"
        if score_B == 0:
            score_B = "None-Model"
        return (score_A, score_B)
    
    pred_json = repair_json(pred_text)
    
    if pred_json is None:
        if score_A == 0:
            score_A = "None-Model"
        if score_B == 0:
            score_B = "None-Model"
        return (score_A, score_B)
    
    try:
        pred_json = ast.literal_eval(pred_json)
    except Exception:
        if score_A == 0:
            score_A = "None-Model"
        if score_B == 0:
            score_B = "None-Model"
        return (score_A, score_B)
    
    if not isinstance(pred_json, dict):
        if score_A == 0:
            score_A = "None-Model"
        if score_B == 0:
            score_B = "None-Model"
        return (score_A, score_B)
    
    pred_A = pred_json.get("A", None)
    pred_B = pred_json.get("B", None)
    
    if pred_A is None:
        if score_A == 0:
            score_A = "None-Model"
    if pred_B is None:
        if score_B == 0:
            score_B = "None-Model"
    
    # Validate predictions
    if score_A == 0 and pred_A:
        if ("남" in pred_A) and ("여" in pred_A):
            score_A = "None-Model"
        if ("알수없음" in pred_A):
            score_A = "None-Model"
    
    if score_B == 0 and pred_B:
        if ("남" in pred_B) and ("여" in pred_B):
            score_B = "None-Model"
        if ("알수없음" in pred_B):
            score_B = "None-Model"
    
    # Score matching
    if score_A == 0:
        if "남" in gt_eval_A and "남" in pred_A:
            score_A = 1
        if "여" in gt_eval_A and "여" in pred_A:
            score_A = 1
    
    if score_B == 0:
        if "남" in gt_eval_B and "남" in pred_B:
            score_B = 1
        if "여" in gt_eval_B and "여" in pred_B:
            score_B = 1
    
    return (score_A, score_B)


def score_gender_diff(pred, gt):
    """
    Calculate gender difference comparison score
    
    Args:
        pred: Prediction
        gt: Ground truth
    
    Returns:
        int or str: Score (1/0) or "None-Human"/"None-Model"
    """
    pred_text = extract_brace_content(pred)
    
    try:
        gt_eval = ast.literal_eval(gt)['compare']
    except Exception:
        return "None-Human"
    
    if "Unknown" in gt_eval:
        return "None-Human"
    
    if pred_text is None:
        return "None-Model"
    
    pred_json = repair_json(pred_text)
    
    if pred_json is None:
        return 0
    
    # Compare operators
    pred_value_str = str(pred_json).strip()
    gt_value_str = str(gt_eval).strip()
    
    operators = ["!=", "=", "Unknown"]
    pred_value_str = decode_unicode_escape(pred_value_str)
    pred_value_str = pred_value_str.replace("다름", "!=")
    pred_value_str = pred_value_str.replace("같음", "=")
    
    for op in operators:
        if (op in pred_value_str) and (op in gt_value_str):
            return 1
    
    return 0


def score_relation(pred, gt):
    """
    Calculate relation prediction score
    
    Args:
        pred: Prediction
        gt: Ground truth
    
    Returns:
        int: Score (1/0)
    """
    return int(pred == gt)


def score_sub_relation(pred, gt):
    """
    Calculate sub-relation prediction score
    
    Args:
        pred: Prediction
        gt: Ground truth (DataFrame row)
    
    Returns:
        dict: Score for each dimension
    """
    score = {}
    pred = ast.literal_eval(pred)
    
    if isinstance(pred, list):
        pred = pred[0]
        pred = ast.literal_eval(pred)
    
    try:
        pred = {str(k).lower(): v.lower() for k, v in pred.items()}
    except Exception:
        pass
    
    gt = gt.to_dict()
    gt = {str(k).lower().split('_gold')[0]: v.lower() for k, v in gt.items()}
    
    for i in pred:
        pred_label = pred[i]
        gt_label = gt[i]
        pred_text = extract_brace_content(pred_label)
        
        if "unknown" in gt_label.lower():
            score[i] = "None-Human"
            continue
        
        if pred_text is None:
            score[i] = "None-Model"
            continue
        
        if isinstance(pred_text, float):
            score[i] = "None-Model"
            continue
        
        if i == 'hierarchy':
            gt_label = gt_label.replace("a>b", "hierarchical")
            gt_label = gt_label.replace("a<b", "hierarchical")
            gt_label = gt_label.replace("a=b", "equal")
            lists = ['hierarchical', 'equal']
            
            if gt_label not in lists:
                score[i] = "None-Human"
                print(f"Unknown gt_label: {gt_label}")
                continue
            
            for j in lists:
                if (j in gt_label) and (j in pred_text):
                    score[i] = 1
                    break
                elif (j in gt_label):
                    score[i] = 0
        
        elif i == 'intimacy':
            gt_label = gt_label.replace("not intimate", "XYZmate")
            gt_label = gt_label.replace("unintimate", "XYZmate")
            pred_text = pred_text.replace("not intimate", "XYZmate")
            pred_text = pred_text.replace("unintimate", "XYZmate")
            gt_label = gt_label.replace("intimate", "QWEmate")
            pred_text = pred_text.replace("intimate", "QWEmate")
            lists = ['QWEmate', 'XYZmate', 'neutral']
            
            if gt_label not in lists:
                score[i] = "None-Human"
                print(f"Unknown gt_label: {gt_label}")
                continue
            
            for j in lists:
                if (j in gt_label) and (j in pred_text):
                    score[i] = 1
                    break
                elif (j in gt_label):
                    score[i] = 0
        
        elif i == 'formality':
            gt_label = gt_label.replace("pleasure-oriented", "XYZmal")
            gt_label = gt_label.replace("task-oriented", "QWEmal")
            
            try:
                pred_text = pred_text.replace("informal", "XYZmal")
                pred_text = pred_text.replace("formal", "QWEmal")
            except Exception:
                pass
            
            lists = ['QWEmal', 'XYZmal', 'neutral']
            
            if gt_label not in lists:
                print(f"Unknown gt_label: {gt_label}")
                score[i] = "None-Human"
                continue
            
            for j in lists:
                if (j in gt_label) and (j in pred_text):
                    score[i] = 1
                    break
                elif (j in gt_label):
                    score[i] = 0
    
    return score
