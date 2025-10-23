"""
Utility functions module
"""
import json
import pandas as pd
import itertools
import csv
import os
import re
import ast
from json_repair import repair_json


def open_json(json_file_path):
    """
    Open JSON file
    
    Args:
        json_file_path: JSON file path
    
    Returns:
        dict: JSON data
    """
    with open(json_file_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)
    return json_data


def open_csv(file_path):
    """
    Open CSV file
    
    Args:
        file_path: CSV file path
    
    Returns:
        DataFrame: CSV data
    """
    df = pd.read_csv(file_path)
    return df


def save_csv(df, file_path):
    """
    Save CSV file
    
    Args:
        df: Dataframe
        file_path: Save path
    
    Returns:
        DataFrame: Saved dataframe
    """
    df.to_csv(file_path, index=False)
    return df


def iterative_tuple_generator(num):
    """
    Generate combination tuples
    
    Args:
        num: Number to combine
    
    Returns:
        list: List of combination tuples
    """
    return list(itertools.combinations(range(num), 2))


def write_csv_row(values, filename):
    """
    Append row to CSV file
    
    Args:
        values: Values to append (list or dict)
        filename: CSV filename
    """
    open_trial = 0
    
    while True:
        if open_trial > 10:
            raise Exception("Failed to open file after 10 attempts")

        try:
            if isinstance(values, list):
                with open(filename, "a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(values)
            elif isinstance(values, dict):
                df = pd.DataFrame([values])
                csv_columns = pd.read_csv(filename, nrows=0).columns
                # Reorder the DataFrame's columns to match the CSV
                df = df.reindex(columns=csv_columns, fill_value=None)
                # Append to CSV
                df.to_csv(filename, mode='a', header=False, index=False)
            break
        except Exception as e:
            print(f"Failed to write to CSV: {e}")
            open_trial += 1
            continue


def find_csv_files(directory, ends):
    """
    Find CSV files ending with specific pattern
    
    Args:
        directory: Directory to search
        ends: File name ending pattern (list)
    
    Returns:
        list: List of found file paths
    """
    split_csv_files = []
    len_end = len(ends)
    
    if len_end == 1:
        for root, _, files in os.walk(directory):
            for file in files:
                if ends[0] in file:
                    full_path = os.path.join(root, file)
                    split_csv_files.append(full_path)
    elif len_end == 2:
        for root, _, files in os.walk(directory):
            for file in files:
                if (ends[0] in file) and (ends[1] in file):
                    full_path = os.path.join(root, file)
                    split_csv_files.append(full_path)
    
    return split_csv_files


def accuracy(val1, val2):
    """
    Calculate accuracy
    
    Args:
        val1: List of predictions
        val2: List of ground truth
    
    Returns:
        float: Accuracy
    """
    cnt = 0
    total = 0
    
    if len(val1) != len(val2):
        return -1
    
    for idx, i in enumerate(val2):
        try:
            total += 1
            if val1[idx] in val2[idx]:
                cnt += 1
        except:
            pass
    
    return cnt / total if total > 0 else 0


def extract_json(text):
    """
    Extract and parse JSON from text
    
    Args:
        text: Text containing JSON
    
    Returns:
        dict or str: Parsed JSON or original text
    """
    text = text.replace("\n", "")
    text = text.split('```json')[-1]
    text = text.split('```')[0]
    text = f"{text}"
    text = repair_json(text)
    
    try:
        output = json.loads(text)
        return output
    except:
        pass
    
    try:
        output = json.loads(re.sub(r"(?<=\{|\s)'|(?<=\s|:)'|(?<=\d)'(?!:)|'(?=\s|,|}|:)", '"', text))
        return output
    except:
        pass
    
    try:
        output = ast.literal_eval(text)
        return output
    except:
        pass
    
    return text


def extract_brace_content(text):
    """
    Extract content within braces
    
    Args:
        text: Text
    
    Returns:
        str: Extracted brace content
    """
    pattern = r"(\{.*?\})"
    text = str(text)
    matches = re.findall(pattern, text, re.DOTALL)
    return matches[-1] if matches else None


def decode_unicode_escape(s):
    """
    Decode unicode escape
    
    Args:
        s: Input string
    
    Returns:
        str: Decoded string
    """
    if isinstance(s, str):
        return bytes(s, "utf-8").decode("unicode_escape")
    return s
