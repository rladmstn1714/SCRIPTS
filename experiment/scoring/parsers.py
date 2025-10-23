"""
Parser utilities for extracting structured information from model outputs

This module provides functions for parsing:
- Age and gender information
- Sub-relations (intimacy, formality, hierarchy)
- Relations
"""
import pandas as pd
import json
import re


def clean_json_string(json_str):
    """Clean JSON string by removing unnecessary characters"""
    # Remove multiple closing braces at the end
    json_str = re.sub(r'[}]+$', '}', json_str)
    
    # Normalize quotes
    json_str = re.sub(r"'([^']*)'", r'"\1"', json_str)
    
    # Normalize keys
    json_str = re.sub(r'([{,])\s*([A-Z])\s*:', r'\1"\2":', json_str)
    json_str = re.sub(r'([{,])\s*([a-zA-Z]+)\s*:', r'\1"\2":', json_str)
    
    # Remove trailing commas
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    
    return json_str


def parse_json_string(json_str):
    """Parse JSON string with error handling"""
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


def parse_age_gender_json(text):
    """
    Parse age and gender information from text
    
    Expected pattern: {"A": {"Age": "...", "Gender": "..."}, "B": {"Age": "...", "Gender": "..."}}
    
    Args:
        text: Text containing JSON with age/gender information
        
    Returns:
        dict: Parsed age/gender information or None if parsing fails
    """
    if not text or pd.isna(text):
        return None
    
    # Try extracting from JSON block
    json_pattern = r'```json\s*(\{.*?\})\s*```'
    matches = re.findall(json_pattern, text, re.DOTALL)
    
    if matches:
        try:
            json_str = matches[0]
            json_str = clean_json_string(json_str)
            parsed = json.loads(json_str)
            
            # Check required keys
            if 'A' in parsed and 'B' in parsed:
                if ('Age' in parsed['A'] and 'Gender' in parsed['A'] and 
                    'Age' in parsed['B'] and 'Gender' in parsed['B']):
                    return parsed
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    
    # Try finding JSON object directly
    json_patterns = [
        r'\{[^{}]*"A"[^{}]*"Age"[^{}]*"Gender"[^{}]*"B"[^{}]*"Age"[^{}]*"Gender"[^{}]*\}',
        r"\{'A':\s*\{[^}]*\},\s*'B':\s*\{[^}]*\}\}"
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        if matches:
            try:
                json_str = clean_json_string(matches[0])
                parsed = json.loads(json_str)
                
                if 'A' in parsed and 'B' in parsed:
                    if ('Age' in parsed['A'] and 'Gender' in parsed['A'] and 
                        'Age' in parsed['B'] and 'Gender' in parsed['B']):
                        return parsed
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    
    return None


def parse_sub_relation_json(text):
    """
    Parse sub-relation information (intimacy, formality, hierarchy) from text
    
    Expected pattern: {"intimacy": "...", "formality": "...", "hierarchy": "..."}
    
    Args:
        text: Text containing JSON with sub-relation information
        
    Returns:
        dict: Parsed sub-relation information or None if parsing fails
    """
    if not text or pd.isna(text):
        return None
    
    # Try extracting from JSON block
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        try:
            json_str = clean_json_string(json_match.group(1))
            parsed = json.loads(json_str)
            
            # Check for sub-relation keys
            if any(key in parsed for key in ['intimacy', 'formality', 'hierarchy']):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
    
    # Try general JSON parsing
    try:
        parsed = json.loads(text)
        if any(key in parsed for key in ['intimacy', 'formality', 'hierarchy']):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    
    return None


def extract_relation_from_generated(generated_text):
    """
    Extract relation from generated text
    
    Args:
        generated_text: Generated text containing relation information
        
    Returns:
        str: Extracted relation or None if not found
    """
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

