"""
Scoring module for social reasoning evaluation

This module provides:
- Universal scoring system for flexible evaluation
- Parser utilities for extracting structured information
"""
from .scoring import (
    parse_json_string,
    extract_relation_from_generated,
    find_relation_match,
    calculate_scores_for_file,
    process_scoring_task
)
from .parsers import (
    clean_json_string,
    parse_age_gender_json,
    parse_sub_relation_json
)

__all__ = [
    # Universal scoring system
    'parse_json_string',
    'extract_relation_from_generated',
    'find_relation_match',
    'calculate_scores_for_file',
    'process_scoring_task',
    
    # Parsers
    'clean_json_string',
    'parse_age_gender_json',
    'parse_sub_relation_json',
]
