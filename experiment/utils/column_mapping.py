"""
Column name standardization utilities

This module provides mapping between different column name formats
to ensure consistent access across all scripts.

Standard format follows English CSV convention with hyphens.
"""

# Standard column names (based on English CSV format)
STANDARD_COLUMNS = {
    # Core columns
    'dialogue': 'dialogue',
    'scene_id': 'scene_id',
    
    # Relation columns
    'relation_high_probable_gold': 'relation_high_probable_gold',
    'relation_impossible_gold': 'relation_impossible_gold',
    'high_probable_agreement': 'high_probable_agreement',
    
    # Sub-relation columns (all with _gold suffix)
    'intimacy_gold': 'intimacy_gold',
    'formality_gold': 'formality_gold',
    'hierarchy_gold': 'hierarchy_gold',
    'intimacy_agreement': 'intimacy_agreement',
    'formality_agreement': 'formality_agreement',
    'hierarchy_agreement': 'hierarchy_agreement',
    
    # Age/Gender columns (all with _gold suffix)
    'age-a_gold': 'age-a_gold',
    'age-b_gold': 'age-b_gold',
    'age_diff_gold': 'age_diff_gold',
    'gender-a_gold': 'gender-a_gold',
    'gender-b_gold': 'gender-b_gold',
    'gender_diff_gold': 'gender_diff_gold',
}

# Mapping from various formats to standard names (English CSV format)
COLUMN_ALIASES = {
    # Dialogue
    'final_diag': 'dialogue',
    
    # Relation
    'relation_best': 'relation_high_probable_gold',
    'relation_1': 'relation_high_probable_gold',
    
    # Age columns - map old formats to standard hyphen format
    'Age_A': 'age-a',
    'Age_B': 'age-b',
    'Age_compare': 'age-diff',
    'age_a': 'age-a',
    'age_b': 'age-b',
    'age_diff': 'age-diff',
    'age_a_gold': 'age-a_gold',
    'age_b_gold': 'age-b_gold',
    
    # Gender columns - map old formats to standard hyphen format
    'Gender_A': 'gender-a',
    'Gender_B': 'gender-b',
    'Gender_compare': 'gender-diff',
    'gender_a': 'gender-a',
    'gender_b': 'gender-b',
    'gender_diff': 'gender-diff',
    'gender_a_gold': 'gender-a_gold',
    'gender_b_gold': 'gender-b_gold',
    
    # Sub-relations
    'Hierarchy': 'hierarchy',
    'Intimacy': 'intimacy',
    'Pleasure': 'formality',
    'Hierarchy_agreement': 'hierarchy_agreement',
    'Intimacy_agreement': 'intimacy_agreement',
    'Pleasure_agreement': 'formality_agreement',
}


def standardize_column_names(df):
    """
    Standardize column names in a dataframe to English CSV format
    
    Args:
        df: pandas DataFrame
        
    Returns:
        DataFrame: DataFrame with standardized column names
    """
    rename_map = {}
    
    for col in df.columns:
        if col in COLUMN_ALIASES:
            rename_map[col] = COLUMN_ALIASES[col]
    
    if rename_map:
        df = df.rename(columns=rename_map)
        print(f"Standardized {len(rename_map)} column names: {rename_map}")
    
    return df


def get_column(df, standard_name, default=''):
    """
    Get column value using standard name, with fallback to aliases
    
    Args:
        df: pandas DataFrame or Series (row)
        standard_name: Standard column name
        default: Default value if column not found
        
    Returns:
        Column value or default
    """
    # Try standard name first
    if standard_name in df:
        return df[standard_name]
    
    # Try aliases
    for alias, std in COLUMN_ALIASES.items():
        if std == standard_name and alias in df:
            return df[alias]
    
    return default

