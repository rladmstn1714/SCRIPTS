"""
Path utilities for getting SCRIPTS root directory from anywhere
"""
from pathlib import Path


def get_scripts_root():
    """
    Get SCRIPTS root directory from any location in the project
    
    This function walks up the directory tree until it finds the SCRIPTS directory
    (identified by having an 'experiment' and 'dataset' subdirectory)
    
    Returns:
        Path: Absolute path to SCRIPTS root directory
    """
    # Start from this file's location
    current = Path(__file__).resolve().parent
    
    # Walk up until we find SCRIPTS root (has experiment and dataset dirs)
    while current != current.parent:
        if (current / 'experiment').exists() and (current / 'dataset').exists():
            return current
        current = current.parent
    
    # Fallback: assume we're in experiment/utils/ and go up two levels
    return Path(__file__).resolve().parent.parent.parent


SCRIPTS_ROOT = get_scripts_root()

Path utilities for getting SCRIPTS root directory from anywhere
"""
from pathlib import Path


def get_scripts_root():
    """
    Get SCRIPTS root directory from any location in the project
    
    This function walks up the directory tree until it finds the SCRIPTS directory
    (identified by having an 'experiment' and 'dataset' subdirectory)
    
    Returns:
        Path: Absolute path to SCRIPTS root directory
    """
    # Start from this file's location
    current = Path(__file__).resolve().parent
    
    # Walk up until we find SCRIPTS root (has experiment and dataset dirs)
    while current != current.parent:
        if (current / 'experiment').exists() and (current / 'dataset').exists():
            return current
        current = current.parent
    
    # Fallback: assume we're in experiment/utils/ and go up two levels
    return Path(__file__).resolve().parent.parent.parent


SCRIPTS_ROOT = get_scripts_root()

