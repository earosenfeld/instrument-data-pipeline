# common.py
# This module contains common functions and utilities for ETL processes.

import pandas as pd
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Column mappings for different test types
COLUMN_MAPPINGS = {
    'burnin': {
        'id': 'id',
        'value': 'value'
    },
    'hipot': {
        'id': 'id',
        'voltage': 'value',
        'current': 'value',
        'test_time': 'timestamp',
        'result': 'result'
    },
    'isolation': {
        'id': 'id',
        'resistance': 'value',
        'voltage': 'value',
        'test_time': 'timestamp',
        'result': 'result'
    },
    'laser': {
        'id': 'id',
        'power': 'value',
        'wavelength': 'value',
        'test_time': 'timestamp',
        'result': 'result'
    },
    'parametric': {
        'id': 'id',
        'voltage': 'value',
        'current': 'value',
        'test_time': 'timestamp',
        'result': 'result'
    }
}

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded data
    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded data from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {str(e)}")
        raise

def normalize_data(df: pd.DataFrame, test_type: str = 'burnin') -> pd.DataFrame:
    """
    Normalize data according to the test type.
    
    Args:
        df (pd.DataFrame): Input data
        test_type (str): Type of test (burnin, hipot, isolation, laser, parametric)
        
    Returns:
        pd.DataFrame: Normalized data
    """
    try:
        # Get column mapping for the test type
        mapping = COLUMN_MAPPINGS.get(test_type, COLUMN_MAPPINGS['burnin'])
        
        # Create a copy of the dataframe
        normalized_df = df.copy()
        
        # Map columns according to the test type
        for target_col, source_col in mapping.items():
            if source_col in normalized_df.columns:
                normalized_df[target_col] = normalized_df[source_col]
        
        # Add timestamp if not present
        if 'test_time' in mapping and 'test_time' not in normalized_df.columns:
            normalized_df['test_time'] = pd.Timestamp.now()
        
        # Add result if not present
        if 'result' in mapping and 'result' not in normalized_df.columns:
            normalized_df['result'] = 'PASS'
        
        logger.info(f"Successfully normalized data for {test_type} test")
        return normalized_df
        
    except Exception as e:
        logger.error(f"Error normalizing data for {test_type} test: {str(e)}")
        raise
