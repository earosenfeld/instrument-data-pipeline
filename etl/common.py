# common.py
# This module contains common functions and utilities for ETL processes.

import pandas as pd
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Required columns per test type. Normalization guarantees these exist in the
# output frame (filling sensible defaults for missing ``test_time`` / ``result``)
# without the lossy many-to-one column aliasing the old mapping used.
REQUIRED_COLUMNS = {
    'burnin': ['id', 'value'],
    'hipot': ['id', 'voltage', 'current', 'test_time', 'result'],
    'isolation': ['id', 'resistance', 'voltage', 'test_time', 'result'],
    'laser': ['id', 'power', 'wavelength', 'test_time', 'result'],
    'parametric': ['id', 'voltage', 'current', 'test_time', 'result'],
    'ict': ['id', 'voltage', 'current', 'test_time', 'result'],
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
    """Normalize raw test data into the canonical schema for ``test_type``.

    Real (non-no-op) behaviour:
      * Aliases a legacy ``timestamp`` column to ``test_time``.
      * Fills a default ``test_time`` (now) when the schema needs one and it is
        absent.
      * Fills a default ``result`` of ``'PASS'`` when required and absent.
      * Validates that the measurement columns required by the test type are
        present, raising ``ValueError`` otherwise (so malformed inputs fail loudly
        rather than silently passing through unchanged).

    Args:
        df: Input data.
        test_type: One of ``REQUIRED_COLUMNS`` keys.

    Returns:
        pd.DataFrame: Normalized data with all required columns present.
    """
    try:
        required = REQUIRED_COLUMNS.get(test_type, REQUIRED_COLUMNS['burnin'])
        normalized_df = df.copy()

        # Alias legacy 'timestamp' -> 'test_time'.
        if 'timestamp' in normalized_df.columns and 'test_time' not in normalized_df.columns:
            normalized_df['test_time'] = normalized_df['timestamp']

        # Fill defaults for bookkeeping columns the schema requires.
        if 'test_time' in required and 'test_time' not in normalized_df.columns:
            normalized_df['test_time'] = pd.Timestamp.now()
        if 'result' in required and 'result' not in normalized_df.columns:
            normalized_df['result'] = 'PASS'

        # Measurement columns must be supplied by the source data.
        measurement_cols = [
            c for c in required if c not in ('id', 'test_time', 'result')
        ]
        missing = [c for c in measurement_cols if c not in normalized_df.columns]
        if missing:
            raise ValueError(
                f"{test_type} data missing required columns: {missing}"
            )

        logger.info(f"Successfully normalized data for {test_type} test")
        return normalized_df

    except Exception as e:
        logger.error(f"Error normalizing data for {test_type} test: {str(e)}")
        raise
