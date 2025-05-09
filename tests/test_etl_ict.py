import pytest
from etl.common import load_data, normalize_data


def test_ict_etl_process(sample_ict_data, in_memory_db):
    # Load the sample data
    df = load_data('data/raw/ict/sample_ict.csv')
    
    # Normalize the data
    normalized_df = normalize_data(df)
    
    # Example assertion: Check if the data is loaded correctly
    assert not df.empty, "Data should be loaded"
    
    # Example assertion: Check if the normalization process works
    assert normalized_df.equals(df), "Normalization should not alter the data in this placeholder test"
    
    # Additional assertions can be added to verify the ETL process
