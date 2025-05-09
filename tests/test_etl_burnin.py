import pytest
from etl.burnin_ingest import ingest_burnin_zero_current_data
from models.burnin import BurnInZeroCurrent


def test_ingest_burnin_zero_current_data(sample_burnin_data, in_memory_db):
    # Ingest the sample data
    ingest_burnin_zero_current_data('data/raw/burnin/sample_burnin.csv', in_memory_db)
    
    # Query the database to check if data was ingested
    result = in_memory_db.query(BurnInZeroCurrent).all()
    
    # Assert that data was ingested
    assert len(result) > 0, "Data should be ingested into the database"
    
    # Assert data integrity
    for record in result:
        assert isinstance(record.id, int), "ID should be an integer"
        assert isinstance(record.value, float), "Value should be a float"
        assert isinstance(record.description, str), "Description should be a string"
        assert 0 <= record.value <= 1000, "Value should be within expected range (0-1000)"
        assert record.description.startswith("Burn-in Test"), "Description should start with 'Burn-in Test'"
    
    # Test with empty data
    ingest_burnin_zero_current_data('data/raw/burnin/empty.csv', in_memory_db)
    empty_result = in_memory_db.query(BurnInZeroCurrent).filter_by(description='Empty Test').all()
    assert len(empty_result) == 0, "No data should be ingested from an empty file"
    
    # Test with malformed data
    with pytest.raises(Exception):
        ingest_burnin_zero_current_data('data/raw/burnin/malformed.csv', in_memory_db)
    
    # Additional assertions can be added to verify data integrity
