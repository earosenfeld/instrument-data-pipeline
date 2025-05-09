import pytest
from etl.hipot_ingest import ingest_hipot_data
from models.hipot import HiPotData


def test_ingest_hipot_data(sample_hipot_data, in_memory_db):
    # Ingest the sample data
    ingest_hipot_data('data/raw/hipot/sample_hipot.csv', in_memory_db)
    
    # Query the database to check if data was ingested
    result = in_memory_db.query(HiPotData).all()
    
    # Assert that data was ingested
    assert len(result) > 0, "Data should be ingested into the database"
    
    # Assert data integrity
    for record in result:
        assert isinstance(record.id, int), "ID should be an integer"
        assert isinstance(record.value, float), "Value should be a float"
        assert isinstance(record.description, str), "Description should be a string"
    
    # Additional assertions can be added to verify data integrity
