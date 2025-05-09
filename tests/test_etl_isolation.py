import pytest
from etl.isolation_ingest import ingest_isolation_resistance_data
from models.isolation import IsolationResistance


def test_ingest_isolation_resistance_data(sample_isolation_data, in_memory_db):
    # Ingest the sample data
    ingest_isolation_resistance_data('data/raw/isolation/sample_isolation.csv', in_memory_db)
    
    # Query the database to check if data was ingested
    result = in_memory_db.query(IsolationResistance).all()
    
    # Assert that data was ingested
    assert len(result) > 0, "Data should be ingested into the database"
    
    # Assert data integrity
    for record in result:
        assert isinstance(record.id, int), "ID should be an integer"
        assert isinstance(record.value, float), "Value should be a float"
        assert isinstance(record.description, str), "Description should be a string"
    
    # Additional assertions can be added to verify data integrity
