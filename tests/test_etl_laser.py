import pytest
from etl.laser_ingest import ingest_laser_profile_data
from models.laser import LaserProfile


def test_ingest_laser_profile_data(sample_laser_data, in_memory_db):
    # Ingest the sample data
    ingest_laser_profile_data('data/raw/laser/sample_laser.csv', in_memory_db)
    
    # Query the database to check if data was ingested
    result = in_memory_db.query(LaserProfile).all()
    
    # Assert that data was ingested
    assert len(result) > 0, "Data should be ingested into the database"
    
    # Assert data integrity
    for record in result:
        assert isinstance(record.id, int), "ID should be an integer"
        assert isinstance(record.value, float), "Value should be a float"
        assert isinstance(record.description, str), "Description should be a string"
    
    # Additional assertions can be added to verify data integrity
