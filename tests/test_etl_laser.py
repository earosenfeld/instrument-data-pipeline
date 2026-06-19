import pytest
from etl.laser_ingest import ingest_laser_profile_data
from models.laser import LaserProfile


def test_ingest_laser_profile_data(in_memory_db):
    ingest_laser_profile_data('data/raw/laser/sample_laser.csv', in_memory_db)

    result = in_memory_db.query(LaserProfile).all()
    assert len(result) > 0, "Data should be ingested into the database"

    powers, wavelengths = [], []
    for record in result:
        assert isinstance(record.id, int), "ID should be an integer"
        assert isinstance(record.power, float), "Power should be a float"
        assert isinstance(record.wavelength, float), "Wavelength should be a float"
        assert record.result in ("PASS", "FAIL"), "Result should be PASS or FAIL"
        powers.append(record.power)
        wavelengths.append(record.wavelength)

    # Regression guard for the original bug where wavelength was literally the
    # same array as power. They must be distinct, independent measurements.
    assert powers != wavelengths, "power and wavelength must not be identical"
