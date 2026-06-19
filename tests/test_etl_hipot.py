import pytest
from etl.hipot_ingest import ingest_hipot_data
from models.hipot import HiPotData


def test_ingest_hipot_data(in_memory_db):
    ingest_hipot_data('data/raw/hipot/sample_hipot.csv', in_memory_db)

    result = in_memory_db.query(HiPotData).all()
    assert len(result) > 0, "Data should be ingested into the database"

    for record in result:
        assert isinstance(record.id, int), "ID should be an integer"
        assert isinstance(record.voltage, float), "Voltage should be a float"
        assert isinstance(record.current, float), "Current should be a float"
        assert record.result in ("PASS", "FAIL"), "Result should be PASS or FAIL"
        assert isinstance(record.description, str), "Description should be a string"
        # Physical sanity: leakage current is non-negative.
        assert record.current >= 0
