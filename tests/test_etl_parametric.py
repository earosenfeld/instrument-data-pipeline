import pytest
from etl.parametric_ingest import ingest_parametric_data
from models.parametric import ParametricData


def test_ingest_parametric_data(in_memory_db):
    ingest_parametric_data('data/raw/parametric/sample_parametric.csv', in_memory_db)

    result = in_memory_db.query(ParametricData).all()
    assert len(result) > 0, "Data should be ingested into the database"

    for record in result:
        assert isinstance(record.id, int), "ID should be an integer"
        assert isinstance(record.voltage, float), "Voltage should be a float"
        assert isinstance(record.current, float), "Current should be a float"
        assert record.result in ("PASS", "FAIL"), "Result should be PASS or FAIL"
