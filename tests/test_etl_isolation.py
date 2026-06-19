import pytest
from etl.isolation_ingest import ingest_isolation_resistance_data
from models.isolation import IsolationResistance


def test_ingest_isolation_resistance_data(in_memory_db):
    ingest_isolation_resistance_data(
        'data/raw/isolation/sample_isolation.csv', in_memory_db
    )

    result = in_memory_db.query(IsolationResistance).all()
    assert len(result) > 0, "Data should be ingested into the database"

    for record in result:
        assert isinstance(record.id, int), "ID should be an integer"
        assert isinstance(record.resistance, float), "Resistance should be a float"
        assert isinstance(record.voltage, float), "Voltage should be a float"
        assert record.result in ("PASS", "FAIL"), "Result should be PASS or FAIL"
        # Resistance is strictly positive (log-normal).
        assert record.resistance > 0
