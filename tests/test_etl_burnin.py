import pytest
from etl.burnin_ingest import ingest_burnin_zero_current_data
from models.burnin import BurnInZeroCurrent


def test_ingest_burnin_zero_current_data(in_memory_db):
    # Ingest the sample data.
    ingest_burnin_zero_current_data('data/raw/burnin/sample_burnin.csv', in_memory_db)

    result = in_memory_db.query(BurnInZeroCurrent).all()
    assert len(result) > 0, "Data should be ingested into the database"

    for record in result:
        assert isinstance(record.id, int), "ID should be an integer"
        assert isinstance(record.value, float), "Value should be a float"
        assert isinstance(record.description, str), "Description should be a string"
        assert 0 <= record.value <= 1000, "Value should be within expected range"
        assert record.description.startswith("Burn-in Test")

    # Empty file: ingests zero rows without error.
    before = in_memory_db.query(BurnInZeroCurrent).count()
    ingest_burnin_zero_current_data('data/raw/burnin/empty.csv', in_memory_db)
    after = in_memory_db.query(BurnInZeroCurrent).count()
    assert after == before, "No data should be ingested from an empty file"

    # Malformed file: non-numeric / ragged rows must raise.
    with pytest.raises(Exception):
        ingest_burnin_zero_current_data('data/raw/burnin/malformed.csv', in_memory_db)
