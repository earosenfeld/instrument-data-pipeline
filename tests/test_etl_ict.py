import pytest
import pandas as pd
from etl.common import load_data, normalize_data


def test_ict_etl_normalizes_data():
    df = load_data('data/raw/ict/sample_ict.csv')
    assert not df.empty, "Data should be loaded"

    normalized_df = normalize_data(df, test_type='ict')

    # Real (non-no-op) normalization behaviour: required schema columns present.
    for col in ('id', 'voltage', 'current', 'test_time', 'result'):
        assert col in normalized_df.columns, f"{col} should exist after normalization"

    # Original measurement values are preserved, not mangled.
    pd.testing.assert_series_equal(
        normalized_df['voltage'], df['voltage'], check_names=False
    )


def test_normalize_fills_defaults_when_missing():
    # A frame without test_time/result gets them filled.
    df = pd.DataFrame({'id': [1, 2], 'voltage': [3.3, 3.4], 'current': [0.5, 0.6]})
    out = normalize_data(df, test_type='ict')
    assert 'test_time' in out.columns
    assert (out['result'] == 'PASS').all()


def test_normalize_raises_on_missing_measurement_column():
    # Missing a required measurement column must fail loudly (not silently pass).
    df = pd.DataFrame({'id': [1], 'voltage': [3.3]})  # no 'current'
    with pytest.raises(ValueError):
        normalize_data(df, test_type='ict')
