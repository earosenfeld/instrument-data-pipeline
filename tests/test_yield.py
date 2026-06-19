"""Validation tests for yield analytics on known pass/fail data."""

import math

import pandas as pd
import pytest

from etl.yield_analysis import (
    first_pass_yield,
    yield_by,
    dpmo,
    pareto_failure_modes,
    yield_summary,
)


def _known_frame():
    # 4 units, 2 parameters each (8 rows). Units 1,2 fully pass; unit 3 fails one
    # parameter; unit 4 fails one parameter.
    return pd.DataFrame(
        {
            "unit_id": [1, 1, 2, 2, 3, 3, 4, 4],
            "parameter": ["A", "B", "A", "B", "A", "B", "A", "B"],
            "station": ["S1", "S1", "S2", "S2", "S1", "S1", "S2", "S2"],
            "passed": [True, True, True, True, False, True, True, False],
            "failure_mode": ["", "", "", "", "A_low", "", "", "B_high"],
        }
    )


def test_first_pass_yield_known():
    df = _known_frame()
    # 2 of 4 units pass every test -> FPY = 0.5
    assert first_pass_yield(df) == pytest.approx(0.5)


def test_yield_by_parameter_known():
    df = _known_frame()
    by_param = yield_by(df, "parameter")
    # Parameter A: units 1,2,4 pass, unit 3 fails -> 3/4 = 0.75
    # Parameter B: units 1,2,3 pass, unit 4 fails -> 3/4 = 0.75
    assert by_param["A"] == pytest.approx(0.75)
    assert by_param["B"] == pytest.approx(0.75)


def test_yield_by_station_known():
    df = _known_frame()
    by_station = yield_by(df, "station")
    # S1 rows: T,T,F,T -> 3/4 ; S2 rows: T,T,T,F -> 3/4
    assert by_station["S1"] == pytest.approx(0.75)
    assert by_station["S2"] == pytest.approx(0.75)


def test_dpmo_known():
    # 2 defects, 4 units, 2 opportunities -> 2/8*1e6 = 250000 DPMO.
    r = dpmo(defects=2, units=4, opportunities=2)
    assert r.dpmo == pytest.approx(250000.0)
    assert r.yield_fraction == pytest.approx(0.75)
    assert r.opportunities == 2


def test_dpmo_six_sigma_reference():
    # 3.4 DPMO is the classic 6-sigma figure; sigma_level ~ 6.0.
    r = dpmo(defects=34, units=10_000_000, opportunities=1)
    assert r.dpmo == pytest.approx(3.4, abs=1e-6)
    assert r.sigma_level == pytest.approx(6.0, abs=0.05)


def test_dpmo_rejects_bad_args():
    with pytest.raises(ValueError):
        dpmo(defects=1, units=0, opportunities=1)
    with pytest.raises(ValueError):
        dpmo(defects=-1, units=1, opportunities=1)


def test_pareto_counts_and_cumulative():
    modes = ["open", "open", "open", "short", "short", "tolerance"]
    p = pareto_failure_modes(modes)
    # Descending by count: open(3), short(2), tolerance(1).
    assert list(p.index) == ["open", "short", "tolerance"]
    assert list(p["count"]) == [3, 2, 1]
    assert p["percent"].iloc[0] == pytest.approx(50.0)
    # Cumulative reaches 100% at the last row.
    assert p["cumulative_percent"].iloc[-1] == pytest.approx(100.0)
    assert p["cumulative_percent"].iloc[1] == pytest.approx(50.0 + 100.0 / 3.0)


def test_pareto_empty():
    assert pareto_failure_modes([]).empty


def test_yield_summary_rollup():
    df = _known_frame()
    s = yield_summary(df)
    assert s["first_pass_yield"] == pytest.approx(0.5)
    assert s["dpmo"]["dpmo"] == pytest.approx(250000.0)
    assert "yield_by_parameter" in s
    assert "pareto" in s
    # Two distinct failure modes recorded.
    assert len(s["pareto"]) == 2
