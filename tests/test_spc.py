"""Validation tests for SPC charts and run-rule detection.

Includes the required planted rule-1 and rule-4 violations flagged at the correct
indices, X-bar/R limit math against hand calculations, and the guarantee that the
R-chart LCL is never negative.
"""

import numpy as np
import pytest

from etl.spc import (
    CONTROL_CONSTANTS,
    imr_chart,
    xbar_r_chart,
    western_electric_rules,
    nelson_trend_rules,
    detect_violations,
    ewma_drift,
    cusum_drift,
)


def test_xbar_r_limits_hand_computed():
    # Two subgroups of size 3:
    #   g1 = [10, 12, 11] -> mean 11, range 2
    #   g2 = [9, 11, 10]  -> mean 10, range 2
    # Xbarbar = 10.5, Rbar = 2.  n=3 -> A2=1.023, D4=2.574, D3=0.0
    chart = xbar_r_chart([[10, 12, 11], [9, 11, 10]])
    c = CONTROL_CONSTANTS[3]
    assert chart.xbar.center == pytest.approx(10.5)
    assert chart.r.center == pytest.approx(2.0)
    assert chart.xbar.ucl == pytest.approx(10.5 + c["A2"] * 2.0)
    assert chart.xbar.lcl == pytest.approx(10.5 - c["A2"] * 2.0)
    assert chart.r.ucl == pytest.approx(c["D4"] * 2.0)
    # D3=0 for n=3 -> R LCL is exactly 0, never negative.
    assert chart.r.lcl == 0.0


def test_r_chart_lcl_never_negative_small_n():
    # For n<=6, D3=0, so even with large Rbar the LCL stays at 0.
    chart = xbar_r_chart([[1, 9], [2, 8], [0, 10]])  # n=2
    assert chart.r.lcl == 0.0
    assert chart.r.lcl >= 0.0


def test_r_chart_lcl_positive_large_n():
    # n=10 has D3=0.223 -> a nonzero positive LCL.
    groups = [list(np.arange(10) + i) for i in range(5)]
    chart = xbar_r_chart(groups)
    assert chart.subgroup_size == 10
    assert chart.r.lcl > 0.0


def test_imr_chart_limits():
    # Constant-ish series; individuals limits use E2=2.66 * MRbar.
    x = [10.0, 10.5, 10.2, 9.9, 10.1]
    chart = imr_chart(x)
    mr = np.abs(np.diff(x))
    mr_bar = mr.mean()
    expected_ucl = np.mean(x) + (3.0 / 1.128) * mr_bar
    assert chart.individuals.ucl == pytest.approx(expected_ucl)
    assert chart.moving_range.lcl == 0.0  # D3(2)=0


def test_rule1_planted_violation_index():
    # 19 in-control points, one gross spike at index 10 -> rule 1 fires there.
    x = np.zeros(20)
    x[10] = 100.0  # way beyond 3 sigma
    # center=0, sigma chosen so 100 is clearly >3 sigma.
    hits = western_electric_rules(x, center=0.0, sigma=1.0)
    rule1 = [idx for idx, rule in hits if rule == 1]
    assert rule1 == [10]


def test_rule4_planted_violation_index():
    # 8 consecutive points above center starting so the 8th is index 7.
    # Points 0..7 above center, rest below.
    x = np.array([1, 1, 1, 1, 1, 1, 1, 1, -1, -1], dtype=float)
    hits = western_electric_rules(x, center=0.0, sigma=1.0)
    rule4 = [idx for idx, rule in hits if rule == 4]
    # The run of 8 same-side completes at index 7.
    assert 7 in rule4
    # No earlier index can satisfy an 8-in-a-row.
    assert min(rule4) == 7


def test_rule2_two_of_three_beyond_2sigma():
    # Points at +2.5 sigma at indices 0 and 2 -> 2 of 3 same side fires at idx 2.
    x = np.array([2.5, 0.0, 2.5, 0.0], dtype=float)
    hits = western_electric_rules(x, center=0.0, sigma=1.0)
    rule2 = [idx for idx, rule in hits if rule == 2]
    assert 2 in rule2


def test_rule3_four_of_five_beyond_1sigma():
    # Four points >1 sigma within a window of five -> rule 3 at index 4.
    x = np.array([1.5, 1.5, 0.0, 1.5, 1.5], dtype=float)
    hits = western_electric_rules(x, center=0.0, sigma=1.0)
    rule3 = [idx for idx, rule in hits if rule == 3]
    assert 4 in rule3


def test_nelson_rule6_increasing_trend():
    # Strictly increasing 6-in-a-row -> rule 6 fires at index 5.
    x = np.array([1, 2, 3, 4, 5, 6, 5], dtype=float)
    hits = nelson_trend_rules(x)
    rule6 = [idx for idx, rule in hits if rule == 6]
    assert 5 in rule6


def test_nelson_rule14_alternating():
    # 14 alternating points -> rule 14 fires at index 13.
    x = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=float)
    hits = nelson_trend_rules(x)
    rule14 = [idx for idx, rule in hits if rule == 14]
    assert 13 in rule14


def test_detect_violations_combines_rules():
    x = np.zeros(20)
    x[10] = 100.0
    chart = imr_chart(x)
    violations = detect_violations(chart.individuals)
    assert any(rule == 1 for _, rule in violations)


def test_ewma_detects_step_shift():
    rng = np.random.default_rng(1)
    base = rng.normal(0.0, 1.0, 50)
    shifted = np.concatenate([base[:25], base[25:] + 3.0])
    _, signals = ewma_drift(shifted, target=0.0, sigma=1.0, lam=0.2, L=3.0)
    assert len(signals) > 0
    assert min(signals) >= 25  # signal arises after the shift


def test_cusum_detects_step_shift():
    rng = np.random.default_rng(2)
    base = rng.normal(0.0, 1.0, 50)
    shifted = np.concatenate([base[:25], base[25:] + 2.0])
    _, _, signals = cusum_drift(shifted, target=0.0, sigma=1.0, k=0.5, h=5.0)
    assert len(signals) > 0
    assert min(signals) >= 25
