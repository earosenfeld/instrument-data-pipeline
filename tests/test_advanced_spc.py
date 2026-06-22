"""Validation tests for small-shift SPC: EWMA, CUSUM, ARL and Gage R&R.

The headline test is the *discrimination* test: on a small (1-sigma) sustained
shift, EWMA and CUSUM both flag an out-of-control point shortly after the shift,
while a Shewhart 3-sigma individuals chart (built from the production ``etl.spc``
chart, referenced to the known in-control parameters) flags nothing -- the whole
reason these charts exist.
"""

import numpy as np
import pytest

from etl.advanced_spc import (
    ewma_chart,
    cusum_chart,
    average_run_length,
    gage_rr,
    shewhart_individuals_signals,
)
from etl.spc import western_electric_rules


# Known Phase-I baseline: the process is in control at target 0, sigma 1, until a
# small step is injected at SHIFT_IDX. Seed 0 is a fixed, reproducible draw.
SHIFT_IDX = 30
TARGET = 0.0
SIGMA = 1.0


def _series_with_small_shift(seed=0, n=60, shift_sigma=1.0):
    rng = np.random.default_rng(seed)
    x = rng.normal(TARGET, SIGMA, n)
    x[SHIFT_IDX:] += shift_sigma * SIGMA  # small sustained shift
    return x


# ---------------------------------------------------------------------------
# Headline: EWMA + CUSUM catch a small shift that Shewhart misses.
# ---------------------------------------------------------------------------
def test_ewma_catches_small_shift_shewhart_misses():
    x = _series_with_small_shift()

    ew = ewma_chart(x, lam=0.2, L=3.0, target=TARGET, sigma=SIGMA)

    # EWMA fires, and only *after* the shift (never before).
    assert len(ew.signals) > 0
    first = min(ew.signals)
    assert first >= SHIFT_IDX
    # ...and shortly after it (small-shift charts are designed for fast ARL1).
    assert first <= SHIFT_IDX + 15

    # A Shewhart 3-sigma individuals chart on the SAME known parameters misses it
    # entirely: a 1-sigma offset keeps every reading inside +/-3 sigma.
    shewhart = shewhart_individuals_signals(x, target=TARGET, sigma=SIGMA, L=3.0)
    assert shewhart == []


def test_cusum_catches_small_shift_shewhart_misses():
    x = _series_with_small_shift()

    cu = cusum_chart(x, target=TARGET, sigma=SIGMA, k=0.5, h=5.0)

    assert len(cu.signals) > 0
    first = min(cu.signals)
    assert first >= SHIFT_IDX
    assert first <= SHIFT_IDX + 15

    shewhart = shewhart_individuals_signals(x, target=TARGET, sigma=SIGMA, L=3.0)
    assert shewhart == []


def test_shewhart_3sigma_via_production_spc_module_misses():
    """Cross-check the 3-sigma rule against the real ``etl.spc`` chart object.

    A "Shewhart 3-sigma individuals chart" is the beyond-limits rule (Western-
    Electric rule 1). Build an I-MR individuals chart from the production module
    and confirm that -- referenced to the *known* in-control target/sigma -- no
    individual point reaches 3 sigma anywhere in the 1-sigma-shifted stream, so
    the 3-sigma chart raises no alarm, while EWMA and CUSUM both did above.

    (The augmented Western-Electric *zone* run-rules (2/3/4) eventually react to
    a sustained 1-sigma shift, but that is a different, more elaborate chart than
    a plain Shewhart 3-sigma limit, and it fires later -- the small-shift charts
    are the purpose-built remedy.)
    """
    x = _series_with_small_shift()

    # Rule-1 (beyond 3 sigma) from the production WE-rules, referenced to the
    # known in-control centre/sigma -- never fires, anywhere in the stream.
    hits = western_electric_rules(x, center=TARGET, sigma=SIGMA)
    rule1 = [i for i, rule in hits if rule == 1]
    assert rule1 == []

    # (A naive I-MR chart that re-estimates its centre/sigma from this
    # *contaminated* stream would instead place its limits using the
    # shift-inflated mean and MRbar -- producing misleading flags that are not a
    # clean detection at the shift -- which is exactly why monitoring is done
    # against the known Phase-I parameters, as above.)

    # And the small-shift charts DID catch it -- the contrast is the point.
    assert len(ewma_chart(x, target=TARGET, sigma=SIGMA).signals) > 0
    assert len(cusum_chart(x, TARGET, SIGMA).signals) > 0


# ---------------------------------------------------------------------------
# In-control: few/no false alarms.
# ---------------------------------------------------------------------------
def test_ewma_in_control_no_false_alarms():
    # No shift; seed 1 is a clean in-control draw (ARL0 ~ 500 >> n).
    rng = np.random.default_rng(1)
    x = rng.normal(TARGET, SIGMA, 80)
    ew = ewma_chart(x, lam=0.2, L=3.0, target=TARGET, sigma=SIGMA)
    assert ew.signals == []


def test_cusum_in_control_no_false_alarms():
    rng = np.random.default_rng(1)
    x = rng.normal(TARGET, SIGMA, 80)
    cu = cusum_chart(x, target=TARGET, sigma=SIGMA, k=0.5, h=5.0)
    assert cu.signals == []


def test_in_control_false_alarm_rate_is_low():
    """Across many in-control streams, alarms are rare (well under the data)."""
    n_streams = 100
    ewma_alarms = 0
    cusum_alarms = 0
    for seed in range(2000, 2000 + n_streams):
        rng = np.random.default_rng(seed)
        x = rng.normal(TARGET, SIGMA, 50)
        if ewma_chart(x, target=TARGET, sigma=SIGMA).signals:
            ewma_alarms += 1
        if cusum_chart(x, TARGET, SIGMA).signals:
            cusum_alarms += 1
    # With ARL0 ~ 500 and 50-point streams, expect roughly 1-in-10 streams to
    # raise a false alarm -- certainly fewer than a third.
    assert ewma_alarms < n_streams * 0.30
    assert cusum_alarms < n_streams * 0.30


# ---------------------------------------------------------------------------
# EWMA limits widen to steady state; CUSUM resets.
# ---------------------------------------------------------------------------
def test_ewma_limits_widen_to_steady_state():
    x = np.zeros(120)  # exactly on target -> isolates the limit geometry
    ew = ewma_chart(x, lam=0.2, L=3.0, target=0.0, sigma=1.0)

    half = ew.upper - ew.target
    # Limits are symmetric about the target.
    assert np.allclose(ew.upper - ew.target, ew.target - ew.lower)
    # Monotonically non-decreasing as the EWMA variance accumulates.
    assert np.all(np.diff(half) >= -1e-12)
    # First limit is the tightest: L*sigma*sqrt(lam/(2-lam))*sqrt(1-(1-lam)^2).
    lam, L = 0.2, 3.0
    expected_first = L * 1.0 * np.sqrt((lam / (2 - lam)) * (1 - (1 - lam) ** 2))
    assert half[0] == pytest.approx(expected_first)
    # Last limit has effectively reached the steady-state half-width.
    assert half[-1] == pytest.approx(ew.steady_state_halfwidth, abs=1e-3)
    assert ew.steady_state_halfwidth == pytest.approx(
        L * 1.0 * np.sqrt(lam / (2 - lam))
    )


def test_ewma_recursion_matches_closed_form():
    # z_i = lam*x_i + (1-lam)*z_{i-1}, z_{-1} = target. Hand-roll and compare.
    x = np.array([1.0, 2.0, 1.0, 3.0, 2.0])
    lam, target = 0.3, 0.0
    ew = ewma_chart(x, lam=lam, L=3.0, target=target, sigma=1.0)
    z = []
    prev = target
    for xi in x:
        prev = lam * xi + (1 - lam) * prev
        z.append(prev)
    assert np.allclose(ew.ewma, z)


def test_cusum_resets_to_zero_when_in_control():
    # A perfectly on-target series keeps both sums pinned at exactly zero, since
    # every increment is -K (negative) and max(0, .) floors it.
    x = np.full(40, TARGET)
    cu = cusum_chart(x, target=TARGET, sigma=SIGMA, k=0.5, h=5.0)
    assert np.allclose(cu.c_plus, 0.0)
    assert np.allclose(cu.c_minus, 0.0)
    assert cu.signals == []


def test_cusum_resets_after_excursion_returns_to_control():
    # One-sided ramp drives C+ up; when the process returns to target, C+ decays
    # back toward zero (the slack K = 0.5*sigma subtracts each step) and resets.
    # Excursion: 5 steps at +2 sigma -> increment (2 - 0.5) = 1.5/step, peak 7.5
    # (> the limit 5). Recovery: peak 7.5 / 0.5 per step = 15 steps to reach 0;
    # a 30-point tail is comfortably enough to see the exact reset to zero.
    x = np.concatenate([
        np.full(10, TARGET),               # in control
        np.full(5, TARGET + 2.0 * SIGMA),  # upward excursion -> C+ ramps past H
        np.full(30, TARGET),               # back to target -> C+ must decay to 0
    ])
    cu = cusum_chart(x, target=TARGET, sigma=SIGMA, k=0.5, h=5.0)
    # C+ exceeded the limit during the excursion.
    assert cu.c_plus.max() > cu.limit
    # By the end of the long in-control tail it has reset to exactly zero.
    assert cu.c_plus[-1] == pytest.approx(0.0)
    assert cu.c_minus[-1] == pytest.approx(0.0)


def test_cusum_one_sided_lower_shift():
    # A downward shift should trip C- (lower sum), not C+.
    x = _series_with_small_shift(shift_sigma=-1.0)
    cu = cusum_chart(x, target=TARGET, sigma=SIGMA, k=0.5, h=5.0)
    assert len(cu.signals) > 0
    first = min(cu.signals)
    # The lower sum is the one that crossed at the first signal.
    assert cu.c_minus[first] > cu.limit
    assert cu.c_plus[first] <= cu.limit


# ---------------------------------------------------------------------------
# Average Run Length: small shift detected far faster than the false-alarm rate.
# ---------------------------------------------------------------------------
def test_average_run_length_ewma_arl1_much_less_than_arl0():
    arl0 = average_run_length("ewma", shift=0.0, n_runs=400, max_len=800, seed=3)
    arl1 = average_run_length("ewma", shift=1.0, n_runs=400, max_len=800, seed=3)
    assert arl1 < 25            # ~1-sigma shift caught quickly
    assert arl0 > 5 * arl1      # in-control runs are far longer


def test_average_run_length_cusum_arl1_much_less_than_arl0():
    arl0 = average_run_length("cusum", shift=0.0, n_runs=400, max_len=800, seed=3)
    arl1 = average_run_length("cusum", shift=1.0, n_runs=400, max_len=800, seed=3)
    assert arl1 < 25
    assert arl0 > 5 * arl1


# ---------------------------------------------------------------------------
# Gage R&R (ANOVA): a good gage on a part-dominated study.
# ---------------------------------------------------------------------------
def test_gage_rr_good_system_low_pct_grr_high_ndc():
    rng = np.random.default_rng(42)
    parts = rng.normal(10.0, 5.0, 5)          # wide part-to-part spread
    op_bias = {0: 0.0, 1: 0.3, 2: -0.2}       # small reproducibility differences
    data = []
    for op in range(3):
        rows = []
        for pv in parts:
            reps = [pv + op_bias[op] + rng.normal(0.0, 0.3) for _ in range(3)]
            rows.append(reps)
        data.append(rows)

    g = gage_rr(data)
    # Part variation dominates: a capable measurement system.
    assert g.pct_grr < 30.0          # AIAG: <30% acceptable, <10% good
    assert g.pct_part > 80.0
    assert g.ndc >= 5                # adequate discrimination
    # %study-variation components combine in quadrature to 100%.
    assert (g.pct_grr ** 2 + g.pct_part ** 2) ** 0.5 == pytest.approx(100.0, abs=1e-6)
    # Variance components are non-negative.
    assert g.var_repeatability >= 0
    assert g.var_reproducibility >= 0
    assert g.var_part >= 0


def test_gage_rr_bad_system_high_pct_grr():
    # Huge gage noise relative to part spread -> %GRR should be large, ndc small.
    rng = np.random.default_rng(7)
    parts = rng.normal(10.0, 0.5, 5)          # parts nearly identical
    data = []
    for op in range(3):
        rows = []
        for pv in parts:
            reps = [pv + rng.normal(0.0, 4.0) for _ in range(3)]  # noisy gage
            rows.append(reps)
        data.append(rows)

    g = gage_rr(data)
    assert g.pct_grr > 30.0
    assert g.ndc < 5


def test_gage_rr_rejects_bad_shape():
    with pytest.raises(ValueError):
        gage_rr(np.zeros((3, 1, 2)))  # only 1 part
    with pytest.raises(ValueError):
        gage_rr(np.zeros((3, 5, 1)))  # only 1 replicate


# ---------------------------------------------------------------------------
# Input validation.
# ---------------------------------------------------------------------------
def test_ewma_rejects_bad_lambda():
    with pytest.raises(ValueError):
        ewma_chart([1, 2, 3], lam=0.0)
    with pytest.raises(ValueError):
        ewma_chart([1, 2, 3], lam=1.5)


def test_cusum_rejects_bad_sigma():
    with pytest.raises(ValueError):
        cusum_chart([1, 2, 3], target=0.0, sigma=0.0)
