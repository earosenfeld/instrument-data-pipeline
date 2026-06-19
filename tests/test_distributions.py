"""Validation tests for the physically realistic part-population generators."""

import numpy as np
import pytest

from etl import distributions as dist
from etl.distributions import Spec


def test_spec_in_spec_two_sided_and_one_sided():
    two = Spec("v", nominal=5.0, lsl=4.5, usl=5.5)
    assert two.in_spec(np.array([4.4, 5.0, 5.6])).tolist() == [False, True, False]
    upper = Spec("leak", nominal=0.1, usl=1.0)
    assert upper.in_spec(np.array([0.5, 1.5])).tolist() == [True, False]
    lower = Spec("res", nominal=500.0, lsl=100.0)
    assert lower.in_spec(np.array([50.0, 200.0])).tolist() == [False, True]


def test_lognormal_leakage_positive_and_skewed():
    x = dist.lognormal_leakage(5000, median=0.1, sigma_log=0.5, breakdown_rate=0.0,
                               rng=np.random.default_rng(0))
    assert np.all(x > 0), "leakage current must be strictly positive"
    # Log-normal is right-skewed: mean > median.
    assert np.mean(x) > np.median(x)
    # Median is preserved approximately.
    assert np.median(x) == pytest.approx(0.1, rel=0.1)


def test_lognormal_leakage_has_breakdown_tail():
    x = dist.lognormal_leakage(5000, median=0.1, sigma_log=0.4, breakdown_rate=0.05,
                               breakdown_mult=40.0, rng=np.random.default_rng(1))
    # Some parts exceed a 1 mA leakage limit due to the breakdown tail.
    assert np.any(x > 1.0)


def test_normal_with_fliers_mean_preserved_but_has_outliers():
    x = dist.normal_with_fliers(5000, nominal=100.0, sigma=1.0, flier_rate=0.02,
                                flier_sigma_mult=8.0, rng=np.random.default_rng(2))
    assert np.mean(x) == pytest.approx(100.0, abs=0.5)
    # Fliers create points far beyond 3 sigma of the base distribution.
    assert np.max(np.abs(x - 100.0)) > 4.0


def test_laser_power_and_wavelength_are_independent_draws():
    df = dist.generate_laser_population(2000, seed=44)
    power = df["power_mW"].to_numpy()
    wave = df["wavelength_nm"].to_numpy()
    # The original bug made these identical arrays. They must differ.
    assert not np.allclose(power, wave)
    # Correlation should be moderate (we injected ~0.3), nowhere near 1.0.
    corr = np.corrcoef(power, wave)[0, 1]
    assert abs(corr) < 0.7


def test_laser_zero_correlation_option():
    df = dist.generate_laser_population(3000, seed=7, power_wavelength_corr=0.0)
    corr = np.corrcoef(df["power_mW"], df["wavelength_nm"])[0, 1]
    assert abs(corr) < 0.1


def test_population_pass_fail_driven_by_specs():
    df = dist.generate_parametric_population(1000, seed=45)
    # PASS exactly iff both parameters are within their spec limits.
    v_ok = dist.PARAMETRIC_SPECS["voltage_mV"].in_spec(df["voltage_mV"].to_numpy())
    i_ok = dist.PARAMETRIC_SPECS["current_mA"].in_spec(df["current_mA"].to_numpy())
    expected = v_ok & i_ok
    assert np.array_equal(df["passed"].to_numpy(), expected)
    # Failure modes are populated for failures only.
    failed = ~df["passed"].to_numpy()
    assert all(df.loc[failed, "failure_mode"] != "")
    assert all(df.loc[df["passed"], "failure_mode"] == "")


def test_isolation_population_has_low_resistance_failures():
    df = dist.generate_isolation_population(2000, seed=43)
    assert np.all(df["resistance_Mohm"] > 0)
    # The short tail produces some sub-100-MOhm failures.
    assert np.any(df["resistance_Mohm"] < 100.0)
    assert not df["passed"].all()


def test_reproducible_with_seed():
    a = dist.generate_hipot_population(100, seed=99)
    b = dist.generate_hipot_population(100, seed=99)
    assert a.equals(b)


def test_burnin_drift_series_ramps():
    s = dist.burnin_drift_series(200, seed=47, drift_start=120)
    # Late-segment mean exceeds early-segment mean due to the ramp.
    assert s[150:].mean() > s[:120].mean()
