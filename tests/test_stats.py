"""Validation tests for process-capability indices.

Known mu/sigma/USL/LSL inputs are checked against hand-computed values so the
formulas cannot silently drift.
"""

import math

import numpy as np
import pytest

from etl.stats import (
    CapabilityResult,
    capability,
    capability_from_values,
    within_subgroup_sigma,
    D2_CONSTANTS,
)


def test_cp_cpk_known_values_centered():
    # mu=10, sigma=1, USL=16, LSL=4  ->  Cp = 12/6 = 2.0 ; centered so Cpk = 2.0
    r = capability(mu=10.0, sigma=1.0, usl=16.0, lsl=4.0)
    assert r.cp == pytest.approx(2.0)
    assert r.cpk == pytest.approx(2.0)
    # On target -> Cpm equals Cp.
    assert r.cpm == pytest.approx(2.0)


def test_cpk_off_center_uses_nearer_limit():
    # mu=12, sigma=1, USL=16, LSL=4.  Cp = 12/6 = 2.0
    # CPU=(16-12)/3=1.333..., CPL=(12-4)/3=2.667 -> Cpk = min = 1.3333
    r = capability(mu=12.0, sigma=1.0, usl=16.0, lsl=4.0)
    assert r.cp == pytest.approx(2.0)
    assert r.cpk == pytest.approx(4.0 / 3.0)
    assert r.cpu == pytest.approx(4.0 / 3.0)
    assert r.cpl == pytest.approx(8.0 / 3.0)


def test_cpm_penalizes_off_target():
    # mu=12, T=10, sigma=1, USL=16, LSL=4.
    # Cpm = (16-4)/(6*sqrt(1 + (12-10)^2)) = 12/(6*sqrt(5)) = 2/sqrt(5)
    r = capability(mu=12.0, sigma=1.0, usl=16.0, lsl=4.0, target=10.0)
    assert r.cpm == pytest.approx(2.0 / math.sqrt(5.0))
    # Cpm < Cp when off target.
    assert r.cpm < r.cp


def test_one_sided_upper_only():
    # Only USL: Cpk == CPU = (USL-mu)/(3 sigma); Cp and Cpm undefined.
    r = capability(mu=8.0, sigma=1.0, usl=11.0)
    assert r.cp is None
    assert r.cpm is None
    assert r.cpu == pytest.approx(1.0)
    assert r.cpk == pytest.approx(1.0)
    assert r.cpl is None


def test_one_sided_lower_only():
    r = capability(mu=8.0, sigma=1.0, lsl=5.0)
    assert r.cp is None
    assert r.cpl == pytest.approx(1.0)
    assert r.cpk == pytest.approx(1.0)
    assert r.cpu is None


def test_requires_at_least_one_limit():
    with pytest.raises(ValueError):
        capability(mu=0.0, sigma=1.0)


def test_zero_sigma_rejected():
    with pytest.raises(ValueError):
        capability(mu=0.0, sigma=0.0, usl=1.0, lsl=-1.0)


def test_within_subgroup_sigma_constant():
    # Rbar/d2 with n=5, d2=2.326.
    assert within_subgroup_sigma(2.326, 5) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        within_subgroup_sigma(1.0, 99)


def test_capability_from_values_pp_ppk_overall_sigma():
    # Construct data with a known overall sample std.
    rng = np.random.default_rng(0)
    data = rng.normal(100.0, 2.0, 2000)
    r = capability_from_values(data, usl=110.0, lsl=90.0, subgroup_size=5)
    # Pp uses overall sigma; with sigma~2 -> Pp ~ 20/(6*2) ~ 1.6
    assert r.pp == pytest.approx(20.0 / (6.0 * r.sigma))
    assert r.ppk == pytest.approx(
        min(110.0 - r.mu, r.mu - 90.0) / (3.0 * r.sigma)
    )
    # Within-subgroup sigma populated and Cp uses it.
    assert r.sigma_within is not None
    assert r.cp == pytest.approx(20.0 / (6.0 * r.sigma_within))


def test_capability_from_values_without_subgroups_cp_equals_pp():
    data = [9.0, 10.0, 11.0, 10.0, 9.5, 10.5, 9.8, 10.2]
    r = capability_from_values(data, usl=13.0, lsl=7.0)
    # No subgroup structure -> Cp falls back to overall sigma -> Cp == Pp.
    assert r.cp == pytest.approx(r.pp)
    assert r.cpk == pytest.approx(r.ppk)


def test_to_dict_roundtrip():
    r = capability(mu=10.0, sigma=1.0, usl=16.0, lsl=4.0)
    d = r.to_dict()
    assert d["cp"] == pytest.approx(2.0)
    assert isinstance(d, dict)
