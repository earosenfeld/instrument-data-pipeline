"""Physically realistic part-population generators for each test type.

The original simulators reused a single rectified sine wave for *every* test,
which is not physical: leakage current does not oscillate sinusoidally and laser
power is not the same array as wavelength. This module replaces that with named
specification limits per parameter and statistically appropriate part-to-part
distributions:

* **Leakage / HiPot current, isolation resistance** -> LOG-NORMAL. Insulation
  and leakage quantities span decades, are strictly positive, and are
  right/left-skewed; a rare breakdown tail is injected (gross escapes).
* **Parametric values** (voltage, current, resistance, capacitance, power
  rails) -> NORMAL(nominal, process sigma) with an occasional flier via a
  small mixture component.

Every generator returns a population of *parts* (one row per part), each with a
PASS/FAIL verdict and, on failure, a failure-mode label -- exactly the shape the
yield / SPC / capability code consumes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class Spec:
    """Named specification for one measured parameter.

    Exactly one of ``lsl`` / ``usl`` may be ``None`` for a one-sided spec.
    ``nominal`` is the design target; ``units`` is cosmetic (for labels).
    """

    name: str
    nominal: float
    lsl: Optional[float] = None
    usl: Optional[float] = None
    units: str = ""

    def in_spec(self, values: np.ndarray) -> np.ndarray:
        ok = np.ones_like(values, dtype=bool)
        if self.lsl is not None:
            ok &= values >= self.lsl
        if self.usl is not None:
            ok &= values <= self.usl
        return ok


def _rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed)


def normal_with_fliers(
    n: int,
    nominal: float,
    sigma: float,
    flier_rate: float = 0.01,
    flier_sigma_mult: float = 6.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Normal(nominal, sigma) contaminated by a small heavy-tailed flier mixture.

    A fraction ``flier_rate`` of parts are drawn from a wider normal centred on
    the nominal (sigma scaled by ``flier_sigma_mult``), modelling the occasional
    gross outlier seen on real parametric stations.
    """
    rng = rng or _rng(None)
    base = rng.normal(nominal, sigma, n)
    is_flier = rng.random(n) < flier_rate
    n_fliers = int(is_flier.sum())
    if n_fliers:
        base[is_flier] = rng.normal(nominal, sigma * flier_sigma_mult, n_fliers)
    return base


def lognormal_leakage(
    n: int,
    median: float,
    sigma_log: float,
    breakdown_rate: float = 0.005,
    breakdown_mult: float = 50.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Strictly-positive log-normal leakage current (decades of spread).

    ``median`` is the median leakage; ``sigma_log`` is the standard deviation of
    the underlying log. A ``breakdown_rate`` fraction suffer dielectric breakdown
    and draw catastrophically higher current (multiplied by ``breakdown_mult``),
    forming the failure tail HiPot/leakage screens are designed to catch.
    """
    rng = rng or _rng(None)
    mu_log = np.log(median)
    base = rng.lognormal(mean=mu_log, sigma=sigma_log, size=n)
    is_breakdown = rng.random(n) < breakdown_rate
    base[is_breakdown] *= breakdown_mult
    return base


def lognormal_resistance(
    n: int,
    median: float,
    sigma_log: float,
    short_rate: float = 0.005,
    short_div: float = 100.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Log-normal isolation resistance with a rare low-resistance (short) tail.

    Isolation resistance is large and positive; a ``short_rate`` fraction have a
    partial short and draw resistance divided by ``short_div`` (the low tail that
    fails a minimum-resistance spec).
    """
    rng = rng or _rng(None)
    mu_log = np.log(median)
    base = rng.lognormal(mean=mu_log, sigma=sigma_log, size=n)
    is_short = rng.random(n) < short_rate
    base[is_short] /= short_div
    return base


# ---------------------------------------------------------------------------
# Default per-test specifications. These are realistic order-of-magnitude values
# for the parameters each screen measures; they drive both data generation and
# the PASS/FAIL limits so the two can never disagree.
# ---------------------------------------------------------------------------
HIPOT_SPECS = {
    # 1 mA max leakage at the applied hipot voltage; one-sided upper.
    "leakage_current_mA": Spec("leakage_current_mA", nominal=0.10, usl=1.0, units="mA"),
    # Applied dielectric-withstand voltage holds within +/- band of 5 kV.
    "applied_voltage_kV": Spec("applied_voltage_kV", nominal=5.0, lsl=4.5, usl=5.5, units="kV"),
}

ISOLATION_SPECS = {
    # >= 100 MOhm isolation; one-sided lower spec.
    "resistance_Mohm": Spec("resistance_Mohm", nominal=500.0, lsl=100.0, units="MOhm"),
}

LASER_SPECS = {
    "power_mW": Spec("power_mW", nominal=55.0, lsl=10.0, usl=100.0, units="mW"),
    "wavelength_nm": Spec("wavelength_nm", nominal=825.0, lsl=800.0, usl=850.0, units="nm"),
}

PARAMETRIC_SPECS = {
    "voltage_mV": Spec("voltage_mV", nominal=3300.0, lsl=3200.0, usl=3400.0, units="mV"),
    "current_mA": Spec("current_mA", nominal=500.0, lsl=450.0, usl=550.0, units="mA"),
}

BURNIN_SPECS = {
    "supply_voltage_V": Spec("supply_voltage_V", nominal=3.3, lsl=3.0, usl=3.6, units="V"),
    "supply_current_A": Spec("supply_current_A", nominal=0.5, lsl=0.3, usl=0.7, units="A"),
    # Junction temperature must stay below 90 C; one-sided upper.
    "junction_temp_C": Spec("junction_temp_C", nominal=70.0, usl=90.0, units="C"),
}


def _verdict(in_spec_flags: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
    """Combine per-parameter in-spec flags into a unit PASS bool + failure mode.

    Returns ``(passed, failure_modes)`` where ``failure_modes[i]`` names the
    first failing parameter for unit ``i`` (empty string when the unit passed).
    """
    names = list(in_spec_flags)
    n = len(next(iter(in_spec_flags.values())))
    passed = np.ones(n, dtype=bool)
    for flags in in_spec_flags.values():
        passed &= flags
    failure_modes = [""] * n
    for i in range(n):
        if not passed[i]:
            for name in names:
                if not in_spec_flags[name][i]:
                    failure_modes[i] = name
                    break
    return passed, failure_modes


def generate_hipot_population(
    n: int = 500, seed: Optional[int] = 42
) -> pd.DataFrame:
    """HiPot population: log-normal leakage current + normal applied voltage."""
    rng = _rng(seed)
    leakage = lognormal_leakage(
        n, median=0.10, sigma_log=0.55, breakdown_rate=0.01, breakdown_mult=40.0, rng=rng
    )
    voltage = normal_with_fliers(
        n, nominal=5.0, sigma=0.12, flier_rate=0.005, rng=rng
    )
    flags = {
        "leakage_current_mA": HIPOT_SPECS["leakage_current_mA"].in_spec(leakage),
        "applied_voltage_kV": HIPOT_SPECS["applied_voltage_kV"].in_spec(voltage),
    }
    passed, modes = _verdict(flags)
    return pd.DataFrame(
        {
            "unit_id": np.arange(1, n + 1),
            "applied_voltage_kV": voltage,
            "leakage_current_mA": leakage,
            "passed": passed,
            "failure_mode": modes,
        }
    )


def generate_isolation_population(
    n: int = 500, seed: Optional[int] = 43
) -> pd.DataFrame:
    """Isolation population: log-normal resistance with a low-resistance tail."""
    rng = _rng(seed)
    resistance = lognormal_resistance(
        n, median=500.0, sigma_log=0.45, short_rate=0.01, short_div=80.0, rng=rng
    )
    flags = {"resistance_Mohm": ISOLATION_SPECS["resistance_Mohm"].in_spec(resistance)}
    passed, modes = _verdict(flags)
    return pd.DataFrame(
        {
            "unit_id": np.arange(1, n + 1),
            "resistance_Mohm": resistance,
            "passed": passed,
            "failure_mode": modes,
        }
    )


def generate_laser_population(
    n: int = 500, seed: Optional[int] = 44, power_wavelength_corr: float = 0.3
) -> pd.DataFrame:
    """Laser population: INDEPENDENT power and wavelength draws.

    Fixes the original bug where wavelength was literally the same array as
    power. Here each is its own normal draw; a modest, physically plausible
    correlation (``power_wavelength_corr``) is injected via a shared latent
    factor so they are correlated but never identical.
    """
    rng = _rng(seed)
    rho = float(np.clip(power_wavelength_corr, -0.95, 0.95))
    # Shared latent + independent components => correlation rho, unit variance.
    latent = rng.standard_normal(n)
    z_power = rho * latent + np.sqrt(1 - rho ** 2) * rng.standard_normal(n)
    z_wave = rho * latent + np.sqrt(1 - rho ** 2) * rng.standard_normal(n)

    power = 55.0 + 12.0 * z_power
    wavelength = 825.0 + 6.0 * z_wave
    # Independent gross fliers on each parameter (no shared array).
    power[rng.random(n) < 0.01] = rng.normal(55.0, 60.0)
    wavelength[rng.random(n) < 0.01] = rng.normal(825.0, 40.0)

    flags = {
        "power_mW": LASER_SPECS["power_mW"].in_spec(power),
        "wavelength_nm": LASER_SPECS["wavelength_nm"].in_spec(wavelength),
    }
    passed, modes = _verdict(flags)
    return pd.DataFrame(
        {
            "unit_id": np.arange(1, n + 1),
            "power_mW": power,
            "wavelength_nm": wavelength,
            "passed": passed,
            "failure_mode": modes,
        }
    )


def generate_parametric_population(
    n: int = 500, seed: Optional[int] = 45
) -> pd.DataFrame:
    """Parametric population: normal voltage & current with occasional fliers."""
    rng = _rng(seed)
    voltage = normal_with_fliers(n, nominal=3300.0, sigma=35.0, flier_rate=0.01, rng=rng)
    current = normal_with_fliers(n, nominal=500.0, sigma=18.0, flier_rate=0.01, rng=rng)
    power = voltage * current / 1000.0  # mW
    flags = {
        "voltage_mV": PARAMETRIC_SPECS["voltage_mV"].in_spec(voltage),
        "current_mA": PARAMETRIC_SPECS["current_mA"].in_spec(current),
    }
    passed, modes = _verdict(flags)
    return pd.DataFrame(
        {
            "unit_id": np.arange(1, n + 1),
            "voltage_mV": voltage,
            "current_mA": current,
            "power_mW": power,
            "passed": passed,
            "failure_mode": modes,
        }
    )


def generate_burnin_population(
    n: int = 500, seed: Optional[int] = 46, n_timepoints: int = 60
) -> pd.DataFrame:
    """Burn-in: per-part supply rails (normal) + a junction-temperature soak.

    Returns one row per part with summary readings. The junction temperature is
    drawn so a small fraction exceed the 90 C limit (thermal escapes), giving the
    one-sided upper spec something to catch.
    """
    rng = _rng(seed)
    voltage = normal_with_fliers(n, nominal=3.3, sigma=0.06, flier_rate=0.005, rng=rng)
    current = normal_with_fliers(n, nominal=0.5, sigma=0.03, flier_rate=0.005, rng=rng)
    # Junction temp: most parts soak around 70 C; a thermal tail pushes past 90.
    temp = rng.normal(70.0, 6.0, n)
    temp[rng.random(n) < 0.01] = rng.normal(95.0, 4.0)
    flags = {
        "supply_voltage_V": BURNIN_SPECS["supply_voltage_V"].in_spec(voltage),
        "supply_current_A": BURNIN_SPECS["supply_current_A"].in_spec(current),
        "junction_temp_C": BURNIN_SPECS["junction_temp_C"].in_spec(temp),
    }
    passed, modes = _verdict(flags)
    return pd.DataFrame(
        {
            "unit_id": np.arange(1, n + 1),
            "supply_voltage_V": voltage,
            "supply_current_A": current,
            "junction_temp_C": temp,
            "passed": passed,
            "failure_mode": modes,
        }
    )


def burnin_drift_series(
    n: int = 200, seed: Optional[int] = 47, drift_start: int = 120
) -> np.ndarray:
    """A burn-in current time-series with a slow upward drift partway through.

    Used to exercise EWMA / CUSUM drift detection: stationary noise until
    ``drift_start``, then a gradual ramp (early-life degradation).
    """
    rng = _rng(seed)
    base = rng.normal(0.50, 0.01, n)
    ramp = np.zeros(n)
    if drift_start < n:
        ramp[drift_start:] = np.linspace(0, 0.05, n - drift_start)
    return base + ramp


# Map a test-type name to its spec dict, for generic tooling.
SPECS_BY_TEST: Dict[str, Dict[str, Spec]] = {
    "hipot": HIPOT_SPECS,
    "isolation": ISOLATION_SPECS,
    "laser": LASER_SPECS,
    "parametric": PARAMETRIC_SPECS,
    "burnin": BURNIN_SPECS,
}
