"""Statistical Process Control: control charts and run-rule detection.

Implements the two work-horse Shewhart chart pairs used on a production line:

* **I-MR** (Individuals / Moving-Range) for one-at-a-time measurements.
* **X-bar / R** for rational subgroups of size ``n``.

Control limits use the standard control-chart constants (Montgomery, Appendix
VI), not the naive ``mean +/- 3*std`` that decorative dashboards use. In
particular the R-chart lower limit is ``max(0, D3*Rbar)`` so it can never go
negative -- a range is non-negative by construction.

Run-rule detection (:func:`western_electric_rules`, :func:`nelson_trend_rules`)
returns a list of ``(point_index, rule_id)`` violations so out-of-control
signals can be located, not just plotted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Shewhart control-chart constants by subgroup size n (Montgomery, Appendix VI).
#   A2 : X-bar chart limit factor       (X-bar +/- A2 * Rbar)
#   D3 : R-chart lower limit factor     (LCL = D3 * Rbar)
#   D4 : R-chart upper limit factor     (UCL = D4 * Rbar)
#   d2 : range-to-sigma unbiasing const (sigma_hat = Rbar / d2)
# ---------------------------------------------------------------------------
CONTROL_CONSTANTS: Dict[int, Dict[str, float]] = {
    2: {"A2": 1.880, "D3": 0.0, "D4": 3.267, "d2": 1.128},
    3: {"A2": 1.023, "D3": 0.0, "D4": 2.574, "d2": 1.693},
    4: {"A2": 0.729, "D3": 0.0, "D4": 2.282, "d2": 2.059},
    5: {"A2": 0.577, "D3": 0.0, "D4": 2.114, "d2": 2.326},
    6: {"A2": 0.483, "D3": 0.0, "D4": 2.004, "d2": 2.534},
    7: {"A2": 0.419, "D3": 0.076, "D4": 1.924, "d2": 2.704},
    8: {"A2": 0.373, "D3": 0.136, "D4": 1.864, "d2": 2.847},
    9: {"A2": 0.337, "D3": 0.184, "D4": 1.816, "d2": 2.970},
    10: {"A2": 0.308, "D3": 0.223, "D4": 1.777, "d2": 3.078},
}

# For an Individuals chart the moving range uses n=2 consecutive points, so the
# I-chart 3-sigma spread derives from d2(2)=1.128: sigma_hat = MRbar / 1.128 and
# the individuals limits are X-bar +/- 3 * sigma_hat = X-bar +/- 2.66 * MRbar.
_E2 = 3.0 / CONTROL_CONSTANTS[2]["d2"]  # 2.660
_D4_MR = CONTROL_CONSTANTS[2]["D4"]     # 3.267
_D3_MR = CONTROL_CONSTANTS[2]["D3"]     # 0.0


@dataclass
class ControlChart:
    """Plotted statistic plus its centre line and 3-sigma control limits."""

    name: str
    values: np.ndarray      # the points actually plotted
    center: float
    ucl: float
    lcl: float
    sigma: Optional[float] = None  # estimated process sigma (where meaningful)

    def out_of_limit_indices(self) -> List[int]:
        """Indices of points beyond the control limits (rule-1 candidates)."""
        return [
            i
            for i, v in enumerate(self.values)
            if not np.isnan(v) and (v > self.ucl or v < self.lcl)
        ]


@dataclass
class IMRChart:
    individuals: ControlChart
    moving_range: ControlChart


@dataclass
class XbarRChart:
    xbar: ControlChart
    r: ControlChart
    subgroup_size: int


def _require_constants(n: int) -> Dict[str, float]:
    if n not in CONTROL_CONSTANTS:
        raise ValueError(f"subgroup size n={n} unsupported (tabulated 2..10)")
    return CONTROL_CONSTANTS[n]


def imr_chart(values: Sequence[float]) -> IMRChart:
    """Build an Individuals & Moving-Range chart from single measurements."""
    x = np.asarray(values, dtype=float)
    if x.size < 2:
        raise ValueError("I-MR chart needs at least 2 points")

    moving_range = np.abs(np.diff(x))
    mr_bar = float(np.mean(moving_range))
    x_bar = float(np.mean(x))
    sigma_hat = mr_bar / CONTROL_CONSTANTS[2]["d2"]

    individuals = ControlChart(
        name="Individuals",
        values=x,
        center=x_bar,
        ucl=x_bar + _E2 * mr_bar,
        lcl=x_bar - _E2 * mr_bar,
        sigma=sigma_hat,
    )
    # The MR series is plotted aligned to points 1..N-1; prepend NaN so indices
    # line up with the individuals series.
    mr_plot = np.concatenate([[np.nan], moving_range])
    mr = ControlChart(
        name="Moving Range",
        values=mr_plot,
        center=mr_bar,
        ucl=_D4_MR * mr_bar,
        lcl=max(0.0, _D3_MR * mr_bar),
        sigma=sigma_hat,
    )
    return IMRChart(individuals=individuals, moving_range=mr)


def xbar_r_chart(subgroups: Sequence[Sequence[float]]) -> XbarRChart:
    """Build X-bar / R charts from rational subgroups of equal size ``n``.

    ``subgroups`` is a sequence of equal-length subgroups (n in 2..10).
    """
    groups = [np.asarray(g, dtype=float) for g in subgroups]
    if len(groups) < 1:
        raise ValueError("need at least one subgroup")
    n = groups[0].size
    if any(g.size != n for g in groups):
        raise ValueError("all subgroups must have the same size")
    const = _require_constants(n)

    means = np.array([g.mean() for g in groups])
    ranges = np.array([g.max() - g.min() for g in groups])
    xbarbar = float(means.mean())
    rbar = float(ranges.mean())
    sigma_hat = rbar / const["d2"] if rbar > 0 else 0.0

    xbar = ControlChart(
        name="X-bar",
        values=means,
        center=xbarbar,
        ucl=xbarbar + const["A2"] * rbar,
        lcl=xbarbar - const["A2"] * rbar,
        sigma=sigma_hat,
    )
    r = ControlChart(
        name="R",
        values=ranges,
        center=rbar,
        ucl=const["D4"] * rbar,
        lcl=max(0.0, const["D3"] * rbar),  # never negative
        sigma=sigma_hat,
    )
    return XbarRChart(xbar=xbar, r=r, subgroup_size=n)


# ---------------------------------------------------------------------------
# Run-rule detection. Each function returns a sorted, de-duplicated list of
# (point_index, rule_id) tuples. Zones are defined relative to the centre line
# and an estimated sigma: zone A = 2..3 sigma, B = 1..2 sigma, C = 0..1 sigma.
# ---------------------------------------------------------------------------
def western_electric_rules(
    values: Sequence[float], center: float, sigma: float
) -> List[Tuple[int, int]]:
    """Western Electric rules 1-4.

    * Rule 1: a single point beyond 3 sigma.
    * Rule 2: 2 of 3 consecutive points beyond 2 sigma on the same side.
    * Rule 3: 4 of 5 consecutive points beyond 1 sigma on the same side.
    * Rule 4: 8 consecutive points on the same side of the centre line.

    The flagged index is the *last* point of the triggering window, which is the
    point at which the rule fires in real-time monitoring.
    """
    x = np.asarray(values, dtype=float)
    if sigma <= 0:
        return []
    z = (x - center) / sigma
    n = x.size
    hits: List[Tuple[int, int]] = []

    # Rule 1: |z| > 3
    for i in range(n):
        if not np.isnan(z[i]) and abs(z[i]) > 3:
            hits.append((i, 1))

    # Rule 2: 2 of 3 with z > 2 (or < -2) on the same side.
    for i in range(2, n):
        win = z[i - 2 : i + 1]
        if np.isnan(win).any():
            continue
        if np.sum(win > 2) >= 2 or np.sum(win < -2) >= 2:
            hits.append((i, 2))

    # Rule 3: 4 of 5 with z > 1 (or < -1) on the same side.
    for i in range(4, n):
        win = z[i - 4 : i + 1]
        if np.isnan(win).any():
            continue
        if np.sum(win > 1) >= 4 or np.sum(win < -1) >= 4:
            hits.append((i, 3))

    # Rule 4: 8 consecutive on the same side of centre.
    for i in range(7, n):
        win = z[i - 7 : i + 1]
        if np.isnan(win).any():
            continue
        if np.all(win > 0) or np.all(win < 0):
            hits.append((i, 4))

    return sorted(set(hits))


def nelson_trend_rules(values: Sequence[float]) -> List[Tuple[int, int]]:
    """Nelson rules that need no sigma: trend (id 6) and alternation (id 14).

    Rule ids follow Nelson's original numbering so they don't collide with the
    Western Electric set above.

    * Rule 6: 6 consecutive points steadily increasing or steadily decreasing.
    * Rule 14: 14 consecutive points alternating up and down (over-control).
    """
    x = np.asarray(values, dtype=float)
    n = x.size
    hits: List[Tuple[int, int]] = []

    # Rule 6: 6-in-a-row monotonic (uses 5 consecutive diffs of the same sign).
    for i in range(5, n):
        win = x[i - 5 : i + 1]
        if np.isnan(win).any():
            continue
        d = np.diff(win)
        if np.all(d > 0) or np.all(d < 0):
            hits.append((i, 6))

    # Rule 14: 14 points alternating direction (13 consecutive sign flips).
    for i in range(13, n):
        win = x[i - 13 : i + 1]
        if np.isnan(win).any():
            continue
        d = np.diff(win)
        if np.any(d == 0):
            continue
        signs = np.sign(d)
        if np.all(signs[1:] != signs[:-1]):
            hits.append((i, 14))

    return sorted(set(hits))


def detect_violations(chart: ControlChart) -> List[Tuple[int, int]]:
    """All Western-Electric + Nelson violations for a single control chart."""
    sigma = chart.sigma if chart.sigma is not None else 0.0
    we = western_electric_rules(chart.values, chart.center, sigma)
    nelson = nelson_trend_rules(chart.values)
    return sorted(set(we + nelson))


# ---------------------------------------------------------------------------
# Drift detection for time-series (e.g. burn-in current over soak time). EWMA and
# CUSUM react to small sustained shifts that a Shewhart chart is slow to catch.
# ---------------------------------------------------------------------------
def ewma_drift(
    values: Sequence[float],
    target: Optional[float] = None,
    sigma: Optional[float] = None,
    lam: float = 0.2,
    L: float = 3.0,
) -> Tuple[np.ndarray, List[int]]:
    """EWMA control chart; returns ``(ewma_series, signal_indices)``.

    ``lam`` is the smoothing weight (0<lam<=1); ``L`` the control-limit width in
    sigma. Target and sigma default to the series mean and std when omitted. The
    EWMA variance grows then plateaus to ``sigma^2 * lam/(2-lam)``; limits are
    computed per-point accordingly.
    """
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return np.array([]), []
    mu0 = float(np.mean(x)) if target is None else float(target)
    sd = float(np.std(x, ddof=1)) if sigma is None else float(sigma)

    z = np.empty_like(x)
    prev = mu0
    signals: List[int] = []
    for i, xi in enumerate(x):
        prev = lam * xi + (1 - lam) * prev
        z[i] = prev
        var_factor = (lam / (2 - lam)) * (1 - (1 - lam) ** (2 * (i + 1)))
        limit = L * sd * np.sqrt(var_factor)
        if abs(z[i] - mu0) > limit:
            signals.append(i)
    return z, signals


def cusum_drift(
    values: Sequence[float],
    target: Optional[float] = None,
    sigma: Optional[float] = None,
    k: float = 0.5,
    h: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Tabular CUSUM; returns ``(c_plus, c_minus, signal_indices)``.

    ``k`` is the slack (in sigma, typically half the shift to detect) and ``h``
    the decision interval (in sigma, commonly 4-5). Indices where either the
    upper or lower cumulative sum exceeds ``h*sigma`` are returned.
    """
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return np.array([]), np.array([]), []
    mu0 = float(np.mean(x)) if target is None else float(target)
    sd = float(np.std(x, ddof=1)) if sigma is None else float(sigma)
    if sd <= 0:
        return np.zeros_like(x), np.zeros_like(x), []

    slack = k * sd
    decision = h * sd
    c_plus = np.zeros_like(x)
    c_minus = np.zeros_like(x)
    signals: List[int] = []
    cp = cm = 0.0
    for i, xi in enumerate(x):
        cp = max(0.0, cp + (xi - mu0) - slack)
        cm = max(0.0, cm + (mu0 - xi) - slack)
        c_plus[i] = cp
        c_minus[i] = cm
        if cp > decision or cm > decision:
            signals.append(i)
    return c_plus, c_minus, signals
