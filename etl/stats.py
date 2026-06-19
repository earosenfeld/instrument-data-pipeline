"""Shared process-capability statistics.

Single source of truth for process-capability indices (Cp, Cpk, Pp, Ppk, Cpm).
All simulation and analysis code imports from here -- there are intentionally
no inline capability formulas anywhere else in the codebase.

Definitions and conventions
----------------------------
Let mu = process mean, sigma = process standard deviation,
USL / LSL = upper / lower specification limits, T = target (nominal).

Two-sided specs::

    Cp  = (USL - LSL) / (6 * sigma)
    Cpk = min(USL - mu, mu - LSL) / (3 * sigma)
    Cpm = (USL - LSL) / (6 * sqrt(sigma**2 + (mu - T)**2))

* Cp / Cpk are *potential* / *actual* capability and conceptually use the
  WITHIN-subgroup (short-term) sigma estimated as ``sigma_within = Rbar / d2``.
* Pp / Ppk are *performance* indices and use the OVERALL (long-term) sigma,
  i.e. the ordinary sample standard deviation of all readings. The formulas are
  identical; only the sigma estimate differs. We expose both so the distinction
  is explicit rather than implied.
* Cpm penalises being off-target; with mu == T it reduces to Cp.

One-sided specs are handled honestly: if only an upper spec is given, only the
upper index ``CPU = (USL - mu) / (3*sigma)`` is defined and Cpk == CPU. We never
fabricate a symmetric limit to force a two-sided number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Control-chart constants (Montgomery, *Introduction to Statistical Quality
# Control*, Appendix VI). d2 converts an average range Rbar into an unbiased
# estimate of the within-subgroup sigma: sigma_within = Rbar / d2.
# ---------------------------------------------------------------------------
D2_CONSTANTS: Dict[int, float] = {
    2: 1.128,
    3: 1.693,
    4: 2.059,
    5: 2.326,
    6: 2.534,
    7: 2.704,
    8: 2.847,
    9: 2.970,
    10: 3.078,
}


@dataclass
class CapabilityResult:
    """Container for capability indices. Fields are ``None`` when undefined."""

    cp: Optional[float] = None
    cpk: Optional[float] = None
    pp: Optional[float] = None
    ppk: Optional[float] = None
    cpm: Optional[float] = None
    cpu: Optional[float] = None  # one-sided upper:  (USL - mu) / (3 sigma)
    cpl: Optional[float] = None  # one-sided lower:  (mu - LSL) / (3 sigma)
    mu: Optional[float] = None
    sigma: Optional[float] = None
    sigma_within: Optional[float] = None
    n: Optional[int] = None

    def to_dict(self) -> Dict[str, Optional[float]]:
        """Plain dict (handy for JSON serialisation)."""
        return asdict(self)


def _sample_std(values: np.ndarray) -> float:
    """Sample standard deviation (ddof=1), matching pandas ``.std()``."""
    return float(np.std(values, ddof=1))


def within_subgroup_sigma(rbar: float, n: int) -> float:
    """Within-subgroup sigma estimate ``Rbar / d2`` for subgroup size ``n``.

    Used for Cp / Cpk (short-term capability). Raises if ``n`` is outside the
    tabulated 2..10 range so callers cannot silently get a wrong constant.
    """
    if n not in D2_CONSTANTS:
        raise ValueError(
            f"subgroup size n={n} not in tabulated range 2..10 for d2 constant"
        )
    if rbar < 0:
        raise ValueError("average range Rbar must be non-negative")
    return rbar / D2_CONSTANTS[n]


def capability(
    mu: float,
    sigma: float,
    usl: Optional[float] = None,
    lsl: Optional[float] = None,
    target: Optional[float] = None,
) -> CapabilityResult:
    """Capability indices from explicit moments and spec limits.

    Parameters
    ----------
    mu, sigma:
        Process mean and standard deviation (sigma > 0). The supplied sigma
        determines whether the result is "potential" (pass a within-subgroup
        sigma) or "performance" (pass the overall sigma); this function does not
        assume which -- :func:`capability_from_values` fills both views.
    usl, lsl:
        Upper / lower spec limits. At least one must be provided. If only one is
        given the result is one-sided (Cpk == the relevant one-sided index).
    target:
        Target value for Cpm. Defaults to the spec midpoint for two-sided specs;
        Cpm is left undefined for one-sided specs (no agreed convention).
    """
    if usl is None and lsl is None:
        raise ValueError("at least one of USL / LSL is required")
    if sigma <= 0:
        raise ValueError("sigma must be > 0 to compute capability indices")
    if usl is not None and lsl is not None and usl <= lsl:
        raise ValueError("USL must be greater than LSL")

    res = CapabilityResult(mu=float(mu), sigma=float(sigma))

    cpu = (usl - mu) / (3.0 * sigma) if usl is not None else None
    cpl = (mu - lsl) / (3.0 * sigma) if lsl is not None else None
    res.cpu, res.cpl = cpu, cpl

    if usl is not None and lsl is not None:
        # Two-sided spec.
        res.cp = (usl - lsl) / (6.0 * sigma)
        res.cpk = min(cpu, cpl)
        t = float(target) if target is not None else (usl + lsl) / 2.0
        res.cpm = (usl - lsl) / (6.0 * math.sqrt(sigma ** 2 + (mu - t) ** 2))
    else:
        # One-sided spec: Cp / Cpm are undefined; Cpk is the lone one-sided index.
        res.cpk = cpu if cpu is not None else cpl

    return res


def capability_from_values(
    values: Sequence[float],
    usl: Optional[float] = None,
    lsl: Optional[float] = None,
    target: Optional[float] = None,
    subgroup_size: Optional[int] = None,
) -> CapabilityResult:
    """Capability indices computed directly from a sample of readings.

    Returns BOTH the performance view (Pp/Ppk/Cpm via overall sigma) and, when a
    ``subgroup_size`` is supplied, the potential view (Cp/Cpk via within-subgroup
    sigma = Rbar/d2). Cpm always uses the overall sigma.

    The within-subgroup sigma is estimated by chunking ``values`` into
    consecutive subgroups of ``subgroup_size``, averaging the subgroup ranges,
    and dividing by d2. If ``subgroup_size`` is omitted, Cp/Cpk fall back to the
    overall sigma and therefore equal Pp/Ppk (documented behaviour, not a bug).
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size < 2:
        raise ValueError("need at least 2 readings to estimate variation")

    mu = float(np.mean(arr))
    sigma_overall = _sample_std(arr)

    # --- Performance indices (overall / long-term sigma) ---
    perf = capability(mu, sigma_overall, usl=usl, lsl=lsl, target=target)
    res = CapabilityResult(
        pp=perf.cp,
        ppk=perf.cpk,
        cpm=perf.cpm,
        mu=mu,
        sigma=sigma_overall,
        n=int(arr.size),
    )

    # --- Potential indices (within-subgroup / short-term sigma) ---
    if subgroup_size and subgroup_size >= 2:
        n_full = arr.size // subgroup_size
        if n_full >= 1:
            groups = arr[: n_full * subgroup_size].reshape(n_full, subgroup_size)
            rbar = float(np.mean(groups.max(axis=1) - groups.min(axis=1)))
            sigma_w = within_subgroup_sigma(rbar, subgroup_size)
            res.sigma_within = sigma_w
            if sigma_w > 0:
                pot = capability(mu, sigma_w, usl=usl, lsl=lsl, target=target)
                res.cp, res.cpk = pot.cp, pot.cpk
                res.cpu, res.cpl = pot.cpu, pot.cpl
    if res.cp is None and res.cpk is None:
        # No subgroup structure: Cp/Cpk default to the overall-sigma values.
        res.cp, res.cpk = perf.cp, perf.cpk
        res.cpu, res.cpl = perf.cpu, perf.cpl

    return res
