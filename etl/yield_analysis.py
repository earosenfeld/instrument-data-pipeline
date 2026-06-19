"""Yield analytics for production test data.

Standard manufacturing-quality metrics computed over per-unit pass/fail records:

* **First-pass yield (FPY)** -- fraction of units that pass every test on the
  first attempt (no retest/rework).
* **Yield by group** -- pass fraction broken out by test, parameter, or station.
* **DPMO** -- defects per million opportunities,
  ``DPMO = defects / (units * opportunities) * 1e6`` with the corresponding
  approximate sigma level.
* **Pareto** -- failure-mode counts ranked descending with cumulative percent,
  for "vital few" prioritisation.

These functions operate on plain pandas frames so they compose with the ETL
layer and the simulators alike.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


@dataclass
class DPMOResult:
    defects: int
    units: int
    opportunities: int
    dpmo: float
    yield_fraction: float
    sigma_level: float  # approximate process sigma (1.5-sigma shifted)


def first_pass_yield(
    df: pd.DataFrame,
    unit_col: str = "unit_id",
    pass_col: str = "passed",
) -> float:
    """First-pass yield: fraction of units passing *all* their tests.

    Each row is one test result for one unit; ``pass_col`` is boolean/0-1. A unit
    counts as a first-pass success only if every one of its rows passed.
    """
    if df.empty:
        return 0.0
    passed = df[pass_col].astype(bool)
    by_unit = passed.groupby(df[unit_col]).all()
    return float(by_unit.mean())


def yield_by(
    df: pd.DataFrame,
    group_col: str,
    pass_col: str = "passed",
) -> pd.Series:
    """Pass fraction for each value of ``group_col`` (test / parameter / station).

    Returned series is sorted ascending by yield so the worst offenders surface
    first.
    """
    if df.empty:
        return pd.Series(dtype=float)
    passed = df[pass_col].astype(bool)
    result = passed.groupby(df[group_col]).mean()
    return result.sort_values()


def dpmo(defects: int, units: int, opportunities: int) -> DPMOResult:
    """Defects Per Million Opportunities and the approximate sigma level.

    ``opportunities`` is the number of independent ways a single unit can fail
    (e.g. number of distinct test parameters). The reported sigma level uses the
    conventional 1.5-sigma long-term shift, so a yield of 99.99966 percent maps
    to roughly 6 sigma / 3.4 DPMO.
    """
    if units <= 0 or opportunities <= 0:
        raise ValueError("units and opportunities must be positive")
    if defects < 0:
        raise ValueError("defects must be non-negative")

    total_opps = units * opportunities
    dpmo_val = defects / total_opps * 1_000_000
    yield_fraction = 1.0 - defects / total_opps

    # Approximate sigma level: z-score of the yield, plus the 1.5-sigma shift.
    if yield_fraction >= 1.0:
        sigma_level = math.inf
    elif yield_fraction <= 0.0:
        sigma_level = 0.0
    else:
        sigma_level = float(scipy_stats.norm.ppf(yield_fraction) + 1.5)

    return DPMOResult(
        defects=int(defects),
        units=int(units),
        opportunities=int(opportunities),
        dpmo=dpmo_val,
        yield_fraction=yield_fraction,
        sigma_level=sigma_level,
    )


def pareto_failure_modes(
    failure_modes: Sequence[str],
) -> pd.DataFrame:
    """Pareto table of failure modes.

    Returns a frame indexed by failure mode with columns ``count``, ``percent``
    and ``cumulative_percent``, ordered by descending count (ties broken by
    name for determinism).
    """
    modes = [m for m in failure_modes if m is not None and str(m) != ""]
    if not modes:
        return pd.DataFrame(
            columns=["count", "percent", "cumulative_percent"]
        )

    counts = pd.Series(modes).value_counts()
    # Stable, deterministic ordering: by count desc, then label asc.
    counts = counts.sort_index().sort_values(ascending=False, kind="stable")
    total = counts.sum()
    percent = counts / total * 100.0
    cumulative = percent.cumsum()

    out = pd.DataFrame(
        {
            "count": counts.astype(int),
            "percent": percent,
            "cumulative_percent": cumulative,
        }
    )
    out.index.name = "failure_mode"
    return out


def yield_summary(
    df: pd.DataFrame,
    unit_col: str = "unit_id",
    pass_col: str = "passed",
    param_col: str = "parameter",
    station_col: Optional[str] = "station",
    failure_mode_col: Optional[str] = "failure_mode",
) -> Dict[str, object]:
    """Convenience roll-up: FPY, per-parameter/-station yield, DPMO, Pareto.

    ``opportunities`` for DPMO is taken as the number of distinct parameters.
    Missing optional columns are skipped gracefully.
    """
    summary: Dict[str, object] = {}
    summary["first_pass_yield"] = first_pass_yield(df, unit_col, pass_col)

    if param_col in df.columns:
        summary["yield_by_parameter"] = yield_by(df, param_col, pass_col).to_dict()
        opportunities = int(df[param_col].nunique())
    else:
        opportunities = 1

    if station_col and station_col in df.columns:
        summary["yield_by_station"] = yield_by(df, station_col, pass_col).to_dict()

    units = int(df[unit_col].nunique())
    defects = int((~df[pass_col].astype(bool)).sum())
    summary["dpmo"] = dpmo(defects, units, opportunities).__dict__

    if failure_mode_col and failure_mode_col in df.columns:
        failed = df.loc[~df[pass_col].astype(bool), failure_mode_col]
        pareto = pareto_failure_modes(failed.tolist())
        summary["pareto"] = pareto.reset_index().to_dict(orient="records")

    return summary
