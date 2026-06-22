"""Small-shift SPC: EWMA and CUSUM control charts (plus Gage R&R).

A Shewhart 3-sigma chart only reacts to a single point that lands beyond the
control limits, so it is *deliberately* deaf to a small, sustained drift: a
0.5-1 sigma offset (slow burn-in degradation, a creeping bias, a warming
fixture) keeps every individual reading comfortably inside +/-3 sigma yet shifts
the whole stream. The two charts here close that gap by accumulating evidence
across consecutive points instead of judging each one in isolation:

* **EWMA** (:func:`ewma_chart`) -- an exponentially weighted moving average. Each
  point is ``z_i = lam * x_i + (1 - lam) * z_{i-1}``, a low-pass filter whose
  memory is set by ``lam`` (small ``lam`` = long memory = sensitive to small
  shifts). Its variance grows from 0 and plateaus, so the control limits are
  *time-varying*, flaring out from the target and settling at the steady-state
  width ``L * sigma * sqrt(lam / (2 - lam))``.

* **CUSUM** (:func:`cusum_chart`) -- a tabular two-sided cumulative sum. It sums
  the running deviation from target after subtracting a slack ``k*sigma`` (half
  the shift you want to catch, in sigma) and resets to zero whenever the sum
  would go negative, so a centred process keeps both sums pinned near zero while
  a sustained offset makes one sum ramp until it crosses the decision interval
  ``h*sigma``.

Both return the plotted statistic, the control limit(s), and the list of flagged
(out-of-control) indices, mirroring the located-violation style of
:mod:`etl.spc` so a signal can be *placed in time*, not merely plotted.

A small :func:`average_run_length` helper (Monte-Carlo ARL) and an ANOVA
:func:`gage_rr` (%GRR / ndc) measurement-systems-analysis routine round out the
module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# EWMA
# ---------------------------------------------------------------------------
@dataclass
class EWMAChart:
    """EWMA statistic with its (time-varying) control limits and signals.

    Attributes
    ----------
    ewma:
        The EWMA statistic ``z_i`` at each point.
    target:
        Centre line ``mu0`` the chart is referenced to.
    sigma:
        Process sigma used to scale the limits.
    upper, lower:
        Per-point upper / lower control limits (arrays the same length as the
        input). They flare out from the target and approach the steady-state
        half-width :pyattr:`steady_state_halfwidth`.
    lam, L:
        Smoothing weight and control-limit width (in sigma) used.
    signals:
        Indices where ``z_i`` fell outside ``[lower_i, upper_i]``.
    """

    ewma: np.ndarray
    target: float
    sigma: float
    upper: np.ndarray
    lower: np.ndarray
    lam: float
    L: float
    signals: List[int]

    @property
    def steady_state_halfwidth(self) -> float:
        """Asymptotic limit half-width ``L * sigma * sqrt(lam/(2-lam))``."""
        return self.L * self.sigma * np.sqrt(self.lam / (2.0 - self.lam))


def ewma_chart(
    values: Sequence[float],
    lam: float = 0.2,
    L: float = 3.0,
    target: Optional[float] = None,
    sigma: Optional[float] = None,
) -> EWMAChart:
    """Exponentially Weighted Moving Average control chart.

    Recursion (with ``z_{-1} = target``)::

        z_i = lam * x_i + (1 - lam) * z_{i-1}

    Time-varying control limits about the target::

        target +/- L * sigma * sqrt( (lam / (2 - lam)) * (1 - (1 - lam)^(2(i+1))) )

    The variance factor starts small and grows to the steady-state
    ``lam / (2 - lam)``, so the limits widen with ``i`` and settle. A point is
    flagged when its EWMA statistic falls outside its own limits.

    Parameters
    ----------
    values:
        Ordered measurements (one-at-a-time individuals).
    lam:
        Smoothing weight ``0 < lam <= 1``. Smaller = more memory = more
        sensitive to small shifts (0.05-0.25 is typical); ``lam = 1`` reduces the
        chart to a Shewhart individuals chart.
    L:
        Control-limit width in sigma (commonly ~2.7-3.0).
    target:
        Centre line ``mu0``. Defaults to the sample mean when omitted.
    sigma:
        Process standard deviation. Defaults to the sample std (ddof=1) when
        omitted -- supply the known in-control sigma for true monitoring.
    """
    if not 0.0 < lam <= 1.0:
        raise ValueError("lam must be in (0, 1]")
    if L <= 0:
        raise ValueError("L (limit width) must be > 0")

    x = np.asarray(values, dtype=float)
    if x.size == 0:
        empty = np.array([], dtype=float)
        return EWMAChart(empty, 0.0, 0.0, empty, empty, lam, L, [])

    mu0 = float(np.mean(x)) if target is None else float(target)
    sd = float(np.std(x, ddof=1)) if sigma is None else float(sigma)

    z = np.empty_like(x)
    upper = np.empty_like(x)
    lower = np.empty_like(x)
    signals: List[int] = []

    prev = mu0
    for i, xi in enumerate(x):
        prev = lam * xi + (1.0 - lam) * prev
        z[i] = prev
        # Variance grows with the observation count (i+1 because i is 0-based).
        var_factor = (lam / (2.0 - lam)) * (1.0 - (1.0 - lam) ** (2 * (i + 1)))
        halfwidth = L * sd * np.sqrt(var_factor)
        upper[i] = mu0 + halfwidth
        lower[i] = mu0 - halfwidth
        if z[i] > upper[i] or z[i] < lower[i]:
            signals.append(i)

    return EWMAChart(
        ewma=z,
        target=mu0,
        sigma=sd,
        upper=upper,
        lower=lower,
        lam=lam,
        L=L,
        signals=signals,
    )


# ---------------------------------------------------------------------------
# CUSUM
# ---------------------------------------------------------------------------
@dataclass
class CUSUMChart:
    """Tabular two-sided CUSUM statistics, decision limit, and signals.

    Attributes
    ----------
    c_plus, c_minus:
        The upper / lower one-sided cumulative sums at each point (both >= 0).
    target:
        Centre value ``mu0`` deviations are measured from.
    sigma:
        Process sigma used to scale the slack and decision interval.
    limit:
        Decision interval ``h * sigma``; a sum exceeding it is out-of-control.
    k, h:
        Reference value (slack) and decision interval, both in sigma units.
    signals:
        Indices where ``c_plus`` or ``c_minus`` exceeded ``limit``.
    """

    c_plus: np.ndarray
    c_minus: np.ndarray
    target: float
    sigma: float
    limit: float
    k: float
    h: float
    signals: List[int]


def cusum_chart(
    values: Sequence[float],
    target: float,
    sigma: float,
    k: float = 0.5,
    h: float = 5.0,
) -> CUSUMChart:
    """Tabular two-sided CUSUM control chart.

    With ``C+_0 = C-_0 = 0`` and slack ``K = k*sigma``::

        C+_i = max(0, C+_{i-1} + (x_i - (target + K)))
        C-_i = max(0, C-_{i-1} + ((target - K) - x_i))

    A point is flagged when ``C+_i`` or ``C-_i`` exceeds the decision interval
    ``H = h*sigma``. The ``max(0, .)`` floor is the reset: a centred process
    keeps both sums pinned at zero, so the chart only ramps under a genuine
    sustained offset, and it ramps fastest for a shift of size ``2k*sigma`` (the
    canonical ``k = 0.5``, ``h = 4..5`` design targets a 1-sigma shift with an
    in-control ARL near 465).

    Parameters
    ----------
    values:
        Ordered measurements.
    target:
        In-control mean ``mu0`` (required -- a CUSUM is always referenced to a
        known target).
    sigma:
        In-control standard deviation (> 0).
    k:
        Reference value / slack in sigma (half the shift to detect).
    h:
        Decision interval in sigma.
    """
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    if k < 0:
        raise ValueError("k (slack) must be >= 0")
    if h <= 0:
        raise ValueError("h (decision interval) must be > 0")

    x = np.asarray(values, dtype=float)
    mu0 = float(target)
    slack = k * sigma
    limit = h * sigma

    if x.size == 0:
        empty = np.array([], dtype=float)
        return CUSUMChart(empty, empty, mu0, float(sigma), limit, k, h, [])

    c_plus = np.empty_like(x)
    c_minus = np.empty_like(x)
    signals: List[int] = []

    cp = 0.0
    cm = 0.0
    for i, xi in enumerate(x):
        cp = max(0.0, cp + (xi - (mu0 + slack)))
        cm = max(0.0, cm + ((mu0 - slack) - xi))
        c_plus[i] = cp
        c_minus[i] = cm
        if cp > limit or cm > limit:
            signals.append(i)

    return CUSUMChart(
        c_plus=c_plus,
        c_minus=c_minus,
        target=mu0,
        sigma=float(sigma),
        limit=limit,
        k=k,
        h=h,
        signals=signals,
    )


# ---------------------------------------------------------------------------
# Average Run Length (Monte-Carlo) -- a quick way to characterise a chart's
# sensitivity: ARL0 = mean points to a (false) alarm in control; ARL1 = mean
# points to detect a shift of ``shift`` sigma.
# ---------------------------------------------------------------------------
def average_run_length(
    chart: str,
    shift: float = 0.0,
    *,
    lam: float = 0.2,
    L: float = 3.0,
    k: float = 0.5,
    h: float = 5.0,
    n_runs: int = 2000,
    max_len: int = 2000,
    seed: Optional[int] = 0,
) -> float:
    """Monte-Carlo Average Run Length for an EWMA or CUSUM chart.

    Simulates standard-normal streams (target 0, sigma 1) with a step ``shift``
    (in sigma) applied from the first sample, and reports the mean number of
    points until the chart first signals. ``shift = 0`` gives the in-control
    ARL0 (bigger is better -- fewer false alarms); a non-zero ``shift`` gives the
    out-of-control ARL1 (smaller is better -- faster detection). Runs that never
    signal within ``max_len`` are counted as ``max_len`` (a censored lower bound,
    so a reported ARL near ``max_len`` means "rarely signals").

    Parameters
    ----------
    chart:
        ``"ewma"`` or ``"cusum"``.
    shift:
        Mean shift in sigma applied to the whole stream.
    n_runs, max_len, seed:
        Number of simulated streams, per-run length cap, and RNG seed.
    """
    chart = chart.lower()
    if chart not in ("ewma", "cusum"):
        raise ValueError("chart must be 'ewma' or 'cusum'")
    rng = np.random.default_rng(seed)
    run_lengths = np.empty(n_runs, dtype=float)

    for r in range(n_runs):
        x = rng.normal(shift, 1.0, max_len)
        if chart == "ewma":
            res = ewma_chart(x, lam=lam, L=L, target=0.0, sigma=1.0)
        else:
            res = cusum_chart(x, target=0.0, sigma=1.0, k=k, h=h)
        run_lengths[r] = (res.signals[0] + 1) if res.signals else max_len

    return float(np.mean(run_lengths))


# ---------------------------------------------------------------------------
# Gage R&R (ANOVA method) -- Measurement Systems Analysis. Partitions the
# measurement variation into part-to-part, repeatability (equipment) and
# reproducibility (operator), then reports %GRR and the number of distinct
# categories (ndc). A crossed study: every operator measures every part, each
# part ``n_replicates`` times.
# ---------------------------------------------------------------------------
@dataclass
class GageRRResult:
    """ANOVA Gage R&R variance components and study-level metrics.

    All ``var_*`` fields are variance components (sigma^2). ``pct_*`` are the
    percent contributions of each component's *standard deviation* to the total
    study standard deviation (the %study-variation convention, summed in
    quadrature -- not the variance-percent convention). ``ndc`` is the number of
    distinct categories.
    """

    var_repeatability: float   # equipment variation (EV), sigma^2
    var_reproducibility: float  # appraiser variation (AV), sigma^2
    var_grr: float             # EV + AV, the gage R&R variance
    var_part: float            # part-to-part variation (PV), sigma^2
    var_total: float           # total study variation, sigma^2
    pct_grr: float             # %GRR as % of total study sigma
    pct_repeatability: float
    pct_reproducibility: float
    pct_part: float
    ndc: float                 # number of distinct categories


def gage_rr(measurements: Sequence[Sequence[Sequence[float]]]) -> GageRRResult:
    """ANOVA-method Gage R&R from a crossed parts x operators x replicates study.

    ``measurements`` is indexed ``[operator][part][replicate]`` (a rectangular
    nested sequence: every operator measures every part the same number of times,
    ``>= 2`` replicates). The two-way ANOVA *with* the part*operator interaction
    is used; following the AIAG convention an interaction that is not significant
    (here: pooled when its mean square is below the repeatability mean square) is
    folded into repeatability so variance components stay non-negative.

    Returns
    -------
    GageRRResult
        Variance components plus %GRR and ``ndc``. ``ndc`` is
        ``floor(1.41 * sqrt(var_part / var_grr))`` (AIAG); a system with
        ``ndc >= 5`` is generally considered adequate, and ``%GRR <= 10%`` is
        good (``<= 30%`` marginal).
    """
    m = np.asarray(measurements, dtype=float)
    if m.ndim != 3:
        raise ValueError(
            "measurements must be 3-D [operator][part][replicate]"
        )
    o, p, r = m.shape
    if o < 1 or p < 2 or r < 2:
        raise ValueError(
            "need >=1 operator, >=2 parts and >=2 replicates per part"
        )

    grand = m.mean()
    n = o * p * r

    # Marginal means.
    part_means = m.mean(axis=(0, 2))       # length p
    op_means = m.mean(axis=(1, 2))         # length o
    cell_means = m.mean(axis=2)            # shape (o, p)

    # Sums of squares for a two-way (crossed) ANOVA with interaction.
    ss_part = o * r * np.sum((part_means - grand) ** 2)
    ss_op = p * r * np.sum((op_means - grand) ** 2)
    # Interaction SS: variation of cell means not explained by the main effects.
    ss_cells = r * np.sum((cell_means - grand) ** 2)
    ss_inter = ss_cells - ss_part - ss_op
    ss_total = np.sum((m - grand) ** 2)
    ss_equip = ss_total - ss_cells  # within-cell (pure replicate) error

    # Degrees of freedom.
    df_part = p - 1
    df_op = max(o - 1, 0)
    df_inter = df_part * df_op
    df_equip = o * p * (r - 1)

    ms_part = ss_part / df_part
    ms_op = ss_op / df_op if df_op > 0 else 0.0
    ms_inter = ss_inter / df_inter if df_inter > 0 else 0.0
    ms_equip = ss_equip / df_equip

    # Variance components (Expected-Mean-Square solution), floored at 0.
    var_repeat = max(ms_equip, 0.0)

    # Pool a non-significant interaction into repeatability (AIAG rule of thumb).
    if df_inter > 0 and ms_inter > ms_equip:
        var_inter = (ms_inter - ms_equip) / r
        var_op = max((ms_op - ms_inter) / (p * r), 0.0) if df_op > 0 else 0.0
    else:
        # Interaction not meaningful: pool into error, drop from reproducibility.
        pooled_ss = ss_inter + ss_equip
        pooled_df = df_inter + df_equip
        var_repeat = max(pooled_ss / pooled_df, 0.0) if pooled_df > 0 else 0.0
        var_inter = 0.0
        var_op = max((ms_op - var_repeat) / (p * r), 0.0) if df_op > 0 else 0.0

    var_reprod = var_op + var_inter
    var_grr = var_repeat + var_reprod
    var_part = max((ms_part - ms_equip) / (o * r), 0.0)
    var_total = var_grr + var_part

    def pct(component_var: float) -> float:
        if var_total <= 0:
            return 0.0
        return float(100.0 * np.sqrt(component_var / var_total))

    ndc = float(np.floor(1.41 * np.sqrt(var_part / var_grr))) if var_grr > 0 else np.inf

    return GageRRResult(
        var_repeatability=float(var_repeat),
        var_reproducibility=float(var_reprod),
        var_grr=float(var_grr),
        var_part=float(var_part),
        var_total=float(var_total),
        pct_grr=pct(var_grr),
        pct_repeatability=pct(var_repeat),
        pct_reproducibility=pct(var_reprod),
        pct_part=pct(var_part),
        ndc=ndc,
    )


def shewhart_individuals_signals(
    values: Sequence[float],
    target: float,
    sigma: float,
    L: float = 3.0,
) -> List[int]:
    """Indices a Shewhart individuals chart flags at ``target +/- L*sigma``.

    A small convenience used to contrast the small-shift charts above with a
    plain Shewhart 3-sigma rule (rule 1 only) referenced to the *known*
    in-control parameters -- the apples-to-apples baseline for a fixed target.
    For the full Western-Electric / Nelson rule set use :mod:`etl.spc`.
    """
    x = np.asarray(values, dtype=float)
    ucl = target + L * sigma
    lcl = target - L * sigma
    return [
        i for i, v in enumerate(x)
        if not np.isnan(v) and (v > ucl or v < lcl)
    ]
