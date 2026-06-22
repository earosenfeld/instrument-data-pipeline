#!/usr/bin/env python3
"""Generate publication-quality figures for the README from the REAL analysis API.

Every figure is produced by calling the production stats / SPC / yield / distribution
functions in ``etl/`` -- nothing here re-derives a control limit or a capability
index by hand. Run headless::

    .venv/bin/python scripts/make_figures.py

PNGs are written to ``assets/`` next to the repo root.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

# --- House style -----------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "savefig.bbox": "tight",
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#334155", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": "#e2e8f0", "grid.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.labelsize": 11, "legend.frameon": False, "lines.linewidth": 2.0,
})
PALETTE = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2"]

# --- Make the repo root importable so ``etl`` resolves regardless of CWD ----
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)

from etl.spc import imr_chart, western_electric_rules  # noqa: E402
from etl.advanced_spc import (  # noqa: E402
    ewma_chart,
    cusum_chart,
    shewhart_individuals_signals,
)
from etl.stats import capability_from_values  # noqa: E402
from etl.yield_analysis import pareto_failure_modes  # noqa: E402
from etl.distributions import (  # noqa: E402
    lognormal_leakage,
    generate_parametric_population,
    HIPOT_SPECS,
    PARAMETRIC_SPECS,
)

# Human-readable Western-Electric rule descriptions (ids match etl/spc.py).
WE_RULE_TEXT = {
    1: "Rule 1: point beyond 3sigma",
    2: "Rule 2: 2 of 3 beyond 2sigma",
    3: "Rule 3: 4 of 5 beyond 1sigma",
    4: "Rule 4: 8 in a row one side",
}


def _annotate(fig, text):
    """Small grey provenance note so the figure reads as engineering output."""
    fig.text(0.99, 0.005, text, ha="right", va="bottom",
             fontsize=7.5, color="#94a3b8")


# ---------------------------------------------------------------------------
# 1. SPC control chart (the money shot): I-MR individuals chart with planted
#    out-of-control excursions, real control limits, and WE rule violations.
# ---------------------------------------------------------------------------
def make_spc_chart():
    rng = np.random.default_rng(7)
    # A burn-in supply-current series (Amps) in good control, then two planted
    # excursions an operator would want flagged.
    n = 40
    series = rng.normal(0.500, 0.006, n)
    series[18] = 0.527   # single gross spike  -> Western-Electric rule 1
    series[19] = 0.523
    # A sustained upward shift near the end -> trips the 2-of-3 / 4-of-5 zone rules.
    series[31:] += 0.011

    chart = imr_chart(series).individuals  # real control limits from MRbar/d2
    x = np.arange(1, n + 1)

    violations = western_electric_rules(chart.values, chart.center, chart.sigma)
    flagged = {idx for idx, _ in violations}
    # Keep the most-severe rule per flagged point for a clean legend.
    rule_by_idx = {}
    for idx, rule in violations:
        rule_by_idx.setdefault(idx, rule)
        if rule == 1:
            rule_by_idx[idx] = 1

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.plot(x, chart.values, color=PALETTE[0], marker="o", markersize=5,
            markerfacecolor="white", markeredgecolor=PALETTE[0],
            markeredgewidth=1.5, zorder=3, label="Supply current")

    # Center line + 3-sigma control limits.
    ax.axhline(chart.center, color="#334155", lw=1.4, zorder=2)
    ax.axhline(chart.ucl, color=PALETTE[1], lw=1.3, ls="--", zorder=2)
    ax.axhline(chart.lcl, color=PALETTE[1], lw=1.3, ls="--", zorder=2)
    # Light sigma zones (Western-Electric A/B/C) for readability.
    for k, alpha in ((1, 0.045), (2, 0.045)):
        ax.axhspan(chart.center + (k - 1) * chart.sigma, chart.center + k * chart.sigma,
                   color=PALETTE[1], alpha=alpha, zorder=0)
        ax.axhspan(chart.center - k * chart.sigma, chart.center - (k - 1) * chart.sigma,
                   color=PALETTE[1], alpha=alpha, zorder=0)

    # Mark violations in red.
    if flagged:
        fx = [x[i] for i in sorted(flagged)]
        fy = [chart.values[i] for i in sorted(flagged)]
        ax.scatter(fx, fy, s=120, color=PALETTE[1], zorder=5,
                   edgecolor="white", linewidth=1.2,
                   label="Out-of-control (WE rule)")
        for i in sorted(flagged):
            ax.annotate(f"R{rule_by_idx[i]}", (x[i], chart.values[i]),
                        textcoords="offset points", xytext=(0, 11),
                        ha="center", fontsize=9, fontweight="bold",
                        color=PALETTE[1])

    # Right-edge labels for the limits.
    pad = (chart.ucl - chart.lcl) * 0.02
    ax.text(n + 0.4, chart.ucl, f"UCL {chart.ucl:.3f}", color=PALETTE[1],
            va="center", fontsize=9, fontweight="bold")
    ax.text(n + 0.4, chart.center, f"CL {chart.center:.3f}", color="#334155",
            va="center", fontsize=9, fontweight="bold")
    ax.text(n + 0.4, chart.lcl, f"LCL {chart.lcl:.3f}", color=PALETTE[1],
            va="center", fontsize=9, fontweight="bold")

    # Which rules actually fired, as a sub-caption.
    fired = sorted({r for _, r in violations})
    sub = "   ".join(WE_RULE_TEXT[r] for r in fired) if fired else "no violations"

    ax.set_title("SPC Individuals Chart — Burn-in Supply Current")
    ax.set_xlabel("Sample (unit order)")
    ax.set_ylabel("Supply current (A)")
    ax.set_xlim(0.3, n + 6)
    ax.legend(loc="lower left", fontsize=9)
    # Rules-fired caption as a tidy box pinned to the lower-right plot area.
    ax.text(0.985, 0.03, "Violations fired:\n" + sub.replace("   ", "\n"),
            transform=ax.transAxes, fontsize=8, color=PALETTE[1],
            va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fef2f2",
                      edgecolor=PALETTE[1], linewidth=0.7, alpha=0.95))
    _annotate(fig, "etl.spc.imr_chart + western_electric_rules")
    out = ASSETS / "spc_chart.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 1b. Small-shift detection: a slow burn-in drift that a Shewhart 3-sigma chart
#     is deaf to, caught by EWMA and CUSUM. Three stacked panels share the same
#     1-sigma-shifted supply-current stream so the contrast is direct.
# ---------------------------------------------------------------------------
def make_ewma_cusum_chart():
    rng = np.random.default_rng(0)
    # Burn-in supply current (A): in control at 0.500 +/- 0.006, then a small,
    # sustained ~1-sigma degradation from sample 30 onward (a creeping drift, not
    # a spike). target/sigma are the KNOWN Phase-I baseline.
    n = 60
    shift_idx = 30
    target, sigma = 0.500, 0.006
    series = rng.normal(target, sigma, n)
    series[shift_idx:] += 1.0 * sigma  # 1-sigma sustained shift

    x = np.arange(1, n + 1)

    # --- Real analysis API: all three charts on the known parameters. ---
    shew = shewhart_individuals_signals(series, target=target, sigma=sigma, L=3.0)
    ew = ewma_chart(series, lam=0.2, L=3.0, target=target, sigma=sigma)
    cu = cusum_chart(series, target=target, sigma=sigma, k=0.5, h=5.0)
    ewma_det = min(ew.signals) if ew.signals else None
    cusum_det = min(cu.signals) if cu.signals else None

    fig, (ax0, ax1, ax2) = plt.subplots(
        3, 1, figsize=(9.6, 9.4), sharex=True,
        gridspec_kw={"hspace": 0.18},
    )

    def _mark_shift(ax):
        ax.axvline(shift_idx + 1, color="#94a3b8", lw=1.2, ls=":", zorder=1)

    # ---- Panel 1: raw data with Shewhart 3-sigma limits (drift NOT caught) ----
    ucl, lcl = target + 3 * sigma, target - 3 * sigma
    ax0.plot(x, series, color=PALETTE[0], marker="o", markersize=4.5,
             markerfacecolor="white", markeredgecolor=PALETTE[0],
             markeredgewidth=1.3, zorder=3, label="Supply current")
    ax0.axhline(target, color="#334155", lw=1.3, zorder=2)
    ax0.axhline(ucl, color=PALETTE[1], lw=1.2, ls="--", zorder=2)
    ax0.axhline(lcl, color=PALETTE[1], lw=1.2, ls="--", zorder=2)
    _mark_shift(ax0)
    ax0.text(shift_idx + 1.4, lcl, "drift starts", color="#64748b",
             fontsize=8.5, va="bottom", ha="left")
    ax0.text(n + 0.5, ucl, "UCL 3$\\sigma$", color=PALETTE[1], va="center",
             fontsize=8.5, fontweight="bold")
    ax0.text(n + 0.5, lcl, "LCL 3$\\sigma$", color=PALETTE[1], va="center",
             fontsize=8.5, fontweight="bold")
    if shew:
        ax0.scatter([x[i] for i in shew], [series[i] for i in shew], s=110,
                    color=PALETTE[1], zorder=5, edgecolor="white", linewidth=1.2)
    shew_txt = ("no point beyond 3$\\sigma$ -> drift MISSED"
                if not shew else f"{len(shew)} pt(s) flagged")
    ax0.set_title("Shewhart 3$\\sigma$ Individuals — small drift slips through")
    ax0.set_ylabel("Supply current (A)")
    ax0.set_xlim(0.3, n + 6)
    ax0.text(0.015, 0.05, shew_txt, transform=ax0.transAxes, fontsize=9,
             color=PALETTE[1], va="bottom", ha="left", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="#fef2f2",
                       edgecolor=PALETTE[1], linewidth=0.7, alpha=0.95))
    ax0.legend(loc="upper left", fontsize=9)

    # ---- Panel 2: EWMA with time-varying limits (catches it). ----
    ax1.plot(x, ew.ewma, color=PALETTE[4], marker="o", markersize=4,
             markerfacecolor="white", markeredgecolor=PALETTE[4],
             markeredgewidth=1.2, zorder=3, label="EWMA ($\\lambda$=0.2)")
    ax1.axhline(ew.target, color="#334155", lw=1.3, zorder=2)
    ax1.plot(x, ew.upper, color=PALETTE[1], lw=1.2, ls="--", zorder=2,
             label="EWMA limits ($L$=3)")
    ax1.plot(x, ew.lower, color=PALETTE[1], lw=1.2, ls="--", zorder=2)
    _mark_shift(ax1)
    if ewma_det is not None:
        ax1.scatter([x[ewma_det]], [ew.ewma[ewma_det]], s=150, color=PALETTE[1],
                    zorder=6, edgecolor="white", linewidth=1.4)
        ax1.annotate(f"detect @ {ewma_det + 1}\n(+{ewma_det - shift_idx} after shift)",
                     (x[ewma_det], ew.ewma[ewma_det]), textcoords="offset points",
                     xytext=(10, -2), fontsize=9, fontweight="bold",
                     color=PALETTE[1], va="center",
                     arrowprops=dict(arrowstyle="->", color=PALETTE[1], lw=1.2))
    ax1.set_title("EWMA — small drift caught (limits widen to steady state)")
    ax1.set_ylabel("EWMA statistic (A)")
    ax1.legend(loc="upper left", fontsize=9)

    # ---- Panel 3: CUSUM tabular sums vs decision interval (catches it). ----
    ax2.plot(x, cu.c_plus, color=PALETTE[2], marker="o", markersize=4,
             markerfacecolor="white", markeredgecolor=PALETTE[2],
             markeredgewidth=1.2, zorder=3, label="$C^{+}$ (upper sum)")
    ax2.plot(x, cu.c_minus, color=PALETTE[3], marker="s", markersize=3.5,
             markerfacecolor="white", markeredgecolor=PALETTE[3],
             markeredgewidth=1.2, zorder=3, label="$C^{-}$ (lower sum)")
    ax2.axhline(cu.limit, color=PALETTE[1], lw=1.3, ls="--", zorder=2)
    ax2.text(n + 0.5, cu.limit, f"$H$ = {cu.h:g}$\\sigma$", color=PALETTE[1],
             va="center", fontsize=8.5, fontweight="bold")
    _mark_shift(ax2)
    if cusum_det is not None:
        ax2.scatter([x[cusum_det]], [cu.c_plus[cusum_det]], s=150,
                    color=PALETTE[1], zorder=6, edgecolor="white", linewidth=1.4)
        ax2.annotate(f"detect @ {cusum_det + 1}\n(+{cusum_det - shift_idx} after shift)",
                     (x[cusum_det], cu.c_plus[cusum_det]),
                     textcoords="offset points", xytext=(10, 4), fontsize=9,
                     fontweight="bold", color=PALETTE[1], va="bottom",
                     arrowprops=dict(arrowstyle="->", color=PALETTE[1], lw=1.2))
    ax2.set_title("CUSUM — small drift caught ($k$=0.5$\\sigma$, $H$=5$\\sigma$)")
    ax2.set_ylabel("Cumulative sum (A)")
    ax2.set_xlabel("Sample (unit order)")
    ax2.legend(loc="upper left", fontsize=9)

    _annotate(fig, "etl.advanced_spc.ewma_chart + cusum_chart")
    out = ASSETS / "ewma_cusum.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 2. Capability histogram: parametric voltage population with USL/LSL, a fitted
#    normal overlay, and a Cp/Cpk/Pp/Ppk text box from the real stats functions.
# ---------------------------------------------------------------------------
def make_capability_histogram():
    df = generate_parametric_population(n=2000, seed=11)
    spec = PARAMETRIC_SPECS["voltage_mV"]
    values = df["voltage_mV"].to_numpy()
    # Clip the rare gross fliers for the view window only; stats use full data.
    lo, hi = spec.lsl - 60, spec.usl + 60
    view = values[(values > lo) & (values < hi)]

    cap = capability_from_values(
        values, usl=spec.usl, lsl=spec.lsl, subgroup_size=5
    )

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.hist(view, bins=46, density=True, color=PALETTE[0], alpha=0.45,
            edgecolor="white", linewidth=0.5, label="Measured population")

    # Fitted normal overlay from the real mu/sigma.
    xs = np.linspace(lo, hi, 400)
    ax.plot(xs, scipy_stats.norm.pdf(xs, cap.mu, cap.sigma),
            color=PALETTE[4], lw=2.2, label="Fitted normal")

    # Spec limits + nominal.
    ax.axvline(spec.lsl, color=PALETTE[1], lw=1.6, ls="--")
    ax.axvline(spec.usl, color=PALETTE[1], lw=1.6, ls="--")
    ax.axvline(spec.nominal, color="#334155", lw=1.2, ls=":")
    ymax = ax.get_ylim()[1]
    ax.text(spec.lsl, ymax * 0.98, "LSL", color=PALETTE[1], ha="right",
            va="top", fontsize=9, fontweight="bold")
    ax.text(spec.usl, ymax * 0.98, "USL", color=PALETTE[1], ha="left",
            va="top", fontsize=9, fontweight="bold")

    # Capability box.
    box = (
        f"$C_p$ = {cap.cp:.2f}    $C_{{pk}}$ = {cap.cpk:.2f}\n"
        f"$P_p$ = {cap.pp:.2f}    $P_{{pk}}$ = {cap.ppk:.2f}\n"
        f"$\\mu$ = {cap.mu:.1f} mV   $\\sigma$ = {cap.sigma:.1f} mV\n"
        f"n = {cap.n}"
    )
    ax.text(0.025, 0.97, box, transform=ax.transAxes, va="top", ha="left",
            fontsize=10, family="DejaVu Sans",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc",
                      edgecolor="#334155", linewidth=0.8))

    ax.set_title("Process Capability — Parametric Supply Voltage")
    ax.set_xlabel(f"Voltage ({spec.units})")
    ax.set_ylabel("Probability density")
    ax.set_xlim(lo, hi)
    ax.legend(loc="center right", fontsize=9)
    _annotate(fig, "etl.stats.capability_from_values")
    out = ASSETS / "capability_histogram.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 3. Pareto chart of failure modes (bars desc + cumulative % line on twin axis).
# ---------------------------------------------------------------------------
def make_pareto():
    # A realistic mix of failure-mode tags from a line's reject bin.
    modes = (
        ["solder_open"] * 47 + ["leakage_high"] * 31 + ["overcurrent"] * 23
        + ["isolation_low"] * 14 + ["wavelength_oos"] * 9
        + ["thermal_runaway"] * 6 + ["power_low"] * 4 + ["misc"] * 2
    )
    rng = np.random.default_rng(3)
    rng.shuffle(modes)

    table = pareto_failure_modes(modes)  # real ranking + cumulative %
    labels = [str(i) for i in table.index]
    counts = table["count"].to_numpy()
    cum = table["cumulative_percent"].to_numpy()

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    bars = ax.bar(labels, counts, color=PALETTE[0], alpha=0.85,
                  edgecolor="white", linewidth=0.6, zorder=3)
    ax.bar_label(bars, padding=2, fontsize=9, color="#334155")
    ax.set_ylabel("Failure count", color=PALETTE[0])
    ax.set_xlabel("Failure mode")
    ax.set_title("Pareto of Failure Modes — Reject Bin")
    ax.tick_params(axis="x", rotation=30)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")

    ax2 = ax.twinx()
    ax2.grid(False)
    ax2.plot(labels, cum, color=PALETTE[1], marker="D", markersize=5,
             lw=2.0, zorder=4)
    ax2.axhline(80, color="#94a3b8", lw=1.0, ls="--", zorder=2)
    ax2.text(len(labels) - 0.5, 81.5, "80%", color="#64748b",
             ha="right", va="bottom", fontsize=8.5)
    ax2.set_ylabel("Cumulative %", color=PALETTE[1])
    ax2.set_ylim(0, 105)
    for x_i, y_i in zip(range(len(labels)), cum):
        ax2.annotate(f"{y_i:.0f}%", (x_i, y_i), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=8, color=PALETTE[1])

    _annotate(fig, "etl.yield_analysis.pareto_failure_modes")
    out = ASSETS / "pareto.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 4. Log-normal leakage-current population on a log x-axis with the spec limit
#    marked and the failing tail shaded.
# ---------------------------------------------------------------------------
def make_leakage_distribution():
    spec = HIPOT_SPECS["leakage_current_mA"]  # one-sided USL = 1.0 mA
    leakage = lognormal_leakage(
        n=4000, median=0.10, sigma_log=0.55,
        breakdown_rate=0.01, breakdown_mult=40.0,
        rng=np.random.default_rng(19),
    )
    usl = spec.usl
    fail_frac = float(np.mean(leakage > usl)) * 100.0

    # Log-spaced bins (decades of spread).
    lo = max(leakage.min(), 1e-3)
    hi = leakage.max()
    bins = np.logspace(np.log10(lo), np.log10(hi), 60)

    fig, ax = plt.subplots(figsize=(9.0, 4.7))
    counts, edges, patches = ax.hist(
        leakage, bins=bins, color=PALETTE[2], alpha=0.55,
        edgecolor="white", linewidth=0.4, zorder=3,
    )
    # Shade the failing tail (bins beyond USL) in red.
    for c_left, patch in zip(edges[:-1], patches):
        if c_left >= usl:
            patch.set_facecolor(PALETTE[1])
            patch.set_alpha(0.75)

    ax.set_xscale("log")
    ax.axvline(usl, color=PALETTE[1], lw=1.8, ls="--", zorder=4)
    ymax = ax.get_ylim()[1]
    ax.text(usl * 1.05, ymax * 0.92, f"USL = {usl:g} mA", color=PALETTE[1],
            ha="left", va="top", fontsize=10, fontweight="bold")
    ax.text(usl * 1.05, ymax * 0.78,
            f"failing tail: {fail_frac:.2f}%", color=PALETTE[1],
            ha="left", va="top", fontsize=9)

    ax.set_title("HiPot Leakage Current — Log-normal Population")
    ax.set_xlabel("Leakage current (mA, log scale)")
    ax.set_ylabel("Part count")
    _annotate(fig, "etl.distributions.lognormal_leakage")
    out = ASSETS / "leakage_distribution.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main():
    figures = [
        make_spc_chart(),
        make_ewma_cusum_chart(),
        make_capability_histogram(),
        make_pareto(),
        make_leakage_distribution(),
    ]
    print("Generated figures:")
    for p in figures:
        kb = p.stat().st_size / 1024
        print(f"  {p.relative_to(ROOT)}  ({kb:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
