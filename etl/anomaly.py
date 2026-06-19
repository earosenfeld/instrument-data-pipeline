"""Robust outlier / anomaly detection for parametric test readings.

Two non-parametric detectors that do not assume normality and resist masking by
the very outliers they hunt:

* **Robust z-score via MAD** -- ``z = 0.6745 * (x - median) / MAD``. The 0.6745
  factor makes MAD a consistent estimator of sigma for Gaussian data, so a
  threshold of ~3.5 is comparable to a 3.5-sigma rule but is not dragged around
  by outliers the way mean/std are.
* **Tukey / IQR fences** -- flag points outside
  ``[Q1 - k*IQR, Q3 + k*IQR]`` (k=1.5 outer-ish fence by convention).

An optional IsolationForest wrapper is provided for multivariate anomaly
detection when scikit-learn is installed; it degrades gracefully (raises a clear
error) if the dependency is absent, so the core suite never requires it.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

_MAD_SCALE = 0.6744897501960817  # Phi^{-1}(0.75); makes MAD ~ sigma for Gaussian.


def robust_zscores(values: Sequence[float]) -> np.ndarray:
    """Modified (MAD-based) z-scores.

    Returns an array the same length as ``values``. When MAD is zero (e.g. more
    than half the points are identical) it falls back to the mean-absolute
    deviation so the result is still finite and usable.
    """
    x = np.asarray(values, dtype=float)
    median = np.median(x)
    mad = np.median(np.abs(x - median))
    if mad > 0:
        return _MAD_SCALE * (x - median) / mad
    mean_ad = np.mean(np.abs(x - median))
    if mean_ad > 0:
        # 1.253314 = sqrt(pi/2); scales mean-abs-dev to sigma for Gaussian.
        return (x - median) / (1.2533141373155003 * mean_ad)
    return np.zeros_like(x)


def mad_outliers(values: Sequence[float], threshold: float = 3.5) -> np.ndarray:
    """Boolean mask of points whose |robust z| exceeds ``threshold``."""
    return np.abs(robust_zscores(values)) > threshold


def tukey_fences(values: Sequence[float], k: float = 1.5):
    """Return the ``(lower, upper)`` Tukey fences for ``values``."""
    x = np.asarray(values, dtype=float)
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def tukey_outliers(values: Sequence[float], k: float = 1.5) -> np.ndarray:
    """Boolean mask of points outside the Tukey/IQR fences."""
    x = np.asarray(values, dtype=float)
    lower, upper = tukey_fences(x, k)
    return (x < lower) | (x > upper)


def isolation_forest_outliers(
    matrix: np.ndarray,
    contamination: float = 0.02,
    random_state: int = 0,
) -> np.ndarray:
    """Optional multivariate anomaly detection via scikit-learn IsolationForest.

    ``matrix`` is ``(n_samples, n_features)``. Returns a boolean mask of anomalies.
    Raises ``ImportError`` with a helpful message if scikit-learn is unavailable.
    """
    try:
        from sklearn.ensemble import IsolationForest  # noqa: WPS433
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "isolation_forest_outliers requires scikit-learn; "
            "install it or use mad_outliers / tukey_outliers instead"
        ) from exc

    clf = IsolationForest(contamination=contamination, random_state=random_state)
    preds = clf.fit_predict(np.asarray(matrix, dtype=float))
    return preds == -1
