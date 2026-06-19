"""Regenerate realistic per-test sample CSVs under ``data/raw/<test>/``.

The original sample files were all ``id,value`` with non-physical numbers. This
script writes one CSV per test type with the columns each ETL ingest actually
expects, populated from the physically realistic generators in
:mod:`etl.distributions`. Run as a module::

    python -m etl.generate_samples
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import pandas as pd

from etl import distributions as dist

RAW = os.path.join("data", "raw")
N = 50  # rows per sample file (small, committable)


def _timestamps(n: int) -> list:
    t0 = datetime(2024, 1, 1, 8, 0, 0)
    return [t0 + timedelta(seconds=2 * i) for i in range(n)]


def _result(passed) -> list:
    return ["PASS" if bool(p) else "FAIL" for p in passed]


def write_all() -> None:
    ts = _timestamps(N)

    hipot = dist.generate_hipot_population(N, seed=42)
    pd.DataFrame(
        {
            "id": hipot["unit_id"],
            "voltage": hipot["applied_voltage_kV"],
            "current": hipot["leakage_current_mA"],
            "test_time": ts,
            "result": _result(hipot["passed"]),
        }
    ).to_csv(os.path.join(RAW, "hipot", "sample_hipot.csv"), index=False)

    iso = dist.generate_isolation_population(N, seed=43)
    pd.DataFrame(
        {
            "id": iso["unit_id"],
            "resistance": iso["resistance_Mohm"],
            "voltage": 500.0,
            "test_time": ts,
            "result": _result(iso["passed"]),
        }
    ).to_csv(os.path.join(RAW, "isolation", "sample_isolation.csv"), index=False)

    laser = dist.generate_laser_population(N, seed=44)
    pd.DataFrame(
        {
            "id": laser["unit_id"],
            "power": laser["power_mW"],
            "wavelength": laser["wavelength_nm"],
            "test_time": ts,
            "result": _result(laser["passed"]),
        }
    ).to_csv(os.path.join(RAW, "laser", "sample_laser.csv"), index=False)

    param = dist.generate_parametric_population(N, seed=45)
    pd.DataFrame(
        {
            "id": param["unit_id"],
            "voltage": param["voltage_mV"],
            "current": param["current_mA"],
            "test_time": ts,
            "result": _result(param["passed"]),
        }
    ).to_csv(os.path.join(RAW, "parametric", "sample_parametric.csv"), index=False)

    # Burn-in zero-current screen: a single 'value' (zero-input leakage current,
    # arbitrary units 0-1000) per part, matching the BurnInZeroCurrent model.
    burn = dist.generate_burnin_population(N, seed=46)
    # Map junction temp tail into a 0-1000 "zero current" reading for the simple
    # single-value burn-in table while staying inside the documented range.
    value = (burn["supply_current_A"] * 1000.0).clip(0, 1000)
    pd.DataFrame({"id": burn["unit_id"], "value": value.round(2)}).to_csv(
        os.path.join(RAW, "burnin", "sample_burnin.csv"), index=False
    )

    # ICT sample retains the simple voltage/current schema used by the ETL
    # normalization test.
    pd.DataFrame(
        {
            "id": list(range(1, 6)),
            "voltage": [3.30, 3.31, 3.29, 3.28, 3.32],
            "current": [0.50, 0.51, 0.49, 0.50, 0.52],
            "test_time": ts[:5],
            "result": ["PASS"] * 5,
        }
    ).to_csv(os.path.join(RAW, "ict", "sample_ict.csv"), index=False)

    # Fixtures for the test-suite edge cases:
    # empty.csv: header only, zero data rows.
    pd.DataFrame({"id": [], "value": []}).to_csv(
        os.path.join(RAW, "burnin", "empty.csv"), index=False
    )
    # malformed.csv: ragged / non-numeric so ingest type-coercion raises.
    with open(os.path.join(RAW, "burnin", "malformed.csv"), "w", newline="") as fh:
        fh.write("id,value\n")
        fh.write("1,not_a_number\n")
        fh.write("oops\n")


if __name__ == "__main__":
    write_all()
    print("Sample CSVs regenerated under data/raw/")
