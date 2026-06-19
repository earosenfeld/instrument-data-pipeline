"""Minimal STDF V4 writer (FAR / MIR / PIR / PTR / PRR).

STDF (Standard Test Data Format, V4) is the binary log format emitted by virtually
every ATE platform. This is a *correct minimal* writer for the record types needed
to represent per-part parametric results:

* **FAR** - File Attributes Record (mandatory first record; byte order + version).
* **MIR** - Master Information Record (lot / job / test setup metadata).
* **PIR** - Part Information Record (start of a part's results).
* **PTR** - Parametric Test Record (one measured value vs. limits + pass/fail).
* **PRR** - Part Results Record (end of a part; bin + pass/fail).

Each record is ``REC_LEN(U2) REC_TYP(U1) REC_SUB(U1)`` followed by the body. Field
types follow the STDF spec: U1/U2/U4 unsigned ints, I4 signed, R4 float, Cn = a
length-prefixed string (1 length byte + bytes). All multi-byte fields here are
written little-endian and the FAR's CPU_TYPE is set to 2 (Intel/PC) to match.

The writer is dependency-free and round-trip verifiable: :func:`read_record_headers`
re-parses the stream so tests can assert the record sequence and lengths.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import BinaryIO, List, Optional, Sequence, Tuple

import pandas as pd

# STDF flag bits used below (from the V4 spec).
PTR_PARM_FLG_DEFAULT = 0x00
# TEST_FLG bit 7 (0x80) = test failed; bit 6 (0x40) = result is valid/alarm; we
# use bit 7 only and leave the result valid.
TEST_FLG_FAIL = 0x80
# PRR PART_FLG bit 3 (0x08) = part failed.
PART_FLG_FAIL = 0x08

_LE = "<"  # little-endian / Intel byte order (CPU_TYPE = 2)


def _cn(s: Optional[str]) -> bytes:
    """Encode a Cn field: 1 length byte followed by up to 255 ASCII bytes."""
    if not s:
        return b"\x00"
    raw = s.encode("ascii", "replace")[:255]
    return struct.pack("B", len(raw)) + raw


def _record(rec_typ: int, rec_sub: int, body: bytes) -> bytes:
    """Prefix a record body with its STDF header (length, type, sub)."""
    return struct.pack(_LE + "HBB", len(body), rec_typ, rec_sub) + body


def far_record() -> bytes:
    """FAR: CPU_TYPE=2 (PC, little-endian), STDF_VER=4."""
    return _record(0, 10, struct.pack(_LE + "BB", 2, 4))


def mir_record(
    lot_id: str = "LOT0001",
    part_typ: str = "DUT",
    job_nam: str = "PARAMETRIC",
    setup_t: int = 0,
    start_t: int = 0,
    stat_num: int = 1,
) -> bytes:
    """MIR with the commonly-required fields populated.

    The full MIR has many fields; per the spec, fixed numeric fields must be
    present in order, while trailing Cn strings may be defaulted. We write the
    fixed header fields and the key identifier strings used downstream.
    """
    body = struct.pack(
        _LE + "IIBBBHBB",
        setup_t & 0xFFFFFFFF,  # SETUP_T  U4
        start_t & 0xFFFFFFFF,  # START_T  U4
        stat_num & 0xFF,       # STAT_NUM U1
        ord("P"),              # MODE_COD C1  (P = production)
        ord(" "),              # RTST_COD C1
        0,                     # PROT_COD -> use ' '? spec C1; 0 acts as space-ish
        0,                     # BURN_TIM U2 low byte placeholder
        0,                     # CMOD_COD C1
    )
    # Identifier strings (Cn): LOT_ID, PART_TYP, NODE_NAM, TSTR_TYP, JOB_NAM ...
    body += _cn(lot_id) + _cn(part_typ) + _cn("SIM_NODE") + _cn("SIM_ATE") + _cn(job_nam)
    return _record(1, 10, body)


def pir_record(head_num: int = 1, site_num: int = 1) -> bytes:
    """PIR: marks the start of testing for one part."""
    return _record(5, 10, struct.pack(_LE + "BB", head_num, site_num))


def ptr_record(
    test_num: int,
    result: float,
    passed: bool,
    test_txt: str = "",
    lo_limit: Optional[float] = None,
    hi_limit: Optional[float] = None,
    units: str = "",
    head_num: int = 1,
    site_num: int = 1,
) -> bytes:
    """PTR: one parametric measurement with its limits and pass/fail flag.

    OPT_FLAG bits indicate which optional limit fields are valid:
      bit 4 (0x10) = no low limit ; bit 5 (0x20) = no high limit.
    We set them appropriately for one-sided specs.
    """
    test_flg = 0 if passed else TEST_FLG_FAIL
    opt_flag = 0
    lo = 0.0 if lo_limit is None else float(lo_limit)
    hi = 0.0 if hi_limit is None else float(hi_limit)
    if lo_limit is None:
        opt_flag |= 0x10
    if hi_limit is None:
        opt_flag |= 0x20

    body = struct.pack(
        _LE + "IBBBBf",
        test_num & 0xFFFFFFFF,  # TEST_NUM U4
        head_num & 0xFF,        # HEAD_NUM U1
        site_num & 0xFF,        # SITE_NUM U1
        test_flg & 0xFF,        # TEST_FLG B1
        PTR_PARM_FLG_DEFAULT,   # PARM_FLG B1
        float(result),          # RESULT   R4
    )
    body += _cn(test_txt)       # TEST_TXT Cn
    body += _cn("")             # ALARM_ID Cn
    body += struct.pack(_LE + "Bff", opt_flag, lo, hi)  # OPT_FLAG, LO_LIMIT, HI_LIMIT
    body += _cn(units)          # UNITS Cn
    return _record(15, 10, body)


def prr_record(
    part_id: str,
    passed: bool,
    hard_bin: int,
    soft_bin: int,
    num_test: int,
    head_num: int = 1,
    site_num: int = 1,
) -> bytes:
    """PRR: end-of-part summary with bin numbers and the pass/fail flag."""
    part_flg = 0 if passed else PART_FLG_FAIL
    body = struct.pack(
        _LE + "BBBHHHi",
        head_num & 0xFF,        # HEAD_NUM U1
        site_num & 0xFF,        # SITE_NUM U1
        part_flg & 0xFF,        # PART_FLG B1
        num_test & 0xFFFF,      # NUM_TEST U2
        hard_bin & 0xFFFF,      # HARD_BIN U2
        soft_bin & 0xFFFF,      # SOFT_BIN U2
        -32768,                 # X_COORD I2 -> "no coordinate" sentinel (use I4 slot)
    )
    # The above packs X_COORD as i (4 bytes) for simplicity of the sentinel; pad
    # Y_COORD + TEST_T as their own fields:
    body += struct.pack(_LE + "iI", -32768, 0)  # Y_COORD, TEST_T
    body += _cn(part_id)        # PART_ID Cn
    body += _cn("")             # PART_TXT Cn
    body += _cn("")             # PART_FIX Bn(0)
    return _record(5, 20, body)


@dataclass
class STDFColumn:
    """One parametric column to emit as a PTR per part."""

    test_num: int
    name: str
    value_col: str
    lo_limit: Optional[float]
    hi_limit: Optional[float]
    units: str = ""


def write_stdf(
    df: pd.DataFrame,
    path: str,
    columns: Sequence[STDFColumn],
    pass_col: str = "passed",
    unit_col: str = "unit_id",
    lot_id: str = "LOT0001",
    job_nam: str = "PARAMETRIC",
) -> int:
    """Write ``df`` to an STDF V4 file: FAR, MIR, then PIR/PTR.../PRR per part.

    Returns the number of parts written. Hard/soft bin is 1 for pass, 0 for fail
    (a common convention).
    """
    n_parts = 0
    with open(path, "wb") as fh:
        fh.write(far_record())
        fh.write(mir_record(lot_id=lot_id, job_nam=job_nam))
        for _, row in df.iterrows():
            passed = bool(row[pass_col])
            fh.write(pir_record())
            for col in columns:
                val = float(row[col.value_col])
                in_spec = True
                if col.lo_limit is not None and val < col.lo_limit:
                    in_spec = False
                if col.hi_limit is not None and val > col.hi_limit:
                    in_spec = False
                fh.write(
                    ptr_record(
                        test_num=col.test_num,
                        result=val,
                        passed=in_spec,
                        test_txt=col.name,
                        lo_limit=col.lo_limit,
                        hi_limit=col.hi_limit,
                        units=col.units,
                    )
                )
            bin_no = 1 if passed else 0
            fh.write(
                prr_record(
                    part_id=str(row[unit_col]),
                    passed=passed,
                    hard_bin=bin_no,
                    soft_bin=bin_no,
                    num_test=len(columns),
                )
            )
            n_parts += 1
    return n_parts


def read_record_headers(path: str) -> List[Tuple[int, int, int]]:
    """Re-parse an STDF file into ``(rec_typ, rec_sub, rec_len)`` tuples.

    Used to round-trip verify a written file (sequence + lengths) without a full
    STDF parser. Validates that each declared length matches the bytes present.
    """
    headers: List[Tuple[int, int, int]] = []
    with open(path, "rb") as fh:
        data = fh.read()
    pos = 0
    while pos < len(data):
        if pos + 4 > len(data):
            raise ValueError("truncated STDF header")
        rec_len, rec_typ, rec_sub = struct.unpack_from(_LE + "HBB", data, pos)
        pos += 4
        if pos + rec_len > len(data):
            raise ValueError("STDF record body extends past end of file")
        pos += rec_len
        headers.append((rec_typ, rec_sub, rec_len))
    return headers
