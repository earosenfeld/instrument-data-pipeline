"""Round-trip tests for the binary STDF V4 writer."""

import struct

import pandas as pd
import pytest

from etl.stdf_writer import STDFColumn, read_record_headers, write_stdf

# STDF V4 (rec_typ, rec_sub) identifiers
FAR = (0, 10)
MIR = (1, 10)
PIR = (5, 10)
PTR = (15, 10)
PRR = (5, 20)


@pytest.fixture()
def sample():
    df = pd.DataFrame(
        {
            "unit_id": ["U001", "U002", "U003"],
            "vout": [3.28, 3.31, 3.62],
            "iddq": [0.012, 0.011, 0.013],
            "passed": [True, True, False],
        }
    )
    columns = [
        STDFColumn(test_num=100, name="vout", value_col="vout",
                   lo_limit=3.2, hi_limit=3.4, units="V"),
        STDFColumn(test_num=200, name="iddq", value_col="iddq",
                   lo_limit=None, hi_limit=0.02, units="A"),
    ]
    return df, columns


def test_round_trip_record_sequence(sample, tmp_path):
    df, columns = sample
    path = str(tmp_path / "lot.stdf")

    n = write_stdf(df, path, columns)
    assert n == 3

    headers = [(t, s) for t, s, _ in read_record_headers(path)]

    # File preamble: FAR then MIR.
    assert headers[0] == FAR
    assert headers[1] == MIR

    # Then per part: PIR, one PTR per column, PRR.
    per_part = [PIR] + [PTR] * len(columns) + [PRR]
    assert headers[2:] == per_part * len(df)


def test_declared_lengths_match_bytes(sample, tmp_path):
    df, columns = sample
    path = str(tmp_path / "lot.stdf")
    write_stdf(df, path, columns)

    # read_record_headers raises if any record's declared length does not
    # match the bytes present; reaching the end cleanly is the assertion.
    headers = read_record_headers(path)
    total = sum(4 + rec_len for _, _, rec_len in headers)
    import os
    assert total == os.path.getsize(path)


def test_ptr_results_survive_binary_encoding(sample, tmp_path):
    df, columns = sample
    path = str(tmp_path / "lot.stdf")
    write_stdf(df, path, columns)

    # Pull each PTR's RESULT field (little-endian float32 at offset 8 of the
    # record body, per STDF V4: TEST_NUM u4, HEAD u1, SITE u1, TEST_FLG u1,
    # PARM_FLG u1, RESULT r4).
    results = []
    with open(path, "rb") as fh:
        data = fh.read()
    pos = 0
    while pos < len(data):
        rec_len, rec_typ, rec_sub = struct.unpack_from("<HBB", data, pos)
        body = data[pos + 4 : pos + 4 + rec_len]
        if (rec_typ, rec_sub) == PTR:
            (test_num,) = struct.unpack_from("<I", body, 0)
            (result,) = struct.unpack_from("<f", body, 8)
            results.append((test_num, result))
        pos += 4 + rec_len

    expected = []
    for _, row in df.iterrows():
        for col in columns:
            expected.append((col.test_num, float(row[col.value_col])))

    assert len(results) == len(expected)
    for (tn_a, r_a), (tn_e, r_e) in zip(results, expected):
        assert tn_a == tn_e
        assert r_a == pytest.approx(r_e, rel=1e-6)


def test_truncated_file_rejected(sample, tmp_path):
    df, columns = sample
    path = str(tmp_path / "lot.stdf")
    write_stdf(df, path, columns)

    blob = open(path, "rb").read()
    open(path, "wb").write(blob[:-3])  # chop mid-record

    with pytest.raises(ValueError):
        read_record_headers(path)
