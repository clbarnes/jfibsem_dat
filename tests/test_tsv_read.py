import numpy as np
import pytest

from jfibsem_dat.core.parse_tsv import (
    HeaderParser,
    ParsedData,
    dat_to_hdf5,
    dat_to_hdf5_meta,
    hdf5_to_bytes,
    hdf5_to_bytes_meta,
    write_header,
)
from jfibsem_dat.read import HEADER_LENGTH, METADATA_VERSIONS

from .conftest import HEADER_PATHS

versions = sorted(set(METADATA_VERSIONS).intersection(HEADER_PATHS))


@pytest.mark.parametrize("version", versions)
def test_can_parse(version):
    fpath = HEADER_PATHS[version]
    parser = HeaderParser()
    result = parser.parse_file(fpath)
    assert len(result) > 80


@pytest.mark.parametrize("version", versions)
def test_can_write(version):
    fpath = HEADER_PATHS[version]
    parser = HeaderParser()
    result = parser.parse_file(fpath)
    written = write_header(result)
    assert set(written) != {b"\0"}


@pytest.mark.parametrize("version", versions)
def test_can_roundtrip(version):
    fpath = HEADER_PATHS[version]
    with open(fpath, "rb") as f:
        b = f.read(HEADER_LENGTH)
    parser = HeaderParser()
    result = parser.parse_bytes(b)
    written = write_header(result)
    assert written == b


@pytest.mark.parametrize("version", versions)
def test_can_write_hdf5(version, tmpdir):
    dat_path = HEADER_PATHS[version]
    h5_path = tmpdir / "data.hdf5"
    dat_to_hdf5_meta(dat_path, h5_path)


@pytest.mark.parametrize("version", versions)
def test_can_roundtrip_hdf5_meta(version, tmpdir):
    dat_path = HEADER_PATHS[version]
    parser = HeaderParser()
    parsed = parser.parse_file(dat_path)
    h5_path = tmpdir / "data.hdf5"
    dat_to_hdf5_meta(dat_path, h5_path)
    roundtripped = hdf5_to_bytes_meta(h5_path)
    rt_parsed = parser.parse_bytes(roundtripped)
    for k, v1 in parsed.items():
        assert k in rt_parsed
        assert np.array_equal(v1, rt_parsed[k])
    with open(dat_path, "rb") as f:
        raw = f.read(HEADER_LENGTH)
    assert roundtripped == raw


def test_can_read_data(real_path):
    _ = ParsedData.from_file(real_path)


def test_can_roundtrip_hdf5(real_path, tmpdir):
    h5_path = tmpdir / "data.hdf5"
    dat_to_hdf5(real_path, h5_path)
    roundtripped = hdf5_to_bytes(h5_path)
    with open(real_path, "rb") as f:
        orig = f.read()

    assert len(orig) == len(roundtripped)

    # excerpt where things are likely to go wrong with byte & axis ordering
    assert orig[1023:1030] == roundtripped[1023:1030]
    assert orig[-256:] == roundtripped[-256:]
    assert orig == roundtripped
