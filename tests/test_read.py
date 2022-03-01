import numpy as np
import pytest

from jfibsem_dat.read import (
    HEADER_LENGTH,
    METADATA_VERSIONS,
    RawFibsemData,
    parse_metadata,
)

from .conftest import HEADER_PATHS

versions = sorted(set(METADATA_VERSIONS).intersection(HEADER_PATHS))


@pytest.mark.parametrize("version", versions)
def test_metadata_read(version):
    METADATA_VERSIONS[version].from_filepath(HEADER_PATHS[version])


@pytest.mark.parametrize("version", versions)
def test_parse_metadata(version):
    fpath = HEADER_PATHS[version]
    clazz = METADATA_VERSIONS[version]
    with open(fpath, "rb") as f:
        header_bytes = f.read(HEADER_LENGTH)
    meta = parse_metadata(header_bytes)
    assert isinstance(meta, clazz)


@pytest.mark.parametrize("version", versions)
def test_memmap_realised(version, faker):
    fpath = HEADER_PATHS[version]
    tmp_fpath = faker.fake(fpath)
    realised = RawFibsemData.from_filepath(tmp_fpath, False)
    mmap = RawFibsemData.from_filepath(tmp_fpath, True)
    assert np.allclose(mmap.data[:], realised.data)


@pytest.mark.parametrize("version", versions)
def test_trunc_pads(version, faker):
    fpath = HEADER_PATHS[version]
    tmp_fpath = faker.fake(fpath, trunc=0.8)
    data = RawFibsemData.from_filepath(tmp_fpath).data
    assert data[-1, -1, -1] == 0


@pytest.mark.parametrize("version", versions)
def test_trunc_mmap_errors(version, faker):
    fpath = HEADER_PATHS[version]
    tmp_fpath = faker.fake(fpath, trunc=0.8)
    with pytest.raises(ValueError):
        RawFibsemData.from_filepath(tmp_fpath, True).data


@pytest.mark.parametrize("version", versions)
def test_trunc_mmap_recovers(version, faker):
    fpath = HEADER_PATHS[version]
    tmp_fpath = faker.fake(fpath, trunc=0.8)
    data = RawFibsemData.from_filepath(tmp_fpath, True, True).data
    assert data[-1, -1, -1] == 0
