import numpy as np
import pytest

from jfibsem_dat.read import RawFibsemData


def test_memmap_realised(fake_path):
    realised = RawFibsemData.from_filepath(fake_path, False)
    mmap = RawFibsemData.from_filepath(fake_path, True)
    assert np.allclose(mmap.data[:], realised.data)


def test_trunc_pads(trunc_fake_path):
    data = RawFibsemData.from_filepath(trunc_fake_path).data
    assert data[-1, -1, -1] == 0


def test_trunc_mmap_errors(trunc_fake_path):
    with pytest.raises(ValueError):
        RawFibsemData.from_filepath(trunc_fake_path, True).data


def test_trunc_mmap_recovers(trunc_fake_path):
    data = RawFibsemData.from_filepath(trunc_fake_path, True, True).data
    assert data[-1, -1, -1] == 0
