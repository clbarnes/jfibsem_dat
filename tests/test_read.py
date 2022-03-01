import numpy as np
import pytest

from jfibsem_dat.read import RawFibsemData


def test_memmap_realised(real_path):
    realised = RawFibsemData.from_filepath(real_path, False)
    mmap = RawFibsemData.from_filepath(real_path, True)
    assert np.allclose(mmap.data[:], realised.data)


def test_trunc_pads(trunc_real_path):
    data = RawFibsemData.from_filepath(trunc_real_path).data
    assert data[-1, -1, -1] == 0


def test_trunc_mmap_errors(trunc_real_path):
    with pytest.raises(ValueError):
        RawFibsemData.from_filepath(trunc_real_path, True).data


def test_trunc_mmap_recovers(trunc_real_path):
    data = RawFibsemData.from_filepath(trunc_real_path, True, True).data
    assert data[-1, -1, -1] == 0
