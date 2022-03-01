import hashlib
from pathlib import Path

import numpy as np
import pytest

TEST_DIR = Path(__file__).resolve().parent
FIXTURE_DIR = TEST_DIR / "fixtures"
BLOCKSIZE = 2**20

STEM = "Merlin-6281_19-08-09_120426_0-0-0"


def md5sum(fpath):
    md5 = hashlib.md5()
    with open(fpath, "rb") as f:
        while True:
            data = f.read(BLOCKSIZE)
            if not data:
                break
            md5.update(data)
    return md5.hexdigest()


@pytest.fixture(scope="session")
def fake_path(tmp_path_factory):
    header_path = FIXTURE_DIR / (STEM + ".header")
    # 2 channels; order doesn't matter
    shape = (2, 10_000, 15_000)
    dtype = np.dtype("uint16")
    rand = np.random.RandomState(1991)
    data = rand.randint(0, np.iinfo(dtype).max, size=shape, dtype=dtype)
    path = Path(tmp_path_factory.mktemp("fake_dats")) / "rand-2c-16b.dat"
    header_bytes = header_path.read_bytes()
    path.write_bytes(header_bytes + data.tobytes())
    return path


def name_append(path: Path, s: str):
    return path.parent / (path.stem + s + path.suffix)


@pytest.fixture(scope="session")
def trunc_fake_path(fake_path):
    path = name_append(fake_path, "_trunc")
    dat_size = fake_path.stat().st_size
    trunc_size = int(dat_size * 0.9)
    with fake_path.open("rb") as src, path.open("wb") as tgt:
        tgt.write(src.read(trunc_size))
    return path


@pytest.fixture(scope="session")
def real_path():
    dat_path = FIXTURE_DIR / (STEM + ".dat")
    if not dat_path.exists():
        pytest.skip("Reference .dat file not found")

    dat_md5 = "ca5d342ef389ab212d523b134144199b"
    if not md5sum(dat_path) == dat_md5:
        pytest.skip("Reference .dat file does not match expected")

    return dat_path


@pytest.fixture(scope="session")
def trunc_real_path(real_path):
    dat_size = real_path.stat().st_size
    trunc_size = int(dat_size * 0.9)
    trunc_path = name_append(real_path, "_trunc")
    if not trunc_path.exists() or trunc_path.stat().st_size != trunc_size:
        with real_path.open("rb") as src, trunc_path.open("wb") as tgt:
            tgt.write(src.read(trunc_size))
    return trunc_path
