import hashlib
import logging
from pathlib import Path
from urllib import request

import numpy as np
import pytest

logger = logging.getLogger(__name__)

TEST_DIR = Path(__file__).resolve().parent
FIXTURE_DIR = TEST_DIR / "fixtures"
BLOCKSIZE = 2**20

EXAMPLE_STEM = "FIBdeSEMAna_21-12-26_005024_0-0-0"
EXAMPLE_HOST = "https://neurophyla.mrc-lmb.cam.ac.uk/share/fibsem_example/"


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
    header_path = FIXTURE_DIR / (EXAMPLE_STEM + ".header")
    # 2 channels; order doesn't matter
    shape = (2, 14_464, 18_214)
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


def fetch(url, fpath, blocksize=100_000_000):
    logger.warning("Downloading FIBSEM example (first time only) at %s", url)
    with (
        request.urlopen(url) as req,
        open(fpath, "wb") as f,
    ):
        while True:
            b = req.read(blocksize)
            f.write(b)
            if len(b) != blocksize:
                break


@pytest.fixture(scope="session")
def real_path():
    dat_path = FIXTURE_DIR / (EXAMPLE_STEM + ".dat")
    if not dat_path.exists():
        fetch(f"{EXAMPLE_HOST}{EXAMPLE_STEM}.dat", dat_path)

    # FIBdeSEMAna
    dat_md5 = "753de6ea77acd4bd86166c459fe84006"
    # Merlin
    # dat_md5 = "ca5d342ef389ab212d523b134144199b"
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
