import hashlib
import logging
from pathlib import Path
from urllib import request

import numpy as np
import pytest

from jfibsem_dat.read import HEADER_LENGTH, parse_metadata

logger = logging.getLogger(__name__)

TEST_DIR = Path(__file__).resolve().parent
FIXTURE_DIR = TEST_DIR / "fixtures"
BLOCKSIZE = 2**20

EXAMPLE_STEM = "FIBdeSEMAna_21-12-26_005024_0-0-0"
EXAMPLE_HOST = "https://neurophyla.mrc-lmb.cam.ac.uk/share/fibsem_example/"


HEADER_PATHS = {
    8: FIXTURE_DIR / "Merlin-6281_19-08-09_120426_0-0-0.header",
}


def md5sum(fpath):
    md5 = hashlib.md5()
    with open(fpath, "rb") as f:
        while True:
            data = f.read(BLOCKSIZE)
            if not data:
                break
            md5.update(data)
    return md5.hexdigest()


def fake_data(
    header_path, out_path, blocksize=None, footer=None, trunc=None, seed=1991
):
    with open(header_path, "rb") as f:
        header_bytes = f.read(HEADER_LENGTH)

    meta = parse_metadata(header_bytes)
    shape = meta.data_shape()

    to_write = np.product(shape)
    if isinstance(footer, int):
        to_write += footer
        footer = None

    if trunc is not None:
        to_write = int(to_write * (1 - trunc))

    dtype = meta.dtype().newbyteorder("=")
    rng = np.random.default_rng(seed)
    if blocksize is None:
        blocksize = to_write

    def rand(size=blocksize):
        return rng.integers(
            np.iinfo(dtype).min,
            np.iinfo(dtype).max,
            size=size,
            dtype=dtype,
        )

    out_path.parent.mkdir(exist_ok=True, parents=True)
    with open(out_path, "wb") as f:
        f.write(header_bytes)
        while to_write > blocksize:
            f.write(rand().tobytes())
            to_write = int(to_write - blocksize)
        if to_write:
            f.write(rand(to_write).tobytes())
        if footer is not None:
            f.write(footer)


class FakeDataFactory:
    def __init__(self, tmp, seed=1991) -> None:
        self.tmp = Path(tmp).resolve()
        self.rng = np.random.default_rng(seed)

    def _random_seed(self):
        return self.rng.integers(np.iinfo("int16").max)

    def fake(self, header_fpath: Path, trunc=None):
        if trunc is None:
            name = header_fpath.with_suffix(".dat").name
        else:
            name = header_fpath.with_suffix("").name + f"_trunc{trunc}.dat"
        path = self.tmp / name
        if path.exists():
            return path

        fake_data(header_fpath, path, 10_000_000, seed=self._random_seed(), trunc=trunc)
        return path


@pytest.fixture(scope="session")
def faker(tmp_path_factory):
    tmp = Path(tmp_path_factory.mktemp("fake_dats"))
    return FakeDataFactory(tmp)


@pytest.fixture(scope="session")
def fake_path(tmp_path_factory):
    header_path = FIXTURE_DIR / (EXAMPLE_STEM + ".header")
    # 2 channels; order doesn't matter
    shape = (2, 14_464, 18_214)
    dtype = np.dtype("uint16")
    rand = np.random.RandomState(1991)
    data = rand.randint(0, np.iinfo(dtype).max, size=shape, dtype=dtype)
    path = Path(tmp_path_factory.mktemp("fake_dats")) / "rand-2c-16b.dat"
    with open(header_path, "rb") as f:
        header_bytes = f.read(HEADER_LENGTH)
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
    with request.urlopen(url) as req, open(fpath, "wb") as f:
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
