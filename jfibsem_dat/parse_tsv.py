import typing as tp
from io import BytesIO
from pathlib import Path

import h5py
import numpy as np
from frozendict import frozendict

from .read import DEFAULT_AXIS_ORDER, DEFAULT_BYTE_ORDER, HEADER_LENGTH

here = Path(__file__).resolve().parent
spec_dir = here / "specs"


def read_value(
    buffer: bytes,
    dtype: np.dtype,
    offset: int = 0,
    shape: tp.Optional[tuple[int, ...]] = None,
):
    if not shape:
        reshape = None
        count = 1
    else:
        try:
            reshape = tuple(shape)
            count = np.prod(reshape)
        except TypeError:
            count = shape
            reshape = None

    data = np.frombuffer(buffer, dtype, count, offset=offset)
    if len(data) < count:
        raise RuntimeError(f"Count not read {count} items from byte {offset}")
    if not reshape:
        return data[0]
    return data.reshape(reshape, order=DEFAULT_AXIS_ORDER)


class SpecTuple(tp.NamedTuple):
    name: str
    dtype: np.dtype
    offset: int
    shape: tp.Optional[tuple[tp.Union[int, str], ...]]

    def realise_shape(self, meta: dict[str, int]) -> tp.Optional[tuple[int]]:
        if not self.shape or self.shape == (0,):
            return None

        out = []
        for item in self.shape:
            if isinstance(item, int):
                out.append(item)
            else:
                out.append(meta[item])
        return tuple(out)

    @classmethod
    def from_line(cls, line: str):
        items = line.strip().split("\t")
        shape = []
        for item in items[3].split(","):
            item = item.strip()
            try:
                shape.append(int(item))
            except ValueError:
                shape.append(item)

        return cls(items[0], np.dtype(items[1]), int(items[2]), tuple(shape))

    @classmethod
    def from_file(cls, path, skip_header=True):
        with open(path) as f:
            if skip_header:
                next(f)
            for line in f:
                yield cls.from_line(line)

    def read_into(self, f, out=None):
        if out is None:
            out = dict()
        if self.name not in out:
            out[self.name] = read_value(
                f, self.dtype, self.offset, self.realise_shape(out)
            )
        return out


SPECS = frozendict(
    {
        int(tsv.stem[1:]): tuple(SpecTuple.from_file(tsv))
        for tsv in spec_dir.glob("*.tsv")
    }
)


class HeaderParser:
    spec_cache = None

    def _parse_with_version(self, b: bytes, version: int):
        spec = SPECS[version]
        out = dict()
        for line in spec:
            line.read_into(b, out)
        return out

    def _parse_core(self, b: bytes):
        return self._parse_with_version(b, 0)

    def parse_bytes(self, b: bytes):
        d = self._parse_core(b)
        return frozendict(self._parse_with_version(b, d["FileVersion"]))

    def parse_file(self, fpath: Path):
        with open(fpath, "rb") as f:
            b = f.read(HEADER_LENGTH)
            return self.parse_bytes(b)


def write_header(data: dict[str, tp.Any]):
    buffer = BytesIO(b"\0" * HEADER_LENGTH)
    for name, dtype, offset, _ in SPECS[data["FileVersion"]]:
        item = data[name]
        if not isinstance(item, np.ndarray):
            item = np.asarray(item, dtype=dtype, order=DEFAULT_AXIS_ORDER)
        b = item.tobytes(DEFAULT_AXIS_ORDER)
        buffer.seek(offset)
        buffer.write(b)

    return buffer.getvalue()


def dat_to_hdf5_meta(dat_path: Path, hdf5_path: Path, hdf5_group=None):
    if hdf5_group is None:
        hdf5_group = "/"
    parser = HeaderParser()
    with open(dat_path, "rb") as f:
        b = f.read(HEADER_LENGTH)
        header = parser.parse_bytes(b)

    with h5py.File(hdf5_path, "a") as h5:
        g = h5.require_group(hdf5_group)
        g.attrs.update(header)
        g.attrs["RawHeader"] = np.frombuffer(b, dtype="uint8")


def hdf5_to_bytes_meta(hdf5_path: Path, hdf5_group=None) -> bytes:
    if hdf5_group is None:
        hdf5_group = "/"

    with h5py.File(hdf5_path) as h5:
        g = h5[hdf5_group]
        return write_header(g.attrs)


def read_data(b: bytes) -> tuple[dict[str, tp.Any], np.ndarray]:
    parser = HeaderParser()
    meta = parser.parse_bytes(b)
    shape = (meta["ChanNum"], meta["XResolution"], meta["YResolution"])
    dtype = np.dtype("u1" if meta["EightBit"] else ">i2")
    data = read_value(b, dtype, HEADER_LENGTH, shape)
    return meta, data


def dat_to_hdf5(dat_path: Path, hdf5_path: Path, hdf5_group=None, inputs=None):
    all_inputs = [1, 2, 3, 4]
    if inputs is not None:
        if np.isscalar(inputs):
            inputs = [inputs]
        if set(inputs) - set(all_inputs):
            raise ValueError("Invalid inputs: must be 1-4 inclusive")

    if hdf5_group is None:
        hdf5_group = "/"
    with open(dat_path, "rb") as f:
        meta, data = read_data(f.read())

    ds_to_channel = dict()
    max_channel = 0
    for input_id in all_inputs:
        ds = f"AI{input_id}"
        exists = bool(meta[ds])

        if inputs is not None:
            if input_id not in inputs:
                continue
            if not exists:
                raise ValueError(f"Requested input {input_id} does not exist")
        if exists:
            ds_to_channel[ds] = max_channel
            max_channel += 1

    with h5py.File(hdf5_path, "a") as h5:
        g = h5.require_group(hdf5_group)
        g.attrs.update(meta)
        # g.attrs["RawHeader"] = np.frombuffer(b, dtype="uint8")
        for ds, channel_idx in ds_to_channel.items():
            g.create_dataset(ds, data=data[channel_idx])


def hdf5_to_bytes(hdf5_path, hdf5_group=None):
    if hdf5_group is None:
        hdf5_group = "/"

    with h5py.File(hdf5_path) as h5:
        g = h5[hdf5_group]
        header = write_header(g.attrs)
        to_stack = []
        for input_id in range(1, 5):
            ds_name = f"AI{input_id}"
            if ds_name not in g:
                continue
            to_stack.append(g[ds_name][:])

    stacked = np.stack(to_stack, axis=0)
    dtype = stacked.dtype.newbyteorder(DEFAULT_BYTE_ORDER)
    b = np.asarray(stacked, dtype, order="F").tobytes(order="F")
    return header + b
