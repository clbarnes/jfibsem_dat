#!/usr/bin/env python
from __future__ import annotations

import dataclasses as dc
import datetime as dt
import io
import json
import logging
import os
import typing as tp
from enum import IntEnum
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)

MAGIC_NUM = 3_555_587_570
I16_MAX = np.iinfo("int16").max
HALF_I16_MAX_UP = I16_MAX // 2 + 1
DEFAULT_BYTE_ORDER = ">"
DEFAULT_AXIS_ORDER = "F"
HEADER_LENGTH = 1024

# I don't know why this exists, but it's in the ref impl...
RESOLUTION_SCALE = 2.54e7


@dc.dataclass
class StagePosition:
    x: float  # mm
    y: float  # mm
    z: float  # mm
    t: float  # degrees
    r: float  # degrees
    m: float  # mm


class FIBMode(IntEnum):
    SEM = 0
    FIB = 1
    MILLING = 2
    SEM_AND_FIB = 3
    MILL_AND_SEM = 4
    SEM_DRIFT_CORRECTION = 5
    FIB_DRIFT_CORRECTION = 6
    NO_BEAM = 7
    EXTERNAL = 8
    EXTERNAL_AND_SEM = 9


class MillingPidMeasured(IntEnum):
    SPECIMEN = 0
    BEAMDUMP_1 = 1
    BEAMDUMP_2 = 2
    MILL = 3
    SPECIMEN_PLUS = 4

    @classmethod
    def from_int(cls, i):
        try:
            return cls(i)
        except ValueError:
            logger.warning("Unknown MillingPidMeasured value, using raw int %s", i)
            return i


@dc.dataclass
class MillingPID:
    is_on: bool
    measured: MillingPidMeasured
    target: float
    target_slope: float
    p: float
    i: float
    d: float


class FileType(IntEnum):
    ZEISS_NEON = 1


class MetadataEncoder(json.JSONEncoder):
    def default(self, o: tp.Any) -> tp.Any:
        if hasattr(o, "dtype") and hasattr(o, "tolist"):
            return o.tolist()
        elif isinstance(o, dt.date):
            return o.isoformat()
        return super().default(o)


@dc.dataclass
class MetadataV8:
    # v1+
    file_magic_num: int
    file_version: int
    file_type: FileType
    sw_date: dt.date
    time_step: float
    n_channels: int
    is_8bit: bool
    # v1
    # v2-6
    # v7+
    scaling: np.ndarray
    # v1+
    resolution_xy: np.ndarray
    # v1-3
    # v4+
    oversampling: int
    # v1+
    zeiss_scan_speed: int
    # v1-3
    # v4+
    scan_rate: float
    frameline_rampdown_ratio: float
    x_coil_voltage: tp.Tuple[float, float]
    detector_voltage: tp.Tuple[float, float]
    decimating_factor: int
    # v1+
    analogue_inputs: tp.Tuple[bool, bool, bool, bool]
    notes: str
    # v1-2
    # v3+
    detector_names: tp.Tuple[str, str, str, str]
    magnification: float
    pixel_size: float  # nm
    working_distance: float  # mm
    EHT: float  # kV
    n_SEM_apertures: int
    is_high_current: bool
    SEM_probe_current: float  # A
    SEM_scan_rotation: float  # degrees
    chamber_vacuum: float
    electron_gun_vacuum: float
    SEM_shift_xy: tp.Tuple[float, float]
    SEM_stigmation_xy: tp.Tuple[float, float]
    SEM_aperture_alignment_xy: tp.Tuple[float, float]
    stage_position: StagePosition
    brightness: tp.Tuple[float, float]
    contrast: tp.Tuple[float, float]
    FIB_mode: FIBMode
    FIB_focus: float  # kV
    n_FIB_probes: int
    FIB_current: float
    FIB_rotation: float
    FIB_aperture_alignment_xy: tp.Tuple[float, float]
    FIB_stigmation_xy: tp.Tuple[float, float]
    FIB_shift_xy: tp.Tuple[float, float]
    # v5-8
    milling_resolution_xy: int
    milling_size_xy: tp.Tuple[float, float]  # um
    milling_angle_upper_left: float  # degrees
    milling_angle_upper_right: float  # degrees
    milling_line_time: float  # seconds
    FIB_field_of_view: float  # um
    milling_lines_per_image: int
    milling_pid: MillingPID
    machine_id: str
    SEM_specimen_current: float  # nA
    # v6-7
    # v8
    beam_dump_2_current: float  # nA
    milling_current: float  # nA
    # v1+
    file_length: int  # bytes

    @classmethod
    @tp.no_type_check
    def from_bytes(cls, b: bytes):
        def rd(dtype, index, shape=None):
            return read_from_bytes(
                b,
                dtype,
                index,
                shape,
            )

        out = cls(
            file_magic_num=rd("uint32", 0),
            file_version=rd("uint16", 4),
            file_type=FileType(rd("uint16", 6)),
            sw_date=rd(dt.date, 8),
            time_step=rd("float64", 24),
            n_channels=rd("uint8", 32),
            is_8bit=rd(bool, 33),
            scaling=rd("float32", 36, (4, 2)),
            resolution_xy=rd("uint32", 100, 2),
            oversampling=rd("uint16", 108),
            zeiss_scan_speed=rd("uint8", 111),
            scan_rate=rd("float32", 112),
            frameline_rampdown_ratio=rd("float32", 116),
            x_coil_voltage=rd("float32", 120, 2),
            detector_voltage=rd("float32", 128, 2),
            decimating_factor=rd("uint16", 136),
            analogue_inputs=rd(bool, 151, 4),
            notes=rd(str, 180, 200),
            detector_names=tuple(
                rd(str, idx, length)
                for idx, length in [(380, 10), (390, 18), (410, 20), (430, 20)]
            ),
            magnification=rd("float32", 460),
            pixel_size=rd("float32", 464),  # nm
            working_distance=rd("float32", 468),  # mm
            EHT=rd("float32", 472),  # kV
            n_SEM_apertures=rd("uint8", 480),
            is_high_current=rd(bool, 481),
            SEM_probe_current=rd("float32", 490),  # A
            SEM_scan_rotation=rd("float32", 494),  # degrees
            chamber_vacuum=rd("float32", 498),
            electron_gun_vacuum=rd("float32", 498),
            SEM_shift_xy=rd("float32", 510, 2),
            SEM_stigmation_xy=rd("float32", 518, 2),
            SEM_aperture_alignment_xy=rd("float32", 526, 2),
            stage_position=StagePosition(
                x=rd("float32", 534),
                y=rd("float32", 538),
                z=rd("float32", 542),
                t=rd("float32", 546),
                r=rd("float32", 550),
                m=rd("float32", 554),
            ),
            brightness=(
                rd("float32", 560),
                rd("float32", 568),
            ),
            contrast=(
                rd("float32", 564),
                rd("float32", 572),
            ),
            FIB_mode=FIBMode(rd("uint8", 600)),
            FIB_focus=rd("float32", 605),  # kV
            n_FIB_probes=rd("uint8", 608),
            FIB_current=rd("float32", 620),
            FIB_rotation=rd("float32", 624),
            FIB_aperture_alignment_xy=rd("float32", 628),
            FIB_stigmation_xy=rd("float32", 636),
            FIB_shift_xy=rd("float32", 644),
            milling_resolution_xy=rd("uint32", 652, 2),
            milling_size_xy=rd("float32", 660, 2),  # um
            milling_angle_upper_left=rd("float32", 668),  # degrees
            milling_angle_upper_right=rd("float32", 672),  # degrees
            milling_line_time=rd("float32", 676),  # seconds
            FIB_field_of_view=rd("float32", 680),  # um
            milling_lines_per_image=rd("uint16", 684),
            milling_pid=MillingPID(
                is_on=rd(bool, 686),
                measured=MillingPidMeasured.from_int(rd("uint8", 689)),
                target=rd("float32", 690),
                target_slope=rd("float32", 694),
                p=rd("float32", 698),
                i=rd("float32", 702),
                d=rd("float32", 706),
            ),
            machine_id=rd(str, 800, 30),
            SEM_specimen_current=rd("float32", 980),  # nA
            beam_dump_2_current=rd("float32", 882),  # nA
            milling_current=rd("float32", 886),  # nA
            file_length=rd("int64", 1000),  # bytes
        )

        if out.file_magic_num != MAGIC_NUM:
            raise ValueError("Invalid file magic number")

        if out.file_version != 8:
            raise ValueError("Incorrect file version")

        return out

    @classmethod
    def from_filelike(cls, f):
        return cls.from_bytes(f.read(HEADER_LENGTH))

    @classmethod
    def from_filepath(cls, fpath):
        logger.debug("Reading metadata from %s", fpath)
        with open(fpath, "rb") as f:
            return cls.from_filelike(f)

    def data_length(self):
        return self.channel_length() * self.n_channels

    def data_shape(self) -> tp.Tuple[int, int, int]:
        return (self.n_channels,) + tuple(self.resolution_xy)

    def channel_length(self):
        return np.product(self.resolution_xy.astype(int))

    def channel_size(self):
        size = self.channel_length()
        if not self.is_8bit:
            size *= 2
        return size

    def channel_to_idx(self, channel: int) -> tp.Optional[int]:
        if (
            channel not in range(len(self.analogue_inputs))
            or not self.analogue_inputs[channel]
        ):
            return None
        return sum(self.analogue_inputs[:channel])

    def scaled_resolution(self):
        return RESOLUTION_SCALE / self.pixel_size

    def to_json(self, **kwargs) -> str:
        return json.dumps(dc.asdict(self), cls=MetadataEncoder, **kwargs)


def infer_dtype(is_8bit, byte_order=DEFAULT_BYTE_ORDER):
    return np.dtype("uint8" if is_8bit else "int16").newbyteorder(byte_order)


def raw_read(
    f: tp.Union[Path, str, io.IOBase],
    shape: tp.Tuple[int, int, int],
    is_8bit: bool = False,
) -> np.ndarray:
    """Read the array directly.

    The array is returned in non-python order (for each channel, X is the first axis).
    You may want to do `raw_read(...).transpose((0, 2, 1))`
    to play nice with python tools like matplotlib.

    Parameters
    ----------
    f : tp.Union[Path, str, io.IOBase]
        File-like object or path to it.
    shape : tp.Tuple[int]
        Shape of the data, i.e. (n_channels, x_max, y_max)
    is_8bit : bool, optional
        Whether data is u8, by default False (i.e. i16)

    Returns
    -------
    np.ndarray
        Of the given shape, dtype, etc.
    """
    dtype = infer_dtype(is_8bit)
    expected_len = int(np.product(shape))

    try:
        current_pos = f.tell()
    except AttributeError:
        current_pos = 0

    offset = HEADER_LENGTH - current_pos
    arr = np.fromfile(f, dtype, count=expected_len, offset=offset)
    if len(arr) < expected_len:
        arr = np.concatenate([arr, np.full(expected_len - len(arr), 0, dtype)])
    return arr.reshape(
        shape,
        order=DEFAULT_AXIS_ORDER,
    )


def raw_memmap(
    f: tp.Union[Path, str, io.IOBase],
    shape: tp.Tuple[int, int, int],
    is_8bit: bool = False,
) -> np.memmap:
    """Memory-map the array directly.

    Like the numpy.memmap function,
    this does not handle closing the underlying file object.
    If you want to ensure the file is closed in a tidy fashion, use

    ```
    with open(some_fpath, "rb") as f:
        raw_memmap(f, some_shape, some_is_8bit)
    ```

    The array is returned in non-python order (for each channel, X is the first axis).
    You may want to do `raw_memmap(...).transpose((0, 2, 1))`
    to play nice with python tools like matplotlib.

    Parameters
    ----------
    f : tp.Union[Path, str, io.IOBase]
        File-like object or a path to it.
    shape : tp.Tuple[int]
        Shape of the data, i.e. (n_channels, x_max, y_max)
    is_8bit : bool, optional
        Whether data is u8, by default False (i.e. i16)

    Returns
    -------
    np.memmap
    """
    dtype = infer_dtype(is_8bit)
    return np.memmap(f, dtype, "r", HEADER_LENGTH, shape, DEFAULT_AXIS_ORDER)


class RawFibsemData:
    MAGIC_NUM = 3_555_587_570
    HEADER_LENGTH = HEADER_LENGTH

    def __init__(self, metadata: MetadataV8, data: np.ndarray, file_handle=None):
        self.metadata = metadata
        self.data = data
        self._file_handle = file_handle

    @classmethod
    def from_filelike(
        cls, f: io.IOBase, memmap=False, handle_close=True, read_if_truncated=False
    ):
        metadata = parse_metadata(f.read(cls.HEADER_LENGTH))
        shape = metadata.data_shape()
        if memmap:
            try:
                data = raw_memmap(f, shape, metadata.is_8bit)
            except ValueError as e:
                if (
                    "mmap length is greater than file size" in str(e)
                    and read_if_truncated
                ):
                    data = raw_read(f, shape, metadata.is_8bit)
                else:
                    raise e
        else:
            data = raw_read(f, shape, metadata.is_8bit)
            handle_close = False

        return cls(metadata, data.transpose((0, 2, 1)), f if handle_close else None)

    @classmethod
    def from_filepath(cls, fpath: os.PathLike, memmap=False, read_if_truncated=False):
        if memmap:
            return cls.from_filelike(open(fpath, "rb"), memmap, True, read_if_truncated)
        else:
            with open(fpath, "rb") as f:
                return cls.from_filelike(f, False)

    def close(self):
        if hasattr(self._file_handle, "close"):
            self._file_handle.close()
        self._file_handle = None
        return self

    def realize(self):
        if isinstance(self.data, np.memmap):
            self.data = self.data[:]

        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def scale(
        self,
        channels: tp.Optional[tp.List[int]] = None,
        calibration: tp.Optional[
            tp.List[tp.Optional[tp.Tuple[tp.List[float], tp.List[float]]]]
        ] = None,
        roi: tp.Optional[tp.Tuple[slice]] = None,
    ) -> tp.List[tp.Optional[Channel]]:
        """roi give as slices into numpy array (Y, X)"""
        if channels is None:
            channels = list(range(self.metadata.n_channels))
        elif not hasattr(channels, "__iter__"):
            logger.warning(
                "Received singleton for `channels` argument. Use a list instead."
            )
            channels = [channels]

        if calibration is None:
            calibration = [None] * len(channels)

        if roi is None:
            raw_slice = channels
        else:
            raw_slice = (channels, *roi)

        raw = self.data[raw_slice]
        scaling = self.metadata.scaling.T[channels]
        names = np.asarray(self.metadata.detector_names)[channels]
        exist = np.asarray(self.metadata.analogue_inputs, bool)[channels]

        out = []
        for raw_c, scaling_c, names_c, exist_c, calib in zip(
            raw, scaling, names, exist, calibration
        ):
            if not exist_c:
                channel = None
            elif calib is None:
                if self.metadata.is_8bit:
                    factor = (
                        self.metadata.scan_rate
                        / scaling_c[0]
                        / scaling_c[2]
                        / scaling_c[3]
                    )
                    scaled = (raw_c.astype("float32") * factor + scaling_c[1]).astype(
                        "int16"
                    )
                else:
                    scaled = (raw_c.astype("float32") - scaling_c[1]) * scaling_c[2]
                channel = Channel(names_c, raw_c, scaled)
            else:
                interp = interp1d(calib[0], calib[1], "cubic", fill_value="extrapolate")
                scaled = interp(raw_c.astype("float32")).astype("uint16")
                channel = Channel(names_c, raw_c, scaled)
            out.append(channel)

        return out


def i16_to_u16(array: np.ndarray):
    if array.dtype != np.dtype("int16"):
        logger.warning(
            "Array does not seem to be of type int16. Trying to convert anyway."
        )
    return (array.astype("float32") + HALF_I16_MAX_UP).astype("uint16")


@dc.dataclass
class Channel:
    name: str
    raw: np.ndarray
    electron_counts: np.ndarray
    # attributes: tp.Dict[str, tp.Any] = dc.field(default_factory=dict)


def parse_metadata(header_bytes: bytes):
    # todo: consider making into an object
    # which composes over different metadata versions
    magic_num = read_from_bytes(header_bytes, "uint32", 0)
    if magic_num != MAGIC_NUM:
        raise ValueError("Incorrect magic number")
    version = read_from_bytes(header_bytes, "uint16", 4)
    classes = {
        8: MetadataV8,
    }
    cls = classes[version]
    return cls.from_bytes(header_bytes)


def read_from_bytes(
    b: bytes,
    dtype: np.dtype,
    index=0,
    size=None,
    byte_order=DEFAULT_BYTE_ORDER,
    fill=None,
    axis_order=DEFAULT_AXIS_ORDER,
):
    """this is necessary because BytesIO does not behave like a file,
    despite literally being designed for that purpose

    Note size behaves differently for str: it reads `size` bytes
    and then converts it into unicode,
    which may not be `size` characters long, and then rstrips null characters.
    """
    if dtype == bytes:
        if size is None:
            sl = slice(index)
        else:
            sl = slice(index, index + size)
        return b[sl]
    elif dtype == str:
        bb = read_from_bytes(b, bytes, index, size, byte_order, fill, axis_order)
        return bb.decode("utf-8").rstrip("\x00")
    elif dtype == dt.date:
        if size is not None and size != 10:
            raise ValueError("Size must be 10 for dates")
        date_bytes = read_from_bytes(b, bytes, index, 10, fill=fill)
        date_str = date_bytes.decode("utf-8")
        # todo: check day/month order
        datetime = dt.datetime.strptime(date_str, "%d/%m/%Y")
        return datetime.date()
    # elif dtype == dt.datetime:
    #     rd = read_from_bytes(b, bytes, index, size, fill=fill)
    #     return dt.datetime.fromisoformat(rd.decode("utf-8"))

    try:
        dtype = np.dtype(dtype)
    except TypeError:
        raise TypeError("dtype must be bytes, str, datetime.date, or numpy-compatible")

    if byte_order is not None:
        dtype = dtype.newbyteorder(byte_order)

    if size is None:
        count = 1
        shape = None
    else:
        try:
            shape = tuple(size)
            count = np.prod(shape)
        except TypeError:
            count = size
            shape = None

    data = np.frombuffer(b, dtype, count, offset=index)
    if len(data) < count:
        if fill is None:
            raise RuntimeError(f"Could not read {count} items and no fill given")
        else:
            data = np.concatenate([data, np.full(count - len(data), fill, dtype)])

    if shape is None:
        if size is None:
            return data[0]
        return data
    else:
        return data.reshape(shape, order=axis_order)
