from io import BytesIO
from os import PathLike
from typing import Any, TypeVar

import h5py

from ..core.parse_tsv import HEADER_LENGTH, HeaderParser

TMetadata = TypeVar("TMetadata", bound="MetadataBase")


class MetadataBase:
    ChanNum: int
    EightBit: int
    XResolution: int
    YResolution: int

    def __init__(self, **kwargs):
        self._other: dict[str, Any] = kwargs

    @classmethod
    def from_dat(cls, f) -> TMetadata:
        parser = HeaderParser()
        if isinstance(f, (str, PathLike)):
            return cls(**parser.parse_file(f))
        elif not isinstance(f, bytes):
            f = f.read(HEADER_LENGTH)
        return cls(**parser.parse_bytes(f))

    @classmethod
    def _from_h5py_group(cls, g: h5py.Group, group_name=None) -> TMetadata:
        if group_name is not None:
            g = g[group_name]
        return cls(**g.attrs)

    @classmethod
    def from_hdf5(cls, f, group_name=None) -> TMetadata:
        if isinstance(f, h5py.Group):
            return cls._from_h5py_group(f, group_name)
        if isinstance(f, bytes):
            f = BytesIO(f)
        with h5py.File(f) as h5:
            return cls._from_h5py_group(f, h5)
