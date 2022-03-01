import fnmatch as fnm
import logging
import os
import typing as tp
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .read import DEFAULT_AXIS_ORDER, HEADER_LENGTH, MetadataV8, infer_dtype

logger = logging.getLogger(__name__)


def get_fpaths(dpath, glob, depth=0):
    for root, dpaths, fpaths in os.walk(dpath):
        matching = fnm.filter(fpaths, glob)
        logger.debug("%s of %s files in %s match", len(matching), len(fpaths), root)
        for match in sorted(matching):
            yield Path(root) / match
        if depth == 0:
            return
        depth -= 1
        dpaths.sort()


class CoreMetadata(tp.NamedTuple):
    pixel_size: float
    resolution_xy: tp.Tuple[float, float]
    analogue_inputs: tp.Tuple[bool, ...]
    is_8bit: bool

    @classmethod
    def from_filepath(cls, fpath):
        return cls.from_metadata(MetadataV8.from_filepath(fpath))

    @classmethod
    def from_metadata(cls, meta: MetadataV8):
        return cls(
            meta.pixel_size,
            tuple(meta.resolution_xy),
            tuple(meta.analogue_inputs),
            meta.is_8bit,
        )


def core_meta_set(fpaths, threads=None) -> tp.Set[CoreMetadata]:
    logger.info("Reading metadata for %s files", len(fpaths))
    with ThreadPoolExecutor(max_workers=threads) as pool:
        metadata_set = set(pool.map(CoreMetadata.from_filepath, fpaths))

    return metadata_set


def read_from_memmap(fpath, dtype, shape, slicing):
    with open(fpath, "rb") as f:
        return np.memmap(f, dtype, "r", HEADER_LENGTH, shape, DEFAULT_AXIS_ORDER)[
            slicing
        ]


class MultiDat:
    ndim = 4

    def __init__(
        self, fpaths: tp.List[Path], metadata: tp.Union[MetadataV8, CoreMetadata]
    ) -> None:
        self.fpaths = np.asarray(fpaths)
        self.metadata = CoreMetadata.from_metadata(metadata)
        self.n_channels = sum(self.metadata.analogue_inputs)
        # cxyz
        self.shape = (self.n_channels, *self.metadata.resolution_xy, len(self.fpaths))
        self.dtype = infer_dtype(self.metadata.is_8bit)

    @classmethod
    def from_directory(cls, dpath: Path, depth=-1, meta_threads=None):
        fpaths = list(get_fpaths(dpath, "*.dat", depth))
        logging.info("Got %s .dat files", len(fpaths))
        if meta_threads is not None and meta_threads < 0:
            meta = CoreMetadata.from_filepath(fpaths[0])
        else:
            metas = core_meta_set(fpaths, meta_threads)
            if len(metas) > 1:
                raise ValueError(
                    "Inconsistent .dat metadata:\n\t"
                    + "\n\t".join(str(m) for m in sorted(metas))
                )
            meta = metas.pop()
        return cls(fpaths, meta)

    def __getitem__(self, items) -> np.ndarray:
        if not isinstance(items, tuple) or len(items) != self.ndim:
            raise IndexError("Must use a 4-length tuple index")
        fpaths = self.fpaths[items[-1]]
        out = len(fpaths)
        shape = self.shape[:-1]
        slice_items = items[:-1]
        first = read_from_memmap(fpaths[0], self.dtype, shape, slice_items)
        out = np.empty_like(out, shape=(*first.shape, len(fpaths)))
        out[..., 0] = first
        for idx, fpath in enumerate(fpaths[1:], 1):
            out[..., idx] = read_from_memmap(fpath, self.dtype, shape, slice_items)
        return out
