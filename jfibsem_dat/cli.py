import json
import logging
import typing as tp
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from .read import MetadataV8, RawFibsemData

logger = logging.getLogger(__name__)
CMAP = "gray_r"


def get_array(
    fpath,
    channel: tp.Optional[int],
    raw: bool = False,
    calibration_path: Path = None,
) -> tp.Tuple[np.ndarray, MetadataV8, int]:
    logging.basicConfig(level=logging.INFO)
    logger.info("Attempting to memmap %s", fpath)
    dat = RawFibsemData.from_filepath(fpath, True, True)
    meta = dat.metadata
    if channel is None:
        channel = np.nonzero(meta.analogue_inputs)[0][0]
        logger.info("Defaulting to channel %s", channel)

    idx = meta.channel_to_idx(channel)
    if idx is None:
        raise ValueError(f"Channel {channel} does not exist")

    if raw:
        logger.info("Using raw data")
        arr = dat.data[meta.channel_to_idx(channel)]
    else:
        kwargs = {}
        if calibration_path is not None:
            kwargs["calibration"] = [
                np.genfromtxt(
                    calibration_path, "float32", delimiter=",", autostrip=True
                ).T
            ]
        logger.info("Scaling data")
        channel_data = dat.scale([channel], **kwargs)[0]
        arr = channel_data.electron_counts

    return arr, meta, channel


def view_single(fpath, channel, raw=False, calibration_path=None):
    from matplotlib import pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib_scalebar.scalebar import ScaleBar

    arr, meta, channel = get_array(fpath, channel, raw, calibration_path)
    name = meta.detector_names[channel]

    logger.info("Displaying")
    fig: Figure = plt.figure()
    ax: Axes = fig.add_subplot()
    ax.set_title(f"{fpath}\n{name}")
    pos = ax.imshow(arr, cmap=CMAP)
    sc = ScaleBar(meta.pixel_size, "nm")
    logger.warning("Scale may be incorrect by factor of 2.54e7")
    ax.add_artist(sc)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    fig.colorbar(pos, ax=ax)
    plt.show()


def expand_paths(paths):
    for path in paths:
        if path.is_file():
            yield path
        elif path.is_dir():
            yield from sorted(path.glob("*.dat"))
        else:
            raise FileNotFoundError(f"Nonexistent path: {path}")


# def view_multi(fpaths, channel, raw):
#     arr, metadata1, channel = get_array(fpaths[0], channel, raw)
#     combined = np.empty_like(arr, shape=(len(fpaths), *arr.shape))
#     combined[0] = arr

#     def fn(f):
#         a, _, _ = get_array(f, channel, raw)
#         if a.shape != arr.shape or a.dtype != arr.dtype:
#             raise ValueError("Arrays are not compatible")
#         return a

#     with ThreadPoolExecutor() as pool:
#         for idx, arr in enumerate(pool.map(fn, fpaths[1:]), 1):
#             combined[idx] = arr

#     import napari
#     viewer = napari.view_image(combined)


def datview(args=None):
    parser = ArgumentParser(
        description=(
            "View a Janelia FIBSEM .dat file. "
            "The data can be scaled using the file's metadata, "
            "viewed --raw, or scaled using a --calibration CSV. "
            "Uses matplotlib."
        )
    )
    parser.add_argument("file", type=Path, help=".dat file to view")
    # parser.add_argument("file", nargs="*", type=Path, help=".dat file(s) to view")
    parser.add_argument(
        "-c",
        "--channel",
        type=int,
        help="Which channel to view (default first). Not all channels exist.",
    )
    parser.add_argument(
        "-C",
        "--calibration",
        type=Path,
        help="CSV file calibrating raw to scaled values",
    )
    parser.add_argument(
        "-r",
        "--raw",
        action="store_true",
        help="Show raw data rather than scaled electron counts.",
    )
    parsed = parser.parse_args(args)

    fpaths = list(expand_paths(parsed.file))
    if len(fpaths) == 0:
        logger.info("No .dat files given")
    elif len(fpaths) == 1:
        view_single(fpaths[0], parsed.channel, parsed.raw, parsed.calibration)
    else:
        raise ValueError("Only 1 dat file can be viewed")
        # view_multi(fpaths, parsed.channel, parsed.raw)


def dathead(args=None):
    parser = ArgumentParser(
        description=(
            "Retrieve metadata from the header of a Janelia FIBSEM .dat file, "
            "in JSON format."
        ),
    )
    parser.add_argument("file", help=".dat file to read headers for")
    parser.add_argument(
        "-p",
        "--pretty",
        action="store_true",
        help="Pretty-printing the JSON.",
    )
    parser.add_argument(
        "-k",
        "--key",
        action="append",
        help=(
            "Read specific key(s), rather than the whole header. "
            "If a single key is given, just the value is returned (as JSON); "
            "if multiple are given, a JSON object is returned with keys and values."
        ),
    )
    parsed = parser.parse_args(args)
    meta = MetadataV8.from_filepath(parsed.file)
    kwargs: tp.Dict[str, tp.Any] = {"sort_keys": True}
    if parsed.pretty:
        kwargs["indent"] = 2

    if not parsed.key:
        print(meta.to_json(**kwargs))
        return

    jso = json.loads(meta.to_json())
    if len(parsed.key) == 1:
        print(json.dumps(jso[parsed.key[0]], **kwargs))
    else:
        reduced = {k: jso[k] for k in parsed.key}
        try:
            del kwargs["sort_keys"]
        except KeyError:
            pass
        print(json.dumps(reduced, **kwargs))
