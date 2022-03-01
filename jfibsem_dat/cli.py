import json
import logging
import typing as tp
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from .calibrate import calibrate
from .read import MetadataV8, RawFibsemData

logger = logging.getLogger(__name__)
CMAP = "gray_r"


def get_array(
    fpath,
    channel: tp.Optional[int],
    raw: bool = False,
    calibration_path: tp.Optional[Path] = None,
    downsample: tp.Optional[int] = None,
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

    if downsample is not None:
        from skimage.transform import downscale_local_mean

        logger.info("Downsampling by factor %s", downsample)
        raw_vals = dat.data[idx].copy()
        shrunk = downscale_local_mean(raw_vals, downsample).astype(raw_vals.dtype)
        new_channels = [np.empty_like(shrunk)] * meta.n_channels
        new_channels[idx] = shrunk
        dat.data = np.array(new_channels)
        meta.pixel_size *= downsample

    if raw:
        logger.info("Using raw data")
        arr = dat.data[idx]
    else:
        kwargs = {}
        if calibration_path is not None:
            kwargs["calibration"] = [
                np.genfromtxt(
                    calibration_path, "float32", delimiter=",", autostrip=True
                ).T
            ]
        logger.info("Scaling pixel values")
        channel_data = dat.scale([channel], **kwargs)[0]
        arr = channel_data.electron_counts

    dat.close()
    return arr, meta, channel


def view_array(array, title=None, pixel_size=None):
    from matplotlib import pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib_scalebar.scalebar import ScaleBar

    logger.info("Displaying")
    fig: Figure = plt.figure()
    ax: Axes = fig.add_subplot()
    if title:
        ax.set_title(title)
    pos = ax.imshow(array, cmap=CMAP)
    if pixel_size:
        sc = ScaleBar(pixel_size, "nm")
        logger.warning("Resolution may be incorrect by factor of 2.54e7")
        ax.add_artist(sc)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    fig.colorbar(pos, ax=ax)
    plt.show()


def view_single(
    fpath,
    channel,
    raw=False,
    calibration_path=None,
    downsample: tp.Optional[int] = None,
):
    arr, meta, channel = get_array(fpath, channel, raw, calibration_path, downsample)
    name = meta.detector_names[channel]

    view_array(arr, f"{fpath}\n{name}", meta.pixel_size)


def expand_paths(paths):
    if isinstance(paths, str) or not hasattr(paths, "__iter__"):
        yield paths
        return

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


def add_array_args(
    parser: ArgumentParser, channel=True, calibration=True, downsample=True, raw=True
):
    if channel:
        parser.add_argument(
            "-c",
            "--channel",
            type=int,
            help="Which channel to view (default first). Not all channels exist.",
        )
    if calibration:
        parser.add_argument(
            "-C",
            "--calibration",
            type=Path,
            help="CSV file calibrating raw to scaled values",
        )
    if downsample:
        parser.add_argument(
            "-d",
            "--downsample",
            type=int,
            help="Downsample the image; good for quicker viewing.",
        )
    if raw:
        parser.add_argument(
            "-r",
            "--raw",
            action="store_true",
            help="Show raw data rather than scaled electron counts.",
        )
    return parser


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
    add_array_args(parser)
    parsed = parser.parse_args(args)

    fpaths = list(expand_paths(parsed.file))
    if len(fpaths) == 0:
        logger.info("No .dat files given")
    elif len(fpaths) == 1:
        view_single(
            fpaths[0], parsed.channel, parsed.raw, parsed.calibration, parsed.downsample
        )
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


def dathist(args=None):
    from matplotlib import pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    parser = ArgumentParser(
        description=(
            "View the histogram of pixel values "
            "for one channel of Janelia FIBSEM .dat file"
        ),
    )
    parser.add_argument("file", type=Path, help=".dat file to view")
    # parser.add_argument("file", nargs="*", type=Path, help=".dat file(s) to view")
    add_array_args(parser)
    parsed = parser.parse_args(args)
    arr, _, _ = get_array(
        parsed.file, parsed.channel, parsed.raw, parsed.calibration, parsed.downsample
    )

    logger.info("Binning and displaying")
    fig: Figure = plt.figure()
    ax: Axes = fig.add_subplot()
    ax.hist(arr.ravel(), bins=256)
    ax.set_title(f"Histogram for {parsed.file}")
    ax.set_xlabel("intensity")
    ax.set_ylabel("frequency")

    plt.show()


def parse_value(s):
    if s == "" or s == "None":
        return None
    if s == "True":
        return True
    if s == "False":
        return False

    try:
        out = float(s)
    except TypeError:
        return s
    as_int = int(out)
    if as_int == out:
        return as_int
    return out


class OpParser:
    def __init__(self) -> None:
        from skimage import exposure as exp

        self.funcs = {
            fn.__name__: fn
            for fn in [
                exp.adjust_gamma,
                exp.adjust_log,
                exp.adjust_sigmoid,
                exp.equalize_hist,
                exp.rescale_intensity,
            ]
        }

    def parse(self, s: str):
        s = "".join(s.split())
        name, *args = s.split(",")
        kwargs = {}
        for arg in args:
            k, v = arg.split("=")
            try:
                kwargs[k] = json.loads(v)
            except json.JSONDecodeError:
                kwargs[k] = v

        fn = self.funcs[name]

        def func(img):
            logger.info("Applying %s", s)
            return fn(img, **kwargs)

        return func


def parse_version(s):
    return tuple(int(n) for n in s.split("."))


def interp_method_name():
    """This changed in numpy 1.22"""
    if parse_version(np.__version__) < (1, 22):
        return "interpolation"
    return "method"


def datcalib(args=None):
    from skimage import exposure as exp

    parser = ArgumentParser(
        description=("Produce a calibration CSV for some simple exposure corrections"),
    )
    parser.add_argument("file", type=Path, help=".dat file to view")
    # parser.add_argument("file", nargs="*", type=Path, help=".dat file(s) to view")
    add_array_args(parser, channel=True, downsample=True)
    op_parser = OpParser()
    parser.add_argument(
        "operation",
        type=op_parser.parse,
        nargs="+",
        help=(
            "Calibration functions to apply. "
            "Multiple functions can be given, and will be applied in order. "
            "Given in the form 'fn_name,kwarg1_name=kwarg1_value,...'. "
            "Values given in JSON format (e.g. 'null' instead of 'None'), "
            "although uncontained strings do not need quoting. "
            "Functions are documented in scikit-image's exposure package. "
            f"Accepted functions are: {', '.join(op_parser.funcs)}"
        ),
    )
    parser.add_argument(
        "-s",
        "--samples",
        type=int,
        default=100,
        help="Maximum number of samples in the calibration CSV",
    )
    parser.add_argument(
        "-V",
        "--view",
        action="store_true",
        help="Whether to show the calibrated result",
    )
    parsed = parser.parse_args(args)

    arr, meta, _ = get_array(
        parsed.file, parsed.channel, True, downsample=parsed.downsample
    )
    out = exp.rescale_intensity(arr.astype("float32"), out_range=(0, 1))
    for op in parsed.operation:
        out = op(out)

    xs, ys = calibrate(arr, out, parsed.samples)
    u16_max = np.iinfo("uint16").max
    ys = (ys * u16_max).astype("uint16")
    for x, y in zip(xs, ys):
        print(f"{x},{y}")

    if parsed.view:
        pixel_size = meta.pixel_size
        if parsed.downsample:
            pixel_size *= parsed.downsample
        view_array(out, str(parsed.file), meta.pixel_size)
