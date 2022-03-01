import logging
from argparse import ArgumentParser
from pathlib import Path

from sanic import Sanic
from sanic.response import json as sjson
from sanic.response import raw as sraw

from .multi import MultiDat

app = Sanic("JFIBSEM-dat-Neuroglancer")

CHUNKS = (256, 256, 10)  # xyz


def make_scale(dat: MultiDat):
    return {
        "chunk_sizes": [list(CHUNKS)],
        "resolution": [dat.metadata.pixel_size.tolist()] * 3,
        "size": [int(s) for s in dat.shape[1:]],
        "key": "0",
        "encoding": "raw",
        "voxel_offset": [0, 0, 0],
    }


@app.route("/info")
async def dataset_info(request):
    dat = app.config["dat"]
    info = {
        "data_type": str(dat.dtype),
        "type": "image",
        "num_channels": 1,  # todo
        "scales": [make_scale(dat)],
    }
    return sjson(info)


@app.route("/<scale:int>/<x1:int>-<x2:int>_<y1:int>-<y2:int>_<z1:int>-<z2:int>")
async def get_data(request, scale, x1, x2, y1, y2, z1, z2):
    # TODO: Enforce a data size limit
    dat: MultiDat = app.config["dat"]
    data = dat[0, x1:x2, y1:y2, z1:z2]
    # Neuroglancer expects an x,y,z array in Fortran order (e.g., z,y,x in C =)
    return sraw(data.tobytes(), content_type="application/octet-stream")


def main(args=None):
    parser = ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "-t", "--threads", type=int, help="Used for discovering metadata"
    )
    parser.add_argument("-d", "--depth", type=int, default=-1, help="Recursion depth")
    parsed = parser.parse_args(args)
    dat = MultiDat.from_directory(parsed.root.resolve(), parsed.depth, parsed.threads)
    app.debug = True
    app.config["dat"] = dat
    app.run(host="0.0.0.0", fast=True, debug=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
