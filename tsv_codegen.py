#!/usr/bin/env python3
import csv
import subprocess as sp
from pathlib import Path

here = Path(__file__).resolve().parent
spec_dir = here / "jfibsem_dat" / "specs"


def map_dtype_shape(dtype_str: str, shape_str: str) -> str:
    if "u" in dtype_str or "i" in dtype_str:
        dtype = "int"
    elif "f" in dtype_str:
        dtype = "float"
    elif "S" in dtype_str:
        dtype = "bytes"
    else:
        raise ValueError("Unknown dtype " + dtype_str)

    if shape_str == "0":
        return dtype

    prev = dtype
    for item in reversed(shape_str.split(",")):
        try:
            length = int(item)
        except ValueError:
            prev = f"list[{prev}]"
        else:
            rep_items = ", ".join([prev] * length)
            prev = f"tuple[{rep_items}]"
    return prev


def make_init_line(d: dict[str, str]) -> tuple[str, str]:
    name = d["name"]
    dtype = map_dtype_shape(d["dtype"], d["shape"])
    return name, f"self.{name}: {dtype} = {name}"


HEADER = """
# This file was autogenerated and should not be edited
from .base import MetadataBase
""".strip()

FOOTER_TEMPLATE = """
VERSION_REGISTRY = {{{registry}}}
""".strip()


TEMPLATE = """
class MetadataV{version}(MetadataBase):
    def __init__(self, {names}, **kwargs):
        super().__init__(**kwargs)

{inits}
""".strip()


def parse_tsv(version) -> str:
    names = []
    inits = []
    with open(spec_dir / f"v{version}.tsv") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            name, init = make_init_line(row)
            names.append(name)
            inits.append(init)

    names_str = ", ".join(names)
    inits_str = "\n".join(" " * 4 * 2 + init for init in inits)

    return TEMPLATE.format(version=version, names=names_str, inits=inits_str)


def path_to_version(fpath: Path) -> int:
    return int(fpath.stem[1:])


def parse_tsvs() -> str:
    blocks = [HEADER]
    versions = []
    for fpath in sorted(spec_dir.glob("v*.tsv")):
        version = path_to_version(fpath)
        if version == 0:
            continue
        s = parse_tsv(version)
        blocks.append(s)
        versions.append(version)

    registry = ", ".join(f"{{{v}: MetadataV{v}}}" for v in versions)
    blocks.append(FOOTER_TEMPLATE.format(registry=registry))
    return ("\n" * 3).join(blocks) + "\n"


def main():
    s = parse_tsvs()
    fpath = here / "jfibsem_dat" / "metadata" / "versions.py"
    fpath.write_text(s)
    sp.run(["black", str(fpath)], check=True)


if __name__ == "__main__":
    main()