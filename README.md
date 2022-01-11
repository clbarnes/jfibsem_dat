# fibsemtools

Python implementation of Janelia Research Campus FIBSEM file format, and associated tools.
Written for CPython 3.8+.

Based on a reference MATLAB implementation formerly at https://github.com/david-hoffman/FIB-SEM-Aligner (now taken down) and vendorised in `./reference`.
The repo was unmaintained when these copies were made, and so may contain errors (indeed, there are some known inconsistencies).

There is a FIJI implementation here: https://github.com/fiji/IO/blob/master/src/main/java/sc/fiji/io/FIBSEM_Reader.java


## Installation

From github:

```sh
pip install git+https://github.com/clbarnes/fibsemtools.git
```

## Utilities

### `datview`

Use matplotlib to view a single channel of a single dat file.

### `dathead`

View a dat file's metadata as JSON.
