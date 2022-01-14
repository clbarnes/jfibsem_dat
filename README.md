# jfibsem_dat

Python implementation of the FIBSEM .dat file format developed at Janelia Research Campus, and associated tools.
Written for CPython 3.8+.

Based on a reference MATLAB implementation formerly at https://github.com/david-hoffman/FIB-SEM-Aligner (now taken down) and vendorised in `./reference`.
The repo was unmaintained when these copies were made, and so may contain errors (indeed, there are some known inconsistencies).

There is a FIJI implementation here: https://github.com/fiji/IO/blob/master/src/main/java/sc/fiji/io/FIBSEM_Reader.java

## Installation

From github:

```sh
pip install git+https://github.com/clbarnes/jfibsem_dat.git
```

## Utilities

### `datview`

```_datview
usage: datview [-h] [-c CHANNEL] [-C CALIBRATION] [-r] file

View a Janelia FIBSEM .dat file. The data can be scaled using the file's
metadata, viewed --raw, or scaled using a --calibration CSV. Uses matplotlib.

positional arguments:
  file                  .dat file to view

optional arguments:
  -h, --help            show this help message and exit
  -c CHANNEL, --channel CHANNEL
                        Which channel to view (default first). Not all
                        channels exist.
  -C CALIBRATION, --calibration CALIBRATION
                        CSV file calibrating raw to scaled values
  -r, --raw             Show raw data rather than scaled electron counts.
```

Uses matplotlib.

### `dathead`

```_dathead
usage: dathead [-h] [-p] [-k KEY] file

Retrieve metadata from the header of a Janelia FIBSEM .dat file, in JSON
format.

positional arguments:
  file               .dat file to read headers for

optional arguments:
  -h, --help         show this help message and exit
  -p, --pretty       Pretty-printing the JSON.
  -k KEY, --key KEY  Read specific key(s), rather than the whole header. If a
                     single key is given, just the value is returned (as
                     JSON); if multiple are given, a JSON object is returned
                     with keys and values.
```

## Format notes

- Field `sw_date` encodes a date as a string of form `DD/MM/YYYY`
