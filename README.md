# jfibsem_dat

Python implementation of the FIBSEM .dat file format developed at Janelia Research Campus, and associated tools.
Written for CPython 3.8+.

Based on a reference MATLAB implementation formerly at https://github.com/david-hoffman/FIB-SEM-Aligner (now taken down) and vendorised in `./reference`.
The repo was unmaintained when these copies were made, and so may contain errors (indeed, there are some known inconsistencies).

There is a FIJI implementation here: https://github.com/fiji/IO/blob/master/src/main/java/sc/fiji/io/FIBSEM_Reader.java

This project currently supports v8 of the image specification.

## Installation

Basic installation from github:

```sh
pip install 'git+https://github.com/clbarnes/jfibsem_dat.git[all]'
```

This package contains a number of extras:

- `vis` contains dependencies used for viewing images
- `skimage` contains dependencies for downsampling images
- `all` contains all of the above

## Utilities

### `datview`

```_datview
usage: datview [-h] [-c CHANNEL] [-C CALIBRATION] [-d DOWNSAMPLE] [-r] file

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
  -d DOWNSAMPLE, --downsample DOWNSAMPLE
                        Downsample the image; good for quicker viewing.
  -r, --raw             Show raw data rather than scaled electron counts.
```

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

### `dathist`

```_dathist
usage: dathist [-h] [-c CHANNEL] [-C CALIBRATION] [-d DOWNSAMPLE] [-r] file

View the histogram of pixel values for one channel of Janelia FIBSEM .dat file

positional arguments:
  file                  .dat file to view

optional arguments:
  -h, --help            show this help message and exit
  -c CHANNEL, --channel CHANNEL
                        Which channel to view (default first). Not all
                        channels exist.
  -C CALIBRATION, --calibration CALIBRATION
                        CSV file calibrating raw to scaled values
  -d DOWNSAMPLE, --downsample DOWNSAMPLE
                        Downsample the image; good for quicker viewing.
  -r, --raw             Show raw data rather than scaled electron counts.
```

### `datcalib`

```_datcalib
usage: datcalib [-h] [-c CHANNEL] [-C CALIBRATION] [-d DOWNSAMPLE] [-r]
                [-s SAMPLES] [-V]
                file operation [operation ...]

Produce a calibration CSV for some simple exposure corrections

positional arguments:
  file                  .dat file to view
  operation             Calibration functions to apply. Multiple functions can
                        be given, and will be applied in order. Given in the
                        form 'fn_name,kwarg1_name=kwarg1_value,...'. Values
                        given in JSON format (e.g. 'null' instead of 'None'),
                        although uncontained strings do not need quoting.
                        Functions are documented in scikit-image's exposure
                        package. Accepted functions are: adjust_gamma,
                        adjust_log, adjust_sigmoid, equalize_hist,
                        rescale_intensity

optional arguments:
  -h, --help            show this help message and exit
  -c CHANNEL, --channel CHANNEL
                        Which channel to view (default first). Not all
                        channels exist.
  -C CALIBRATION, --calibration CALIBRATION
                        CSV file calibrating raw to scaled values
  -d DOWNSAMPLE, --downsample DOWNSAMPLE
                        Downsample the image; good for quicker viewing.
  -r, --raw             Show raw data rather than scaled electron counts.
  -s SAMPLES, --samples SAMPLES
                        Maximum number of samples in the calibration CSV
  -V, --view            Whether to show the calibrated result
```

## Format notes

- Field `sw_date` encodes a date as a string of form `DD/MM/YYYY`
- There are a number of unexplained constants and unanswered questions about the format/ its reference MATLAB implementation: grains of salt are a dependency of using this project.

## Contributing

Contributions are welcome!

This project uses `black` and `isort` for formatting (run `make fmt`), and `pre-commit` for general code quality checks.

Use `make fmt` for formatting and `make lint` for spot checks, and `pre-commit run --all` to run all hooks.

If you modify any part of the CLI, use `make readme` to update the help text in the README.
