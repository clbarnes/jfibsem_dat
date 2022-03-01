from itertools import chain
from pathlib import Path

from setuptools import find_packages, setup

with open(Path(__file__).resolve().parent / "README.md") as f:
    readme = f.read()

extras = {"skimage": ["scikit-image"], "vis": ["matplotlib", "matplotlib_scalebar"]}
extras["all"] = sorted(set(chain.from_iterable(extras.values())))

setup(
    name="jfibsem_dat",
    url="https://github.com/clbarnes/jfibsem_dat",
    author="Chris L. Barnes",
    description="",
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=find_packages(include=["jfibsem_dat"]),
    install_requires=[
        "numpy>=1.22",
        "scipy",
    ],
    extras_require=extras,
    python_requires=">=3.8, <4.0",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    entry_points={
        "console_scripts": [
            "datview=jfibsem_dat.cli:datview",
            "dathead=jfibsem_dat.cli:dathead",
            "dathist=jfibsem_dat.cli:dathist",
            "datcalib=jfibsem_dat.cli:datcalib",
        ]
    },
)
