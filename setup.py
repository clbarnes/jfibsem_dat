from pathlib import Path

from setuptools import find_packages, setup

with open(Path(__file__).resolve().parent / "README.md") as f:
    readme = f.read()

setup(
    name="fibsemtools",
    url="https://github.com/clbarnes/fibsemtools",
    author="Chris L. Barnes",
    description="",
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=find_packages(include=["fibsemtools"]),
    install_requires=[
        "numpy",
        "matplotlib",
        "matplotlib_scalebar",
    ],
    python_requires=">=3.8, <4.0",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    entry_points={
        "console_scripts": [
            "datview=fibsemtools.cli:datview",
            "dathead=fibsemtools.cli:dathead",
        ]
    },
)
