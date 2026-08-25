# Copyright (c) 2026, Mars_rover_rl contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the 'mars_rover_rl' Python package.

Run from the repo root (on the NVIDIA machine, inside Isaac Lab's Python):

    python -m pip install -e source/mars_rover_rl

The `-e` means *editable*: you can change code without reinstalling.
"""

import os

import toml
from setuptools import find_packages, setup

# Absolute path to this file's directory (= the extension root).
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))

# Load metadata from extension.toml so we do not duplicate version/description.
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

# Pip dependencies installed WITH this package.
# Keep this light: Isaac Lab / torch / isaacsim are already provided by your
# Lab conda (or Isaac) environment on the NVIDIA machine. Do NOT pin isaaclab
# here the way older third-party repos sometimes do — it fights Lab 3 installs.
INSTALL_REQUIRES = [
    "psutil",  # small utility dep used in many Lab extensions; harmless default
    "toml",  # so setup can always read extension.toml (also a build need)
]

setup(
    # Pip / import distribution name.
    name="mars_rover_rl",
    # Folders under this extension root that contain Python packages.
    # Matches: source/mars_rover_rl/mars_rover/
    packages=find_packages(),
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    install_requires=INSTALL_REQUIRES,
    license="BSD-3-Clause",
    # Include non-.py files declared by package data / MANIFEST if we add any.
    include_package_data=True,
    # Isaac Lab 3.0 Beta 2 on the ASUS uses Python 3.12; Mac tests can be 3.10+.
    python_requires=">=3.10",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Isaac Sim :: 6.0.1",
    ],
    # zip_safe=False: prefer extracted files on disk (extensions + assets).
    zip_safe=False,
)