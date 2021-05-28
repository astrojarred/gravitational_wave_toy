from setuptools import setup

import pathlib

# The directory which contains this file
HERE = pathlib.Path(__file__).parent
PKG_PATH = HERE.resolve().as_posix()

# The readme file
README = (HERE / "README.md").read_text()

setup(
    name="gravitational_wave_toy",
    description="A tool to .",
    version="1.1",
    packages=["gravitational_wave_toy"],
    url="https://github.com/astrojarred/gravitational_wave_toy/",
    author="Jarred Green (INAF-OAR/ASI), Barbara Patricelli (INAF)",
    long_description=README,
    long_description_content_type="text/markdown",
    author_email="jarred.green@inaf.it",
    entry_points={
        "console_scripts": [
            "gw-toy=gravitational_wave_toy.main:run",
            "gw-plotter=gravitational_wave_toy.main:plot",
        ]
    },
)
