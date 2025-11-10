import pathlib

from setuptools import setup

# The directory which contains this file
HERE = pathlib.Path(__file__).parent
PKG_PATH = HERE.resolve().as_posix()

# The readme file
README = (HERE / "README.md").read_text()

setup(
    name="gravitational_wave_toy",
    description="A tool to simulate gravitational wave follow-up observations with gamma-ray observatories.",
    version="2.0",
    packages=["gravitational_wave_toy"],
    url="https://github.com/astrojarred/gravitational_wave_toy/",
    author="Jarred Green (MPP), Barbara Patricelli (INAF)",
    long_description=README,
    long_description_content_type="text/markdown",
    author_email="jarred.green@inaf.it",
)
