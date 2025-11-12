from enum import Enum, IntEnum
from itertools import product
from pathlib import Path
from typing import Literal, Optional

from astropy import units as u
from astropy.io import fits
from pydantic import BaseModel, field_validator, model_validator


class Site(Enum):
    south = "south"
    north = "north"


class Configuration(Enum):
    alpha = "alpha"
    omega = "omega"


class Azimuth(Enum):
    south = "south"
    north = "north"
    average = "average"


class Zenith(IntEnum):
    z20 = 20
    z40 = 40
    z60 = 60


class Duration(IntEnum):
    t1800 = 1800
    t18000 = 18000
    t180000 = 180000


class IRF(BaseModel):
    """
    Represents the Instrument Response Function (IRF) for the CTA (Cherenkov Telescope Array).

    Attributes:
        base_directory (Optional[Path]): The base directory for the IRF files.
        filepath (Path): The path to the IRF file.
        configuration (Configuration): The configuration of the IRF.
        site (Site): The site where the IRF is located.
        duration (int): The duration of the IRF in seconds.
        zenith (Optional[Zenith]): The zenith angle of the IRF.
        azimuth (Azimuth): The azimuth angle of the IRF.
        has_nsb (bool): Indicates whether the IRF includes Night Sky Background (NSB) data.
        n_sst (Optional[int]): The number of Single-Size Telescopes (SSTs) in the IRF.
        n_mst (Optional[int]): The number of Medium-Size Telescopes (MSTs) in the IRF.
        n_lst (Optional[int]): The number of Large-Size Telescopes (LSTs) in the IRF.
        version (Optional[str]): The version of the IRF.

    Methods:
        validate_base_directory(cls, base_directory): Validates the base directory path.
        validate_filepath(cls, filepath, values): Validates the filepath and resolves it relative to the base directory if provided.
        __repr__(self): Returns a string representation of the IRF.
        __fspath__(self): Returns the filepath as a string.

    """

    base_directory: Optional[Path] = None
    filepath: Path
    configuration: Configuration
    site: Site
    duration: int
    zenith: Optional[Zenith] = None
    azimuth: Azimuth
    has_nsb: bool = False
    n_sst: Optional[int] = None
    n_mst: Optional[int] = None
    n_lst: Optional[int] = None
    energy_min: Optional[float] = None
    energy_max: Optional[float] = None
    version: Optional[str] = None

    @field_validator("base_directory", mode="before")
    @classmethod
    def validate_base_directory(cls, base_directory):
        """
        Validates the base directory path.

        Args:
            base_directory (str): The path to the base directory.

        Returns:
            str: The validated base directory path.

        Raises:
            ValueError: If the base directory does not exist or is not a directory.
        """
        if base_directory:
            base_directory = Path(base_directory)
            if not base_directory.exists():
                raise ValueError(f"Base directory {base_directory} does not exist")
            if not base_directory.is_dir():
                raise ValueError(f"Base directory {base_directory} is not a directory")
            if not base_directory.is_absolute():
                base_directory = base_directory.resolve()
        return base_directory

    @model_validator(mode="before")
    @classmethod
    def validate_filepath(cls, data):
        """
        Validates the given filepath by checking if it exists.

        Args:
            data: The model data dictionary.

        Returns:
            dict: The validated data dictionary.

        Raises:
            ValueError: If the filepath does not exist.
        """
        if isinstance(data, dict):
            base_directory = data.get("base_directory")
            filepath = data.get("filepath")
            
            if filepath:
                filepath = Path(filepath)
                
                if base_directory:
                    base_directory = Path(base_directory).absolute()
                    filepath = Path(base_directory).absolute() / filepath
                if not filepath.exists():
                    raise ValueError(f"File {filepath} does not exist")
                data["filepath"] = filepath
        return data

    def __repr__(self):
        title = "CTA IRF" + (f" [{self.version}]" if self.version else "")
        filepath = f"    filepath: {self.filepath}"
        config = f"    config: {self.configuration} - {self.n_sst} SSTs // {self.n_mst} MSTs // {self.n_lst} LSTs"
        site = f"    site: {self.site} {'(with NSB)' if self.has_nsb else ''}"
        zenith = f"    zenith: {self.zenith}º"
        duration = f"    duration: {self.duration}s"
        azimuth = f"    azimuth: {self.azimuth}"

        return "\n".join([title, filepath, config, site, zenith, duration, azimuth])

    def __fspath__(self):
        return str(self.filepath)

    def get_energy_limits(self):
        with fits.open(self.filepath) as hdul:
            self.energy_min = min(hdul[1].data["ENERG_LO"][0]) * u.TeV
            self.energy_max = max(hdul[1].data["ENERG_HI"][0]) * u.TeV

    @property
    def energy_limits(self):
        if self.energy_min is None or self.energy_max is None:
            self.get_energy_limits()
        return self.energy_min, self.energy_max


class IRFHouse(BaseModel):
    base_directory: Path | str
    check_irfs: bool = True

    @field_validator("base_directory", mode="before")
    @classmethod
    def validate_base_directory(cls, base_directory):
        base_directory = Path(base_directory)
        if not base_directory.exists():
            raise ValueError(f"Base directory {base_directory} does not exist")
        if not base_directory.is_dir():
            raise ValueError(f"Base directory {base_directory} is not a directory")
        if not base_directory.is_absolute():
            base_directory = base_directory.resolve()
        return base_directory

    @model_validator(mode="after")
    def validate_check_irfs(self):
        if self.check_irfs:
            self.check_all_paths()
        return self

    # ALPHA SOUTH           =         14 MST  37 SST
    # ALPHA SOUTH MODIFIED  =  4 LST  14 MST  40 SST
    # ALPHA NORTH           =  4 LST   9 MST
    # OMEGA SOUTH           =  4 LST  25 MST  70 SST
    # OMEGA NORTH           =  4 LST  15 MST

    def get_alpha_v0p1(
        self,
        site: Site,
        zenith: Zenith,
        duration: Duration,
        azimuth: Azimuth = "average",
    ):
        site_string = site.value.capitalize()
        azimuth_string = f"{azimuth.value.capitalize()}Az"
        if site.value == "north":
            n_lst = 4
            n_mst = 9
            n_sst = 0
            telescope_string = "4LSTs09MSTs"
        elif site.value == "south":
            n_lst = 0
            n_mst = 14
            n_sst = 37
            telescope_string = "14MSTs37SSTs"
        else:
            raise ValueError(f"Invalid site {site}")

        return IRF(
            base_directory=self.base_directory,
            filepath=f"prod5-v0.1/fits/CTA-Performance-prod5-v0.1-{site_string}-{zenith}deg.FITS/Prod5-{site_string}-{zenith}deg-{azimuth_string}-{telescope_string}.{duration}s-v0.1.fits.gz",
            configuration="alpha",
            site=site,
            zenith=zenith,
            duration=duration,
            azimuth=azimuth,
            n_sst=n_sst,
            n_mst=n_mst,
            n_lst=n_lst,
            version="prod5-v0.1",
        )

    def get_v0p2(
        self,
        site: Literal["north", "south"],
        configuration: Literal["alpha", "omega"],
        zenith: int,
        duration: Literal[1800, 18000, 180000],
        azimuth: Literal["north", "south", "average"] = "average",
        modified: bool = False,
        nsb: bool = False,
    ):
        site_string = site.capitalize()
        azimuth_string = f"{azimuth.capitalize()}Az"

        if site == "north" and modified:
            raise ValueError("No modified configuration for North site")
        elif site == "north" and configuration == "alpha":
            n_lst = 4
            n_mst = 9
            n_sst = 0
            telescope_string = "4LSTs09MSTs"
        elif site == "north" and configuration == "omega":
            n_lst = 4
            n_mst = 15
            n_sst = 0
            telescope_string = "4LSTs15MSTs"
        elif site == "south" and configuration == "alpha" and not modified:
            n_lst = 0
            n_mst = 14
            n_sst = 37
            telescope_string = "14MSTs37SSTs"
        elif site == "south" and configuration == "alpha" and modified:
            n_lst = 4
            n_mst = 14
            n_sst = 40
            telescope_string = "4LSTs14MSTs40SSTs"
        elif site == "south" and configuration == "omega":
            n_lst = 4
            n_mst = 25
            n_sst = 70
            telescope_string = "4LSTs25MSTs70SSTs"
        else:
            raise ValueError(f"Invalid configuration {configuration} for site {site}")

        return IRF(
            base_directory=self.base_directory,
            filepath=f"prod5-v0.2/fits/Prod5-{site_string}{'-NSB5x' if nsb else ''}-{zenith}deg-{azimuth_string}-{telescope_string}.{duration}s-v0.2.fits.gz",
            configuration="alpha",
            site=site,
            zenith=zenith,
            duration=duration,
            azimuth=azimuth,
            has_nsb=nsb,
            n_sst=n_sst,
            n_mst=n_mst,
            n_lst=n_lst,
            version="prod5-v0.2",
        )

    def get_prod3b_v2(
        self,
        site: Literal["north", "south"],
        zenith: Literal[20, 40, 60],
        duration: Literal[1800, 18000, 180000],
        azimuth: Literal["north", "south", "average"],
    ):
        site_string = site.capitalize()

        if azimuth == "average":
            azimuth_string = ""
        else:
            azimuth_string = "_" + azimuth.capitalize()[0]

        if duration == 1800:
            duration_string = "0.5"
        elif duration == 18000:
            duration_string = "5"
        elif duration == 180000:
            duration_string = "50"
        else:
            raise ValueError(f"Invalid duration {duration}")

        if site == "north":
            n_lst = 4
            n_mst = 15
            n_sst = 0
        elif site == "south":
            n_lst = 4
            n_mst = 25
            n_sst = 70
        else:
            raise ValueError(f"Invalid site {site}")

        return IRF(
            base_directory=self.base_directory,
            filepath=f"prod3b-v2/fits/caldb/data/cta/prod3b-v2/bcf/{site_string}_z{zenith}{azimuth_string}_{duration_string}h/irf_file.fits",
            configuration="alpha",
            site=site,
            zenith=zenith,
            duration=duration,
            azimuth=azimuth,
            n_sst=n_sst,
            n_mst=n_mst,
            n_lst=n_lst,
            version="prod3b-v2",
        )

    def get_irf(
        self,
        site: Literal["north", "south"],
        configuration: Literal["alpha", "omega"],
        zenith: int,
        duration: Literal[1800, 18000, 180000],
        azimuth: Literal["north", "south", "average"],
        version: Literal["prod5-v0.1", "prod5-v0.1", "prod3b-v2"],
        modified: bool = False,
        nsb: bool = False,
    ):
        if version == "prod5-v0.1":
            if configuration == "omega":
                raise ValueError("No omega configuration for prod5-v0.1")

            return self.get_alpha_v0p1(
                site=site, zenith=zenith, duration=duration, azimuth=azimuth
            )

        elif version == "prod5-v0.2":
            return self.get_v0p2(
                site=site,
                configuration=configuration,
                zenith=zenith,
                duration=duration,
                azimuth=azimuth,
                modified=modified,
                nsb=nsb,
            )

        elif version == "prod3b-v2":
            if configuration == "alpha":
                raise ValueError("No alpha configuration for prod3b-v2")

            return self.get_prod3b_v2(
                site=site, zenith=zenith, duration=duration, azimuth=azimuth
            )
        else:
            raise ValueError(f"Invalid version {version}")

    def check_all_paths(self):
        missing_irf_count = 0

        sites = ["north", "south"]
        configurations = ["alpha", "omega"]
        zeniths = [20, 40, 60]
        durations = [1800, 18000, 180000]
        azimuths = ["north", "south", "average"]
        versions = ["prod5-v0.1", "prod5-v0.2", "prod3b-v2"]
        modifieds = [False, True]

        for (
            site,
            configuration,
            zenith,
            duration,
            azimuth,
            version,
            modified,
        ) in product(
            sites, configurations, zeniths, durations, azimuths, versions, modifieds
        ):
            if (
                (version == "prod5-v0.1" and configuration == "omega")
                or (version == "prod3b-v2" and configuration == "alpha")
                or (modified and site == "north")
            ):
                continue
            try:
                self.get_irf(
                    site=site,
                    configuration=configuration,
                    zenith=zenith,
                    duration=duration,
                    azimuth=azimuth,
                    version=version,
                    modified=modified,
                )
            except ValueError as e:
                print(e)
                print(
                    f"Failed to find IRF for site={site}, configuration={configuration}, zenith={zenith}, duration={duration}, azimuth={azimuth}, version={version}"
                )
                missing_irf_count += 1

        if missing_irf_count > 0:
            print(
                f"⚠️ Missing {missing_irf_count} IRF{'' if missing_irf_count == 1 else 's'}"
            )
        else:
            print("✅ All IRFs found")
