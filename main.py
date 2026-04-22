#!/usr/bin/env python3
"""spateGAN-ERA5: ERA5 Precipitation Downscaling CLI.

Usage:
    uv run main.py --config config/config.yml
    uv run main.py --config config/config.yml --center-lat 50.0 --center-lon 10.0
    uv run main.py --help
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path
import yaml
import os
import sys
from datetime import datetime
from pathlib import Path

import xarray as xr
from src.spategan_era5.analysis import compare_prediction
from ecmwf.opendata import Client

# Setup paths before imports
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="spategan-era5",
        description="Downscale ERA5 precipitation to 2km/10min resolution",
    )

    parser.add_argument("-c", "--config", type=Path, default=Path("config/config.yml"),
                        help="Configuration YAML file")
    parser.add_argument("-i", "--input", type=str, help="Input ERA5 NetCDF file")
    parser.add_argument("--output-utm", type=str, help="Output directory for UTM")
    parser.add_argument("--output-latlon", type=str, help="Output directory for lat-lon")
    parser.add_argument("--center-lat", type=float, help="Center latitude (-90 to 90)")
    parser.add_argument("--center-lon", type=float, help="Center longitude (-180 to 180)")
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], help="Compute device")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--stride", type=int, choices=range(1, 9), metavar="[1-8]",
                        help="Sliding window stride (hours)"),
    parser.add_argument("--mode", type=str, choices=["downscaling", "validation"], help="Execution mode")
    parser.add_argument("--era5-path", type=str, help="Path to ERA5 NetCDF file for validation")
    # AIFS specific fields
    parser.add_argument("--download-aifs", action="store_true")
    parser.add_argument("--forecast-date", type=str, help="Format: YYYYMMDD")
    parser.add_argument("--forecast-stop", type=int, help="Forecast range stop (exclusive)")
    parser.add_argument("--forecast-step", type=int, help="Forecast range step")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress output")

    return parser


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Override config values with CLI arguments."""
    overrides = {
        ("data", "input_path"): args.input,
        ("data", "output_utm_path"): args.output_utm,
        ("data", "output_latlon_path"): args.output_latlon,
        ("domain", "center_lat"): args.center_lat,
        ("domain", "center_lon"): args.center_lon,
        ("data", "download_aifs"): True if getattr(args, "download_aifs", False) else None,
        ("data", "forecast_date"): args.forecast_date,
        ("data", "forecast_stop"): args.forecast_stop,
        ("data", "forecast_step"): args.forecast_step,
        ("time", "start_date"): args.start_date,
        ("time", "end_date"): args.end_date,
        ("processing", "seed"): args.seed,
        ("processing", "device"): args.device,
        ("inference", "stride_hours"): args.stride,
        ("mode", "type"): args.mode,
        ("validation", "era5_validation_path"): args.era5_path,
    }

    for (section, key), value in overrides.items():
        if value is not None:
            config[section][key] = value

    return config

def main() -> int:
    """Main entry point."""
    # Parse CLI
    args = create_parser().parse_args()

    # Configure logging
    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
        warnings.filterwarnings("ignore")
    elif args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load and merge config
    try:
        config = load_config(PROJECT_ROOT / args.config)
    except (FileNotFoundError, yaml.YAMLError) as e:
        logger.error("Configuration error: %s", e)
        return 1

    config = apply_cli_overrides(config, args)

    if config.get("data", {}).get("download_aifs", False):
            aifs_out_dir = Path(config.get("data", {}).get("aifs_out_dir", "./data/aifs"))

            f_date = config.get("data", {}).get("forecast_date", None) or datetime.now().strftime("%Y%m%d")
            f_stop = config.get("data", {}).get("forecast_stop", 72)
            f_step = config.get("data", {}).get("forecast_step", 6)

            f_steps = list(range(0, f_stop + f_step, f_step))

            try:
                from src.spategan_era5.download import prepare_ecmwf_data
                processed_file = prepare_ecmwf_data(aifs_out_dir, f_date, f_steps)
                config["data"]["input_path"] = str(processed_file)
            except Exception as e:
                logger.error(f"Failed to prepare AIFS data: {e}")
                return 1

    logger.info("Starting downscaling: center=(%.2f°N, %.2f°E), device=%s",
                config["domain"]["center_lat"],
                config["domain"]["center_lon"],
                config["processing"]["device"])

    # Execute selected mode
    mode = config.get("mode", {}).get("type", "downscaling")

    # Run pipeline
    try:
        if mode == "downscaling":
                    logger.info("Starting downscaling: center=(%.2f°N, %.2f°E), device=%s",
                                config["domain"]["center_lat"],
                                config["domain"]["center_lon"],
                                config["processing"]["device"])

                    from src.spategan_era5.pipeline import run_downscaling_pipeline
                    run_downscaling_pipeline(config, PROJECT_ROOT)

        elif mode == "validation":
            logger.info("Starting validation analysis...")

            from src.spategan_era5.analysis import compare_prediction
            compare_prediction(
                era5_file=Path(config["validation"]["era5_path"]),
                spategan_file=Path(config["validation"]["spategan_file_path"])
            )

        else:
            logger.error("Unknown mode: %s", mode)
            return 1

        return 0
    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        return 1
    except ValueError as e:
        logger.error("Validation error: %s", e)
        return 1
    except Exception as e:
        logger.exception("Unexpected error: %s", e)
        return 1

if __name__ == "__main__":
    sys.exit(main())
