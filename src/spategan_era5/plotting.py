import logging
import math
from dataclasses import dataclass
from pathlib import Path

from matplotlib import animation
import pandas as pd
import xarray as xr

from src.spategan_era5.utils import _combine_datasets, _combine_datasets, generate_output_filename

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)

@dataclass
class Plots:
    predictions_utm: xr.Dataset
    ds_utm_28: xr.Dataset
    project_root: Path
    config: dict

    def __post_init__(self):
        self.plots_dir = (
                self.project_root / self.config.get("data", {}).get("plots_path", "plots")
        )
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.plot_name = generate_output_filename(self.predictions_utm, self.config, projection="utm")

    def create_precipitation_sums_plot(self) -> None:
        """Create a simple side-by-side precipitation sums plot.

        Left: (ds_utm_28.cp + ds_utm_28.lsp).sum(dim='time')
        Right: predictions_utm.precipitation.sum(dim='time')

        Saves PNG to `project_root / plots` (or config['data']['plots_path']).
        """
        plotting_cfg = self.config.get("plotting", {})
        if not plotting_cfg.get("precipitation_sums", False):
            return

        try:
            left = None
            if ("cp" in self.ds_utm_28.data_vars) and ("lsp" in self.ds_utm_28.data_vars):
                left = (self.ds_utm_28["cp"] + self.ds_utm_28["lsp"]).sum(dim="time")
                left = left[8:-8, 8:-8]
            else:
                logger.warning("ds_utm_28 missing 'cp' and/or 'lsp' variables; left plot will be empty")

            right = None
            if "precipitation" in self.predictions_utm.data_vars:
                right = (self.predictions_utm["precipitation"]/6).sum(dim="time") # /6 since 6 time steps per hour
            else:
                logger.warning("predictions_utm missing 'precipitation' variable; right plot will be empty")

            if (left is None) and (right is None):
                logger.warning("No data available for precipitation sums plotting; skipping")
                return

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            if left is not None:
                left.plot(ax=axes[0], cmap="turbo")
                axes[0].set_title("Input (target domain) precipitation sum (cp + lsp) - mm")
            else:
                axes[0].axis("off")

            if right is not None:
                right.plot(ax=axes[1], cmap="turbo")
                axes[1].set_title("Predicted precipitation sum - mm")
            else:
                axes[1].axis("off")

            plt.tight_layout()

            plot_name = "precip_sums_" + self.plot_name.replace(".nc", ".png")
            out_path = self.plots_dir / plot_name
            fig.savefig(out_path, dpi=150)
            plt.close(fig)

            logger.info("Saved precipitation sums plot to %s", out_path)

        except Exception as e:
            logger.exception("Error creating precipitation sums plot: %s", e)


    def _plot_layout(self, num_plots: int) -> int:
        return math.ceil(math.sqrt(num_plots))


    def plot_forecast(self) -> None:
        """Create forecast plots driven by 'forecast_intervals' setting.
        Saves PNG to `project_root / plots` (or config['data']['plots_path']) for each set of intervals
        """
        plotting_cfg = self.config.get("plotting", {})
        if not plotting_cfg.get("forecast_plots", False):
            return

        try:
            interval_dict = plotting_cfg.get("forecast_intervals", {})
            if "precipitation" in self.predictions_utm.data_vars:
                start = 0
                max_h = self.predictions_utm.time.size // 6
                for day, interval in sorted(interval_dict.items()):
                    end = day * 24
                    sums = []
                    labels = []

                    for h in range(start, end, interval):
                        chunk = self.predictions_utm.isel(time=slice(h * 6 , (h + interval) * 6))
                        if chunk.time.size > 0:
                            start_time = pd.Timestamp(chunk.time.values[0]).strftime("%d.%m.%y %H:%M")
                            sums.append((chunk['precipitation'] / 6).sum(dim="time"))
                            labels.append(f"+{h}h {interval}h sum \n {start_time}")

                    if len(sums) > 0:
                        da = xr.concat(sums, dim="time").assign_coords(time=labels)
                        vmax = float(da.max())
                        fg = da.plot(
                            col="time",
                            col_wrap=self._plot_layout(len(sums)),
                            vmin=0,
                            vmax=vmax if vmax > 0 else 1,
                            cmap="turbo"
                        )
                        fg.set_titles(template="{value}")
                        plt.savefig(self.plots_dir / f"{self.plot_name}_day_{day}.png")
                        plt.close()

                    start = end

                    if start >= max_h:
                        break

            else:
                logger.warning("predictions_utm missing 'precipitation' variable; right plot will be empty")

        except Exception as e:
            logger.exception("Error creating forecast plots: %s", e)

    def create_precipitation_animation(self, num_frames=24, save_filename=None):
            combined_ds = _combine_datasets(self.ds_utm_28, self.predictions_utm)

            if len(combined_ds.time) < num_frames:
                num_frames = len(combined_ds.time)
            ds_subset = combined_ds.isel(time=slice(0, num_frames)).copy()

            ds_subset["era5_tp"] = ds_subset["era5_tp"] * 1000

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            era5_max = ds_subset["era5_tp"].max().item()
            pred_max = ds_subset["pred_precipitation"].max().item()
            global_max = max(era5_max, pred_max)

            ds_0 = ds_subset.isel(time=0)

            # Plotting
            era5_plot = ds_0["era5_tp"].plot(
                ax=axes[0], cmap="YlGnBu", vmin=0, vmax=global_max,
                cbar_kwargs={"label": "Precipitation (mm)"}, add_colorbar=True
            )
            axes[0].set_title("Ground Truth (ERA5 Interpolated)")

            pred_plot = ds_0["pred_precipitation"].plot(
                ax=axes[1], cmap="YlGnBu", vmin=0, vmax=global_max,
                cbar_kwargs={"label": "Precipitation (mm)"}, add_colorbar=True
            )
            axes[1].set_title("Prediction (SpateGAN)")

            time_str = ds_0.time.dt.strftime("%Y-%m-%d %H:%M").item()
            title = fig.suptitle(f"Precipitation Comparison at {time_str}")

            def update(frame_index):
                ds_frame = ds_subset.isel(time=frame_index)

                era5_plot.set_array(ds_frame["era5_tp"].values.ravel())
                pred_plot.set_array(ds_frame["pred_precipitation"].values.ravel())

                new_time_str = ds_frame.time.dt.strftime("%Y-%m-%d %H:%M").item()
                title.set_text(f"Precipitation Comparison at {new_time_str}")

                return era5_plot, pred_plot, title

            ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=200, blit=False)

            if save_filename:
                save_path = self.plots_dir / save_filename
                writer = animation.FFMpegWriter(fps=5, bitrate=1800)
                ani.save(save_path, writer=writer)
                plt.close(fig)
            elif 'get_ipython' in globals() and 'IPython' in globals():
                from IPython.display import HTML
                plt.close(fig)
                return HTML(ani.to_jshtml())
            else:
                plt.tight_layout()
                plt.show()
