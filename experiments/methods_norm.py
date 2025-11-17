#!/usr/bin/env python
# -*- coding: utf-8 -*-
from cmocean.cm import cmap_d
import pandas as pd

from analysis.config import AnomalyConfig, VodConfig
from definitions import FIG
from analysis.VODProcessor import VODProcessor
from processing.helper import print_color

# from gnssvod.processing.helper import create_time_intervals, print_color

"""
Run step #3 of the gnssvod processing chain:
- Process VOD data for a given time interval
- Calculate anomalies using specified methods
- Visualize results with time series and histograms

This is meant for testing normalization methods and to directly compare results

Maximum time series for this script is about 10 months due to memory constraints.
"""

FIG = FIG / "methods_norm"
FIG.mkdir(parents=True, exist_ok=True)

# -----------------------------------
# FLOW
# long_timeseries = True  # currently not implemented

# -----------------------------------
# LOCAL SETTINGS (that deviate from settings.py)
# interval = ('2023-01-01', '2023-10-31')  # single_file_interval: ('2024-07-15', '2024-08-12') is the longest non-rain period
# interval = ('2024-01-01', '2024-10-31')
interval = ('2024-07-15', '2024-08-12') # single_file_interval: ('2024-07-15', '2024-08-12') is the longest non-rain period
# interval = ('2023-07-15', '2023-08-12') # single_file_interval: ('2024-07-15', '2024-08-12') is the longest non-rain period
# for moflux: 1.6.24 - 1.11.24
interval = ('2024-01-06', '2024-11-06')
# interval = ('2024-01-06', '2024-01-08')


# new parameters for VODProcessor.process_anomaly
station = 'MOz'
visualization_timezone = "etc/GMT+6"

# -----------------------------------
# Configuration dataclasses
vod_cfg = VodConfig(local_file=False, overwrite=False)
cfg = AnomalyConfig(
    angular_resolution=1,
    angular_cutoff=30,
    temporal_resolution=30,
    make_ke=False,
    agg_fun_vodoffset="median",  # Operator used for calculating mean VOD offset
    agg_fun_ts="median",  # Operator used for temporal aggregation of time series
    agg_fun_satincell="median",
    anom_ak_timedelta=pd.Timedelta(days=1),  # temporal window for AK mean VOD offset
    constellations=["gps", "glonass", "galileo"],
    drop_clearsky=False,
    drop_clearsky_threshold=0.1,
    drop_outliersats=True,  # Remove manually inspected orbit outliers
    ks_strategy="con",  # "con" or "sv"
    calculate_biomass_bins=False,
    overwrite=True,  # overwrite existing anomaly results
    show=False
)

# -----------------------------------
# Plotting parameters

main_vodvar = "VOD1_anom"
gnss_band = 1
visualized_anoms = [f"VOD{gnss_band}_anom_ks",
                f"VOD{gnss_band}_anom_ak",
                f"VOD{gnss_band}_anom_vh",
                f"VOD{gnss_band}_anom_ksak"]

visualized_anoms = [
    f"VOD{gnss_band}_anom_ks",
    # f"VOD{gnss_band}_anom_ks_backup",
    # f"VOD{gnss_band}_anom_ks_con",
    # f"VOD{gnss_band}_anom_ks_clearsky",
    # f"VOD{gnss_band}_anom_ks_nooutliers",
]

# -----------------------------------
# make a color map of mid blue to anthracite
from matplotlib.colors import LinearSegmentedColormap

cmap_bluegrey = LinearSegmentedColormap.from_list(
    "midblue_to_anthracite",
    ["#4682B4", "#383E42"],
    N=256
)

cmap_plasma = "plasma"

# -----------------------------------

def main():
    """Process a single time interval with given parameters"""
    start_date, end_date = interval
    print("" + "=" * 50)
    print_color(f"Processing interval: {start_date} to {end_date}")
    
    # -----------------------------------
    # Instantiate processor and process VOD
    processor = VODProcessor(station=station, time_interval=interval)

    # -----------------------------------
    # 1) Process VOD
    # -----------------------------------
    # Input data:
    #   - Dimensions: (sensor, time, SV)
    #   - Temporal spacing: 15 seconds
    # Output data:
    #   - Dimensions: (time, SV)
    #   - Temporal spacing: 15 seconds
    
    # cannot handle more than 10 months at once
    processor.process_vod(cfg=vod_cfg)
    
    # -----------------------------------
    # working on the 15 sec SV dataset
    print_color("Computing satellite statistics...")
    sv_stats = processor.compute_sv_statistics(make=False)
    # save to figdir
    sv_stats.to_csv(FIG / "sv_statistics.csv") if sv_stats is not None else None
    print_color("Satellite statistics:")
    print(sv_stats) if sv_stats is not None else print("No statistics available.")

    # Plot overpass time-of-day densities (example: all constellations, elevation >= 10)
    print_color("Plotting overpass time-of-day densities...")
    processor.plot_overpass_tod(
        constellations=['gps', 'glonass', 'galileo'],
        viz="kde",  # 'hist' or 'kde'
        elevation_min=30,
        elevation_max=None,
        time_zone=visualization_timezone,
        show_sv=True,
        figsize=(8, 6),
        save_path=FIG / "overpass_tod.png",
        interactive=True,
        make=False,
    )
    
    # -----------------------------------
    # 2) ANOMALY DETECTION
    # -----------------------------------
    # Build a concise anomaly configuration and run
    # Input data:
    #   - Dimensions: (time, SV)
    #   - Temporal spacing: 15 seconds
    # Output data:
    #   - Dimensions: (time)
    #   - Temporal spacing: cfg (mostly 30 minutes)
    
    processor.process_anomaly(cfg=cfg)
    print_color("Processing complete.")
    
    # -----------------------------------
    # VISUALIZATION
    # 1) Time series plot, is an xarray
    # print data variables in the xarray
    print(processor.results.data_vars)
    
    processor.plot_diel(
        vars=visualized_anoms,
        time_zone=visualization_timezone,
        y_range=(0.4, 0.8),
        figsize=(8, 4),
        gradient_cmap=cmap_plasma,
        save_path=FIG / "vod_diel.png",
        make=True
    )
    
    # -----------------------------------
    # 2) Histograms
    processor.plot_hist(
        vars=visualized_anoms,
        bins=50,
        cmap=cmap_plasma,
        sharex=True,
        combine=True,
        figsize=(4,3),
        save_path=FIG / "vod_hist.png",
        make=False
    )
    
    # -----------------------------------
    # 3) Time series plot
    processor.plot_time_series(
        vars=visualized_anoms,
        time_zone=visualization_timezone,
        figsize=(10, 7),
        gradient_cmap=cmap_plasma,
        save_path=FIG / "vod_timeseries.png",
        make=False,
        clip_percentiles=(0.5, 99.5)  # NEW: percentile clipping
    )
    
    # -----------------------------------
    # 4) correlation matrix plot
    processor.plot_correlation_matrix(
        vars=visualized_anoms,
        figsize=(6, 5),
        cmap=cmap_plasma,
        save_path=FIG / "vod_correlation_matrix.png",
        make=False
    )
    
    # -----------------------------------
    # X) Hemispheric mean and standard deviation plot
    print(processor.hemi.columns.tolist())
    make_hemi = False
    processor.plot_hemispheric(var="VOD1_mean", type="vod", angle_res=cfg.angular_resolution,
                              angle_cutoff=30., clim="auto", title="MOz: VOD1 Mean", make=make_hemi)
    # processor.plot_hemispheric(var="VOD1_std", type="std", angle_res=cfg.angular_resolution,
    #                           angle_cutoff=30., clim="perc_975", title="MOz: VOD1 Standard Deviation", make=make_hemi)

if __name__ == "__main__":
    main()