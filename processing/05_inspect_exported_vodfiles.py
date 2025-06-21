#!/usr/bin/env python
# -*- coding: utf-8 -*-

from definitions import *
from processing.settings import *
from processing.inspect_vod_funs import *

FIG = FIG / "vod_inspection"
FIG.mkdir(parents=True, exist_ok=True)

# -----------------------------------
# local settings

plotall = False  # Set to True to plot all VOD data

# -----------------------------------

# read parquet to pandas DataFrame
vod_ts = pd.read_parquet(vod_file, engine='pyarrow')

if visualization_timezone:
    # Convert index to timezone-aware datetime if not already
    vod_ts.index = pd.to_datetime(vod_ts.index, utc=True).tz_convert(visualization_timezone)

# calc prod of shape
# np.prod(vod_ts.shape)

# -----------------------------------
# vizualization

fingerprint = True  # Set to True to plot the fingerprint of the VOD data
if fingerprint or plotall:
    # Plot the fingerprint of the VOD data
    
    # Trend
    plot_vod_fingerprint(vod_ts, 'VOD1_anom', title="VOD1 Anomaly", save_dir = FIG)
    plot_vod_fingerprint(vod_ts, 'VOD1_anom_tp', title="VOD1 Anomaly", save_dir = FIG)
    plot_vod_fingerprint(vod_ts, 'VOD1_anom_gps+gal', title="VOD1 Anomaly (GPS+Galileo)",save_dir = FIG)
    plot_vod_fingerprint(vod_ts, 'VOD1_anom_bin3-5_gps+gal', title="VOD1 Anomaly (GPS+Galileo)\n dense",save_dir = FIG)
    plot_vod_fingerprint(vod_ts, 'wetness_flag',save_dir = FIG)
    plot_vod_fingerprint(vod_ts, 'VOD1_daily', title="Daily Mean VOD1 Anomaly",save_dir = FIG)
    #
    # # seasonal VOD
    plot_vod_fingerprint(vod_ts, 'VOD1_SBAS_anom', title="Weekly Mean SBAS VOD1",save_dir = FIG)
    plot_vod_fingerprint(vod_ts, 'VOD1_S33', title="SBAS VOD1 S33",save_dir = FIG)
    plot_vod_fingerprint(vod_ts, 'VOD1_S35', title="SBAS VOD1 S35",save_dir = FIG)
    #
    # # diurnal VOD
    plot_vod_fingerprint(vod_ts, 'VOD1_anom_bin3-5_gps+gal', title="VOD1 Anomaly (GPS+Galileo)\ndense",save_dir = FIG)
    plot_vod_fingerprint(vod_ts, 'VOD1_diurnal', title="Diurnal",save_dir = FIG)
    
    # Result
    plot_vod_fingerprint(vod_ts, 'VOD1_daily', title="Daily Mean VOD1 Anomaly",save_dir = FIG)
    plot_vod_fingerprint(vod_ts, 'VOD1_SBAS_anom', title="Weekly Mean SBAS VOD1")
    plot_vod_fingerprint(vod_ts, 'VOD1_diurnal', title="Diurnal VOD1 Anomaly")
    
    # Optimal VOD
    # 1) add
    plot_vod_fingerprint(vod_ts, 'VOD_optimal_add', title="Optimal VOD (Addition Method)",save_dir = FIG)
    # # 2) multiplication
    # plot_vod_fingerprint(vod_ts, 'VOD_optimal_mult', title="Optimal VOD (Multiplication Method)",save_dir = FIG)
    # # 3) weighted mean
    # plot_vod_fingerprint(vod_ts, 'VOD_optimal_weighted', title="Optimal VOD (Weighted Mean Method)",save_dir = FIG)
    # 4) z-score
    plot_vod_fingerprint(vod_ts, 'VOD_optimal_zscore', title="Optimal VOD (Z-Score Method)",save_dir = FIG)

# -----------------------------------
# time series plot
# Static matplotlib plot

figsize = (4, 3)

plot = True
if plot or plotall:
    # Components
    plot_vod_timeseries(vod_ts, ['VOD_optimal_zscore', 'VOD1_SBAS_anom', 'VOD1_daily', 'VOD1_diurnal'], figsize=figsize, save_dir = FIG,
                        legend_loc="lower left")
    # plot_vod_timeseries(vod_ts, ['VOD1_daily'], figsize=figsize, save_dir = FIG)
    # plot_vod_timeseries(vod_ts, ['VOD1_diurnal'], figsize=figsize, save_dir = FIG)
    
    # Result z-score
    # plot_vod_timeseries(vod_ts, ['VOD1_anom'], interactive=True)

# -----------------------------------
wvlt = False
if wvlt or plotall:
    analyze_wavelets(vod_ts, 'VOD1_SBAS_anom')
    analyze_wavelets(vod_ts, 'VOD1_daily')
    analyze_wavelets(vod_ts, 'VOD1_diurnal')
    analyze_wavelets(vod_ts, 'VOD_optimal_zscore')

# -----------------------------------
diurnal = False
if diurnal or plotall:
    vod_yitong = vod_ts.copy()
    # only data between 2022-06-01 and 2022-10-31
    vod_yitong = vod_yitong[(vod_yitong.index >= pd.to_datetime('2022-05-30', utc=True)) &
                            (vod_yitong.index <= pd.to_datetime('2022-08-30', utc=True))]
    plot_diurnal_cycle(vod_yitong, ['VOD1_anom_tp'],
                       normalize=None, ncols=1,
                       show_std = False,
                       title="Diurnal Cycle of VOD1 Anomaly (Yitong)",
                      figsize=(4, 2.5), save_dir = FIG)
    plot_diurnal_cycle(vod_ts, ['VOD_optimal_zscore'],
                           normalize=None, ncols=1,
                           title="Optimal Diurnal Cycles",
                          figsize=(4, 4), save_dir = FIG)
    plot_diurnal_cycle(vod_ts, ['VOD1_SBAS_anom'],
                           normalize=None, ncols=1,
                           title="SBAS diurnal Cycles",
                          figsize=(4, 4), save_dir = FIG)
    plot_diurnal_cycle(vod_ts, ['VOD1_diurnal'],
                           normalize=None, ncols=1,
                           title="High Biomass diurnal Cycles",
                          figsize=(4, 4), save_dir = FIG)

hist = True
if hist or plotall:
    # Single histogram
    # Multiple histograms in a grid
    plot_histogram(vod_ts, ['VOD_optimal_zscore', 'VOD1_SBAS_anom', 'VOD1_daily', 'VOD1_diurnal'],
                   bins=50, percentiles=[5, 95], save_dir = FIG)

# -----------------------------------
authors = False
if authors:
    # only run, if data (time_subset) encompasses 2022-2024 data
    # if not vod_ts.index.min() < pd.to_datetime('2022-05-01', utc=True) or vod_ts.index.max() > pd.to_datetime('2024-12-31', utc=True):
    #     raise ValueError("Data does not cover the required time range for author plots.")
    # Plot in different author styles
    plot_vod_by_author(vod_ts, 'yitong')
    # plot_vod_by_author(vod_ts, 'humphrey')
    # plot_vod_by_author(vod_ts, 'burns')