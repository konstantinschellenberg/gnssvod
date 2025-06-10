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
# calc prod of shape
# np.prod(vod_ts.shape)

# -----------------------------------
# vizualization

plot_vod_fingerprint(vod_ts, 'VOD_optimal_zscore', title="Optimal VOD (Z-Score Method)")
# plot_vod_fingerprint(vod_ts, 'VOD1_S_weekly', title="Weekly Mean SBAS VOD1", save_dir=FIG)
# plot_vod_fingerprint(vod_ts, 'VOD1_S_weekly_trend', title="Weekly Mean SBAS VOD1", save_dir=FIG)

fingerprint = False  # Set to True to plot the fingerprint of the VOD data
if fingerprint or plotall:
    # Plot the fingerprint of the VOD data
    
    # Trend
    plot_vod_fingerprint(vod_ts, 'VOD1_anom', title="VOD1 Anomaly", save_dir = FIG)
    plot_vod_fingerprint(vod_ts, 'VOD1_anom_gps+gal', title="VOD1 Anomaly (GPS+Galileo)",save_dir = FIG)
    plot_vod_fingerprint(vod_ts, 'precip_flag',save_dir = FIG)
    plot_vod_fingerprint(vod_ts, 'VOD1_anom_masked', title="Masked VOD1 Anomaly (trend Events)",save_dir = FIG)
    plot_vod_fingerprint(vod_ts, 'VOD1_daily', title="Daily Mean VOD1 Anomaly",save_dir = FIG)
    
    # seasonal VOD
    plot_vod_fingerprint(vod_ts, 'VOD1_S_weekly', title="Weekly Mean SBAS VOD1",save_dir = FIG)
    plot_vod_fingerprint(vod_ts, 'VOD1_S33', title="SBAS VOD1 S33",save_dir = FIG)
    plot_vod_fingerprint(vod_ts, 'VOD1_S35', title="SBAS VOD1 S35",save_dir = FIG)

    # diurnal VOD
    plot_vod_fingerprint(vod_ts, 'VOD1_anom_highbiomass', title="VOD1 Anomaly High Biomass",save_dir = FIG)
    plot_vod_fingerprint(vod_ts, 'VOD1_diurnal', title="Diurnal VOD1 Anomaly (Savitzky-Golay Filtered)",save_dir = FIG)
    
    # Result
    # plot_vod_fingerprint(vod_ts, 'VOD1_S_weekly', title="Weekly Mean SBAS VOD1")
    # plot_vod_fingerprint(vod_ts, 'VOD1_daily', title="Daily Mean VOD1 Anomaly")
    # plot_vod_fingerprint(vod_ts, 'VOD1_diurnal', title="Diurnal VOD1 Anomaly (Savitzky-Golay Filtered)")
    
    # Optimal VOD
    # 1) add
    plot_vod_fingerprint(vod_ts, 'VOD_optimal_add', title="Optimal VOD (Addition Method)",save_dir = FIG)
    # 2) multiplication
    plot_vod_fingerprint(vod_ts, 'VOD_optimal_mult', title="Optimal VOD (Multiplication Method)",save_dir = FIG)
    # 3) weighted mean
    plot_vod_fingerprint(vod_ts, 'VOD_optimal_weighted', title="Optimal VOD (Weighted Mean Method)",save_dir = FIG)
    # 4) z-score
    plot_vod_fingerprint(vod_ts, 'VOD_optimal_zscore', title="Optimal VOD (Z-Score Method)",save_dir = FIG)

# -----------------------------------
# time series plot
# Static matplotlib plot

figsize = (4, 3)

plot = False
if plot or plotall:
    # Components
    plot_vod_timeseries(vod_ts, ['VOD1_S_weekly'], figsize=figsize, save_dir = FIG)
    plot_vod_timeseries(vod_ts, ['VOD1_daily'], figsize=figsize, save_dir = FIG)
    plot_vod_timeseries(vod_ts, ['VOD1_diurnal'], figsize=figsize, save_dir = FIG)
    
    # Result z-score
    plot_vod_timeseries(vod_ts, ['VOD_optimal_zscore'], interactive=True)

# -----------------------------------
wvlt = False
if wvlt or plotall:
    analyze_wavelets(vod_ts, 'VOD1_S_weekly')
    
# -----------------------------------
diurnal = False
if diurnal or plotall:
    plot_diurnal_cycle(vod_ts, ['VOD_optimal_zscore'],
                           normalize=None, ncols=1,
                           title="Optimal Diurnal Cycles",
                          figsize=(4, 4), save_dir = FIG)
    plot_diurnal_cycle(vod_ts, ['VOD1_S_weekly'],
                           normalize=None, ncols=1,
                           title="SBAS diurnal Cycles",
                          figsize=(4, 4), save_dir = FIG)
    plot_diurnal_cycle(vod_ts, ['VOD1_diurnal'],
                           normalize=None, ncols=1,
                           title="High Biomass diurnal Cycles",
                          figsize=(4, 4), save_dir = FIG)

hist = False
if hist or plotall:
    # Single histogram
    # Multiple histograms in a grid
    plot_histogram(vod_ts, ['VOD_optimal_zscore', 'trend', 'weekly_trend', 'diurnal'],
                   bins=50, percentiles=[5, 95], save_dir = FIG)

# -----------------------------------
# authors = False
# if authors:
#     # only run, if data (time_subset) encompasses 2022-2024 data
#     if not vod_ts.index.min() < pd.to_datetime('2022-05-01', utc=True) or vod_ts.index.max() > pd.to_datetime('2024-12-31', utc=True):
#         raise ValueError("Data does not cover the required time range for author plots.")
#     # Plot in different author styles
#     plot_vod_by_author(vod_ts, 'yitong')
#     plot_vod_by_author(vod_ts, 'humphrey')
#     plot_vod_by_author(vod_ts, 'burns')