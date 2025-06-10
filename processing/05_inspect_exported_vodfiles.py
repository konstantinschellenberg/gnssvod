#!/usr/bin/env python
# -*- coding: utf-8 -*-

from definitions import *
from processing.inspect_vod_funs import *
from gnssvod.io.vodreader import VODReader
from processing.metadata import create_vod_metadata
from processing.settings import *
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# -----------------------------------
# Define the load mode
if load_mode == 'single_file':
    reader = VODReader(single_file_settings)
elif load_mode == 'multi_year':
    reader = VODReader(gatheryears=time_intervals)
elif load_mode == 'final_vod':
    reader = VODReader(file_path_or_settings=DATA / "ard" / final_vod_path,)
else:
    raise ValueError(f"Unknown load mode: {load_mode}. Please choose 'single_file', 'multi_year', or 'final_vod'.")

# -----------------------------------
# Get long data
vod_ts = reader.get_data(format='long')

# available columns sorted by band (1, 2, 3, 4), satellite gps (gps+gal), sbas (S31, ...)
print("Available columns in vod_ts:")
print(vod_ts.columns.tolist())
# -----------------------------------
# subset time

if time_subset:
    # Convert time_subset to datetime if it's a string
    if isinstance(time_subset, str):
        time_subset = pd.to_datetime(time_subset, format='%Y-%m-%d', utc=True).tz_convert(visualization_timezone)
    
    # Filter vod_ts based on the time_subset
    vod_ts = vod_ts[vod_ts.index >= time_subset[0]]
    vod_ts = vod_ts[vod_ts.index <= time_subset[1]]

vod_ts.index.freq = pd.infer_freq(vod_ts.index)  # infer frequency of the time series

# -----------------------------------
# filter for

# todo: This is an oddity for now as the metrics are written in rows while the rest of the data variability is cols
# temporal_resolution = 60
# angular_resolution = 0.5
# angular_cutoff = 10
#
# vod_ts = vod_ts[(vod_ts['temporal_resolution'] == temporal_resolution) &
#           (vod_ts['angular_resolution'] == angular_resolution) &
#           (vod_ts['angular_cutoff'] == angular_cutoff)].copy()

# -----------------------------------
# KE
# todo: wouldn't you need to multiply by the mean path length? not the height of the canopy?
# d_ast = canopy_height - z0  # d_ast is the height of the canopy above the ground receiver
# # make new columns called <band>_kevod_anom
# vod_ts['VOD1_kevod_anom'] = vod_ts["VOD1_ke_anom"] * d_ast
# vod_ts['VOD2_kevod_anom'] = vod_ts["VOD2_ke_anom"] * d_ast


minimum_nsat = 13
min_vod_quantile = 0.05

metadata = create_vod_metadata()

# filters low-availability data
total_nan_prior = vod_ts.isna().sum().sum()
nsat_mask = create_satellite_mask(vod_ts, min_satellites=minimum_nsat, satellite_col="Ns_t")
satellite_mask_columns = filter_vod_columns(vod_ts, exclude_metrics=True, exclude_sbas=True)

# apply mask to nan to the cols in satellite_mask_columns but keep all cols
vod_ts[satellite_mask_columns] = vod_ts[satellite_mask_columns].where(nsat_mask, np.nan)
total_nan_after = vod_ts.isna().sum().sum()
print(f"Filtered VOD data to {vod_ts.shape[0]} rows with at least {minimum_nsat} satellites.")
print(f"Total NaN values before filtering: {total_nan_prior}, after filtering: {total_nan_after}, filtered out: {total_nan_after - total_nan_prior}")
print("+" * 50)

# filter dip-artifacts
total_nan_prior = vod_ts.isna().sum().sum()
percvod_mask = create_vod_percentile_mask(vod_ts, vod_column="VOD1_anom", percentile=min_vod_quantile)
vod_percentile_columns1 = filter_vod_columns(vod_ts, column_type='binned anom', exclude_sbas=True, is_binned=True)
vod_percentile_columns2 = filter_vod_columns(vod_ts, column_type='anom', exclude_sbas=True, is_binned=False)
vod_perc_cols = vod_percentile_columns1 + vod_percentile_columns2
# apply filter
vod_ts[vod_perc_cols] = vod_ts[vod_perc_cols].where(percvod_mask, np.nan)
total_nan_after = vod_ts.isna().sum().sum()
print(f"Filtered VOD data to {vod_ts.shape[0]} rows with VOD1_anom above the {min_vod_quantile * 100}% quantile.")
print(f"Total NaN values before filtering: {total_nan_prior}, after filtering: {total_nan_after}, filtered out: {total_nan_after - total_nan_prior}")
print("+" * 50)

# -----------------------------------
# HIGHBIOMASS
# make the mean of VOD1_anom_bin3 and VOD1_anom_bin4
vod_ts['VOD1_anom_highbiomass'] = vod_ts[['VOD1_anom_bin2', 'VOD1_anom_bin3', 'VOD1_anom_bin4']].mean(axis=1)
# -----------------------------------
"""
Gallery:
---------

Goals:
1. Best long-term vod estimate
2. Best weekly vod (dry-down)
3. Best diurnal vod (daily)

Manufacture signal by
VOD = 1 + 2 + 3

-----
1. Best long-term vod estimate
    - Fullest canopy view of high-credibility constellations:
    - Includes canopy gaps to
    
    Tests:
    * compare with literature VOD values
    
    Candidates:
    * mean (VOD(GPS, GLONASS)), θ > 30°
    
-----
2. Best weekly vod (dry-down)
    - Usually not diurnal trend
    - Precipitation events
    - Trend
    
    Tests:
    * compare with radiometer
    * compare with branch water potential
    
    Candidates:
    * SBAS (33 > 35 > [31]*excluded) –> all or mixture?
    
    Todo:
    - normalize magnitude to VOD1_anom magnitude

-----
3. Best diurnal vod (daily)
    - Strong diurnal vars don't show seasonal trend and precipitation (but dew)
    
    Tests:
    * compare with branch water potential
    * independent of temperature?
    
    Candidates:
    * VOD1_anom_highbiomass (VOD1_anom_bin2, VOD1_anom_bin3, VOD1_anom_bin4) (60% of high biomass)
        - does not show strong seasonal trend
        - does not show precipitation events

    Todo:
    * heavy smoothing (savitzky-golay filter)
    * detrending (lowess filter) –> the overall dynamics are reduce...

    
    
Sbas:
- Why are stripes in S31 ref? is the WAAS sat moving? However, not seen in VOD :)

Claude 3.7 Sonnet Thinking:
I want you to do the following analysis. I want to create an optimal VOD estimator variable that harnesses the different strength of signales (cols) as desribed in  the underneat text. Please 1)  Characterize Precipitation patterns using temperal anomalies in VOD1_anom (create a flag based on the quantile of upper 10% of the signal). Mask anom by the flag. Calculate mean daily VOD values (transform not summarize) and add to vod_ts. 2) use the SBAS data to characterize the weekly trends (dry-down events). First, subtract the mean of each VOD1_S?? band from the time series, then caluculate the mean VOD1_S. 3) The best diurnal VOD descriptor is VOD1_anom_highbiomass. Please apply a window-size (6 hours, polyorder 2) savitzky-golay filter to the data. Finally merge all three product to a new dataframe and add a new column where all of the them are added
"""

# -----------------------------------
# 1. Characterize precipitation patterns using VOD1_anom

precips = characterize_precipitation(vod_ts, dataset_col='VOD1_anom_gps+gal', precip_quantile=0.9,
                               min_hours_per_day=12)

weekly = characterize_weekly_trends(vod_ts, sbas_bands=['VOD1_S33', 'VOD1_S35'])

diurnal = process_diurnal_vod(vod_ts, diurnal_col='VOD1_anom_highbiomass',
                        window_hours=6, polyorder=2, apply_loess=True,
                        loess_frac=0.1)

# Combine the results
optimal_vod = "VOD_optimal_zscore"
vod_optimal = create_optimal_estimator(
    pd.DataFrame({
        'VOD1_daily': precips['VOD1_daily'],
        'VOD1_S_weekly': weekly['VOD1_S_weekly'],
        'VOD1_diurnal': diurnal['VOD1_diurnal']
    }, index=vod_ts.index)
)

# Add the optimal estimator back to the original dataframe
vod_optimal['VOD_optimal'] = vod_optimal[optimal_vod]

# join on index with vod_ts
intermediate_steps = pd.DataFrame({
    'VOD1_daily': precips['VOD1_daily'],
    'VOD1_S_weekly': weekly['VOD1_S_weekly'],
    'VOD1_diurnal': diurnal['VOD1_diurnal']
}, index=vod_ts.index
)

# Only join columns that don't already exist in vod_ts
vod_ts = vod_ts.join(vod_optimal[[col for col in vod_optimal.columns if col not in vod_ts.columns]])
vod_ts = vod_ts.join(precips[[col for col in precips.columns if col not in vod_ts.columns]])
vod_ts = vod_ts.join(weekly[[col for col in weekly.columns if col not in vod_ts.columns]])
vod_ts = vod_ts.join(diurnal[[col for col in diurnal.columns if col not in vod_ts.columns]])

# sort columns by alphanumeric order
vod_ts = vod_ts.reindex(sorted(vod_ts.columns), axis=1)

# -----------------------------------
# vizualization

plot_vod_fingerprint(vod_ts, 'VOD_optimal_zscore', title="Optimal VOD (Z-Score Method)")

fingerprint = True  # Set to True to plot the fingerprint of the VOD data
if fingerprint:
    # Plot the fingerprint of the VOD data
    
    # Trend
    plot_vod_fingerprint(vod_ts, 'VOD1_anom', title="VOD1 Anomaly")
    plot_vod_fingerprint(vod_ts, 'VOD1_anom_gps+gal', title="VOD1 Anomaly (GPS+Galileo)")
    plot_vod_fingerprint(vod_ts, 'precip_flag')
    plot_vod_fingerprint(vod_ts, 'VOD1_anom_masked', title="Masked VOD1 Anomaly (trend Events)")
    plot_vod_fingerprint(vod_ts, 'VOD1_daily', title="Daily Mean VOD1 Anomaly")
    
    # seasonal VOD
    plot_vod_fingerprint(vod_ts, 'VOD1_S_weekly', title="Weekly Mean SBAS VOD1")
    plot_vod_fingerprint(vod_ts, 'VOD1_S33', title="SBAS VOD1 S33")
    plot_vod_fingerprint(vod_ts, 'VOD1_S35', title="SBAS VOD1 S35")

    # diurnal VOD
    plot_vod_fingerprint(vod_ts, 'VOD1_anom_highbiomass', title="VOD1 Anomaly High Biomass")
    plot_vod_fingerprint(vod_ts, 'VOD1_diurnal', title="Diurnal VOD1 Anomaly (Savitzky-Golay Filtered)")
    
    # Result
    # plot_vod_fingerprint(vod_ts, 'VOD1_S_weekly', title="Weekly Mean SBAS VOD1")
    # plot_vod_fingerprint(vod_ts, 'VOD1_daily', title="Daily Mean VOD1 Anomaly")
    # plot_vod_fingerprint(vod_ts, 'VOD1_diurnal', title="Diurnal VOD1 Anomaly (Savitzky-Golay Filtered)")
    
    # Optimal VOD
    # 1) add
    plot_vod_fingerprint(vod_ts, 'VOD_optimal_add', title="Optimal VOD (Addition Method)")
    # 2) multiplication
    plot_vod_fingerprint(vod_ts, 'VOD_optimal_mult', title="Optimal VOD (Multiplication Method)")
    # 3) weighted mean
    plot_vod_fingerprint(vod_ts, 'VOD_optimal_weighted', title="Optimal VOD (Weighted Mean Method)")
    # 4) z-score
    plot_vod_fingerprint(vod_ts, 'VOD_optimal_zscore', title="Optimal VOD (Z-Score Method)")

# -----------------------------------
# time series plot
# Static matplotlib plot

figsize = (4, 3)

plot = True
if plot:
    # Components
    plot_vod_timeseries(vod_ts, ['VOD1_S_weekly'], figsize=figsize)
    plot_vod_timeseries(vod_ts, ['VOD1_daily'], figsize=figsize)
    plot_vod_timeseries(vod_ts, ['VOD1_diurnal'], figsize=figsize)
    
    # Result z-score
    plot_vod_timeseries(vod_optimal, ['VOD_optimal_zscore'], interactive=True)

wvlt = False
if wvlt:
    analyze_wavelets(vod_ts, 'VOD_optimal_zscore')
    
diurnal = True
if diurnal:
    plot_diurnal_cycle(vod_ts, ['VOD_optimal_zscore'],
                           normalize=None, ncols=1,
                           title="Optimal Diurnal Cycles",
                          figsize=(4, 4))
    plot_diurnal_cycle(vod_ts, ['VOD1_S_weekly'],
                           normalize=None, ncols=1,
                           title="SBAS diurnal Cycles",
                          figsize=(4, 4))
    plot_diurnal_cycle(vod_ts, ['VOD1_diurnal'],
                           normalize=None, ncols=1,
                           title="High Biomass diurnal Cycles",
                          figsize=(4, 4))

hist = True
if hist:
    # Single histogram
    # Multiple histograms in a grid
    plot_histogram(vod_ts, ['VOD_optimal_zscore', 'trend', 'weekly_trend', 'diurnal'],
                   bins=50, percentiles=[5, 95])

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