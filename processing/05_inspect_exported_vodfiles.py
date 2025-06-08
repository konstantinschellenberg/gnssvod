#!/usr/bin/env python
# -*- coding: utf-8 -*-

from definitions import *
from processing.inspect_vod_funs import *
from gnssvod.io.vodreader import VODReader
from processing.settings import *


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
vod = reader.get_data(format='long')

# todo: to be

# available columns sorted by band (1, 2, 3, 4), satellite gps (gps+gal), sbas (S31, ...)
print("Available columns in vod_ts:")
print(vod.columns.tolist())

# -----------------------------------
# filter for

# temporal_resolution = 60
# angular_resolution = 0.5
# angular_cutoff = 10
#
# vod = vod[(vod['temporal_resolution'] == temporal_resolution) &
#           (vod['angular_resolution'] == angular_resolution) &
#           (vod['angular_cutoff'] == angular_cutoff)].copy()


# -----------------------------------
# selecting only the 'tps' algorithm
# get both algos to compare them
# vod_algo = reader.get_data(format='wide').copy()

# -----------------------------------
# todo: filter for gnss_parameters!!

minimum_nsat = 13
mask = (vod['Ns_t'] >= minimum_nsat)

# make a mask of >10% of VOD1_anom
min_vod_quantile = 0.05
mask2 = (vod['VOD1_anom'] >= vod['VOD1_anom'].quantile(min_vod_quantile))

mask = mask & mask2  # combine both masks

# Filter columns that match the criteria for entire canopy measurements
filtered_vars = []

for col in vod.columns:
    import re
    # Check if column is a VOD band or VOD band extinction coefficient
    is_vod_base = col.startswith('VOD') and len(col) > 3 and col[3].isdigit()
    is_vod_ke = col.startswith('VOD') and '_ke' in col
    
    # Exclude patterns we don't want
    has_bin = bool(re.search(r'bin\d', col))
    has_gps_suffix = col.endswith('gps')
    has_gps_gal_suffix = 'gps+gal' in col
    has_s_in_name = 'S' in col
    
    # Apply all filters
    if (is_vod_base or is_vod_ke or has_bin or has_gps_suffix or has_gps_gal_suffix) and not (has_s_in_name):
        filtered_vars.append(col)

print("Filtered variables for entire canopy:")
print(filtered_vars)

# assign nan to all filtered variables where mask is False
vod.loc[~mask, filtered_vars] = np.nan

vod_ts = vod.copy()  # make a copy of the filtered data

# -----------------------------------
# fail fast ke*d
d_ast = canopy_height - z0  # d_ast is the height of the canopy above the ground receiver

# todo: wouldn't you need to multiply by the mean path length? not the height of the canopy?
# make new columns called <band>_kevod_anom
vod_ts['VOD1_kevod_anom'] = vod_ts["VOD1_ke_anom"] * d_ast
vod_ts['VOD2_kevod_anom'] = vod_ts["VOD2_ke_anom"] * d_ast

# make the mean of VOD1_anom_bin3 and VOD1_anom_bin4
vod_ts['VOD1_anom_highbiomass'] = vod_ts[['VOD1_anom_bin2', 'VOD1_anom_bin3', 'VOD1_anom_bin4']].mean(axis=1)
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
# SIGNAL FILTERING

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# -----------------------------------
# 1. Characterize precipitation patterns using VOD1_anom
# Create flag for upper 10% of VOD1_anom (precipitation events)
daily_dataset = "VOD1_anom_gps+gal"  # Use this dataset for daily mean calculation
precip_threshold = vod_ts[daily_dataset].quantile(0.9)
vod_ts['precip_flag'] = (vod_ts[daily_dataset] > precip_threshold).astype(int)

# Mask anomalies during precipitation events
vod_ts['VOD1_anom_masked'] = vod_ts[daily_dataset].copy()
vod_ts.loc[vod_ts['precip_flag'] == 1, 'VOD1_anom_masked'] = np.nan

# Calculate daily mean VOD only for days with sufficient data (at least 6 hours worth)
# Count non-NaN values per day
print(f"Nan in VOD1_anom_masked: {vod_ts['VOD1_anom_masked'].isna().sum()}")
daily_counts = vod_ts['VOD1_anom_masked'].groupby(pd.Grouper(freq='D')).count()
daily_means = vod_ts['VOD1_anom_masked'].groupby(pd.Grouper(freq='D')).mean()

# Identify days with insufficient data
min_hours_per_day = 12  # At least 6 hours worth of data
time_delta = vod_ts.index[1] - vod_ts.index[0]  # Assuming uniform frequency
min_samples_per_day = min_hours_per_day * (3600 / pd.Timedelta(time_delta).total_seconds())  # Convert to number of samples per day
insufficient_days = daily_counts < min_samples_per_day
daily_means[insufficient_days] = np.nan
print(f"Days with insufficient data: {insufficient_days.sum()} out of {len(daily_counts)} total days")

# Interpolate values for days with insufficient data from surrounding days
interpolated_daily = daily_means.interpolate(method='linear', limit=5)

# Reindex back to original timestamp frequency
vod_ts['VOD1_daily'] = interpolated_daily.reindex(vod_ts.index, method='ffill')

# add mean VOD1 value to VOD1_daily
vod_ts['VOD1_daily'] = vod_ts['VOD1_daily'] + vod_ts['VOD1'].mean()

# -----------------------------------
# 2. Characterize weekly trends using SBAS data
# Normalize each SBAS band by subtracting its mean
sbas_bands = ['VOD1_S33', 'VOD1_S35']
normalized_bands = []

for band in sbas_bands:
    if band in vod_ts.columns:
        band_mean = vod_ts[band].mean()
        normalized_band = f"{band}_norm"
        vod_ts[normalized_band] = vod_ts[band] - band_mean
        normalized_bands.append(normalized_band)

# Calculate mean of normalized SBAS bands for weekly trends
if normalized_bands:
    vod_ts['VOD1_S_weekly'] = vod_ts[normalized_bands].mean(axis=1)
else:
    print("Warning: No SBAS bands found in the dataset")
    vod_ts['VOD1_S_weekly'] = np.nan

# -----------------------------------
# 3. Process diurnal VOD descriptor with Savitzky-Golay filter
if 'VOD1_anom_highbiomass' in vod_ts.columns:
    # Convert to numpy array for filtering, replacing NaNs with interpolation
    diurnal_data = vod_ts['VOD1_anom_highbiomass'].copy()

    # For any remaining NaNs, fall back to hourly means
    hourly_means = diurnal_data.groupby(vod_ts.index.hour).mean()
    for hour, mean_value in hourly_means.items():
        hour_mask = (vod_ts.index.hour == hour) & diurnal_data.isna()
        diurnal_data.loc[hour_mask] = mean_value
    
    # For any still remaining NaNs, use linear interpolation
    diurnal_data = diurnal_data.interpolate(method='linear', limit=6)
    
    # Determine window length (6 hours) based on data frequency
    # Assuming hourly data, adjust if different frequency
    freq_minutes = pd.Timedelta(vod_ts.index[1] - vod_ts.index[0]).total_seconds() / 60
    window_length = int(6 * 60 / freq_minutes)
    
    # Make window length odd (required by savgol_filter)
    window_length = window_length + 1 if window_length % 2 == 0 else window_length
    
    # Apply Savitzky-Golay filter
    filtered_values = savgol_filter(
        diurnal_data.fillna(method='ffill').fillna(method='bfill').values,
        window_length,
        polyorder=2
    )
    
    apply_loess = True  # Set to True to apply LOESS detrending
    if apply_loess:
        # detrend the filtered values by long-term loess filter
        from statsmodels.nonparametric.smoothers_lowess import lowess
        # Apply LOWESS to detrend the filtered values
        lowess_smoothed = lowess(filtered_values, vod_ts.index, frac=0.1, it=0)
        # Subtract the LOWESS smoothed values from the filtered values
        filtered_values = filtered_values - lowess_smoothed[:, 1]
        # Ensure the filtered values are aligned with the original index
        filtered_values = pd.Series(filtered_values, index=vod_ts.index)
        
    # Add filtered diurnal signal
    vod_ts['VOD1_diurnal'] = pd.Series(filtered_values, index=vod_ts.index)
else:
    print("Warning: VOD1_anom_highbiomass not found in the dataset")
    vod_ts['VOD1_diurnal'] = np.nan

# 4. Create a new dataframe with all three components and their sum
vod_optimal = pd.DataFrame({
    'trend': vod_ts['VOD1_daily'],
    'weekly_trend': vod_ts['VOD1_S_weekly'],
    'diurnal': vod_ts['VOD1_diurnal'],
})

# -----------------------------------
# Add the combined optimal estimator with different arithmetic methods

# 1. Simple addition (original method)
vod_optimal['VOD_optimal_add'] = (
        vod_optimal['weekly_trend'].fillna(0) +
        vod_optimal['diurnal'].fillna(0)
)

# 2. Multiplication (components must be positive for meaningful results)
vod_optimal['VOD_optimal_mult'] = (
        vod_optimal['weekly_trend'].fillna(1) *
        vod_optimal['diurnal'].fillna(1)
)

# 3. Weighted mean
weights = {'weekly_trend': 0.3, 'diurnal': 0.4}
weighted_sum = (
        vod_optimal['weekly_trend'].fillna(0) * weights['weekly_trend'] +
        vod_optimal['diurnal'].fillna(0) * weights['diurnal']
)
vod_optimal['VOD_optimal_weighted'] = weighted_sum

# 4. Z-score transformation, sum, and back-transform
# Z-score transform each component
components = ['weekly_trend', 'diurnal']
z_transformed = {}
means = {}
stds = {}

for component in components:
    data = vod_optimal[component].dropna()
    if len(data) > 0:
        means[component] = data.mean()
        stds[component] = data.std()
        z_transformed[component] = (vod_optimal[component] - means[component]) / stds[component]
    else:
        z_transformed[component] = pd.Series(0, index=vod_optimal.index)
        means[component] = 0
        stds[component] = 1

# Sum the z-transformed components
z_sum = (
        z_transformed['weekly_trend'].fillna(0) +
        z_transformed['diurnal'].fillna(0)
)

# Calculate average of means and stds for back-transformation
avg_mean = sum(means.values()) / len(means)
avg_std = sum(stds.values()) / len(stds)

# Back-transform to original scale
vod_optimal['VOD_optimal_zscore'] = z_sum * avg_std + avg_mean

# add the trend to each option
vod_optimal['VOD_optimal_add'] += vod_optimal['trend']
vod_optimal['VOD_optimal_mult'] += vod_optimal['trend']
vod_optimal['VOD_optimal_weighted'] += vod_optimal['trend']
vod_optimal['VOD_optimal_zscore'] += vod_optimal['trend']

# Set the default VOD_optimal to the z-score method
vod_optimal['VOD_optimal'] = vod_optimal['VOD_optimal_zscore']

# Add the optimal estimator back to the original dataframe
vod_ts['VOD_optimal'] = vod_optimal['VOD_optimal']

# Show the first few rows of the new dataframe
print(vod_optimal.head())

# -----------------------------------
# vizualization

plot_vod_fingerprint(vod_optimal, 'VOD_optimal_zscore', title="Optimal VOD (Z-Score Method)")

fingerprint = False  # Set to True to plot the fingerprint of the VOD data
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
    plot_vod_fingerprint(vod_ts, 'VOD1_S_weekly', title="Weekly Mean SBAS VOD1")
    plot_vod_fingerprint(vod_ts, 'VOD1_daily', title="Daily Mean VOD1 Anomaly")
    plot_vod_fingerprint(vod_ts, 'VOD1_diurnal', title="Diurnal VOD1 Anomaly (Savitzky-Golay Filtered)")
    
    # Optimal VOD
    # 1) add
    plot_vod_fingerprint(vod_optimal, 'VOD_optimal_add', title="Optimal VOD (Addition Method)")
    # 2) multiplication
    plot_vod_fingerprint(vod_optimal, 'VOD_optimal_mult', title="Optimal VOD (Multiplication Method)")
    # 3) weighted mean
    plot_vod_fingerprint(vod_optimal, 'VOD_optimal_weighted', title="Optimal VOD (Weighted Mean Method)")
    # 4) z-score
    plot_vod_fingerprint(vod_optimal, 'VOD_optimal_zscore', title="Optimal VOD (Z-Score Method)")

# -----------------------------------
# time series plot
# Static matplotlib plot

figsize = (4, 3)

plot = False
if plot:
    # Components
    plot_vod_timeseries(vod_ts, ['VOD1_S_weekly'], figsize=figsize)
    plot_vod_timeseries(vod_ts, ['VOD1_daily'], figsize=figsize)
    plot_vod_timeseries(vod_ts, ['VOD1_diurnal'], figsize=figsize)
    
    # Result z-score
    plot_vod_timeseries(vod_optimal, ['VOD_optimal_zscore'], interactive=True)

wvlt = False
if wvlt:
    analyze_wavelets(vod_optimal, 'VOD_optimal_zscore')
    
diurnal = False
if diurnal:
    plot_diurnal_cycle(vod_optimal, ['VOD_optimal_zscore'],
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
    plot_histogram(vod_optimal, ['VOD_optimal_zscore', 'trend', 'weekly_trend', 'diurnal'],
                   bins=50, percentiles=[5, 95])

# # -----------------------------------
# # diurnal plot
#
# vars = ['VOD1_anom', 'VOD2_anom']
# diurnal = False
# if diurnal:
#     # Example usage:
#     # Single variables in separate subplots
#
#     """
#     MOST PROMISING PLOT
#     """
#     plot_diurnal_cycle(vod_ts, ['VOD1_anom_highbiomass'],
#                        normalize=None, ncols=1,
#                        title="Daily Normalized Diurnal Cycles",
#                           figsize=(4, 4))
#
#     # Group variables in the same subplots
#     plot_diurnal_cycle(vod_ts, [
#         ('VOD1_S31', 'VOD1_S33', 'VOD1_S35'),
#         ('VOD1_anom_bin0', 'VOD1_anom_bin1', 'VOD1_anom_bin2', 'VOD1_anom_bin3', "VOD1_anom_bin4"),
#         ('VOD1_anom_bin0_gps', 'VOD1_anom_bin1_gps', 'VOD1_anom_bin2_gps', 'VOD1_anom_bin3_gps', "VOD1_anom_bin4_gps"),
#     ], normalize='zscore', ncols=2, figsize=(8, 8), title="Standardized Diurnal Cycles by Group")
#
#
# # -----------------------------------
# # fingerprint plot
# plot = True
# if plot:
#     # plot_vod_fingerprint(vod_ts, 'VOD1', title="Comparing algorithms (1/3)\n No anomaly calculation\n\n band: (L1)")
#
#     # -----------------------------------
#     # extinction fingerprint
#     # plot_vod_fingerprint(vod_algo, 'VOD1_ke_anom_tp')
#     plot_vod_fingerprint(vod_ts, 'VOD1_anom')
#     # plot_vod_fingerprint(vod_ts, 'S1_grn_S35')
#     # plot_vod_fingerprint(vod_ts, 'S1_ref_S35')
#     plot_vod_fingerprint(vod_ts, 'VOD1_S35')
#     # plot_vod_fingerprint(vod_ts, 'S1_grn_S33')
#     # plot_vod_fingerprint(vod_ts, 'S1_ref_S33')
#     plot_vod_fingerprint(vod_ts, 'VOD1_S33')
#     # plot_vod_fingerprint(vod_ts, 'S1_grn_S31')
#     # plot_vod_fingerprint(vod_ts, 'S1_ref_S31')
#     plot_vod_fingerprint(vod_ts, 'VOD1_S31')
#     plot_vod_fingerprint(vod_ts, 'VOD1_S')
#     plot_vod_fingerprint(vod_ts, 'VOD1_anom_highbiomass')
#     # plot_vod_fingerprint(vod_ts, 'VOD1_anom_bin0')
#     # plot_vod_fingerprint(vod_ts, 'VOD1_anom_bin1')
#     # plot_vod_fingerprint(vod_ts, 'VOD1_anom_bin2')
#     # plot_vod_fingerprint(vod_ts, 'VOD1_anom_bin3')
#     # plot_vod_fingerprint(vod_ts, 'VOD1_anom_bin4')
#
#
# # print perc nan in vod_ts, 'S1_grn_S33'
# print("Percentage of NaN values in vod_ts:")
# for col in vod_ts.columns:
#     if 'S' in col and col not in ['hour', 'doy', 'year', 'month', 'day']:
#         perc_nan = vod_ts[col].isna().mean() * 100
#         print(f"{col}: {perc_nan:.2f}% NaN values")
# # -----------------------------------
# # Two-variable scatter plot
# plot = False
# if plot:
#     # With linear fit and custom settings
#     # Compare polarizations using tps algorithm
#     kwargs = {
#         "figsize": (5, 5),
#     }
#     # Custom variable selection
#     # plot_vod_scatter(vod_algo, x_var='VOD1_anom_tp', y_var='VOD2_anom_tps', hue='hour', only_outliers=90, **kwargs)
#     # Compare algorithms for VOD1
#     plot_vod_scatter(vod_algo, polarization='VOD1', algo='compare', hue='doy', only_outliers=90, **kwargs)
#
# # -----------------------------------
# # diurnal power
# vars = ['VOD1_anom', 'VOD2_anom']
# vars = ['VOD1_kevod_anom', 'VOD2_kevod_anom']
# vars = ['VOD1_S31', 'VOD1_S33', 'VOD1_S35']
# plot = True
# if plot:
#     plot_daily_diurnal_range(vod_ts, vars_to_plot=vars,
#                              title="Daily Diurnal Range of VOD1 and VOD2",
#                              qq99=99.5,)
#
#     # plot_daily_diurnal_range(vod_algo, vars_to_plot=['VOD1_anom_tp', 'VOD1_anom_tps'],
#     #                          title="Daily Diurnal Range of VOD1 and VOD2")
#
#
# # -----------------------------------
# # wavelet
#
# wvlt = False
# if wvlt:
#     analyze_wavelets(vod_ts, 'VOD1_anom_highbiomass')
#
#
# # -----------------------------------
# # recreate plot from the literature
# """
# 1. Yitong:
#     - interval: 05-2022 to 11-2023
#     - VOD1_anom
#     - figsize=(5, 3)
#     - daily average (red)
#     - 95% percentile shaded area (light red)
#     - linewidth 1.2
#     - ylims=(0.4, 0.9)
# 2. Humphrey:
#     - interval: 05-2023 to 12-2023
#     - hourly data
#     - figsize=(10, 5)
#     - two curves: "raw" VOD1 (grey) and "processed" VOD1_anom (black), line width 0.8
#     - ylims=(0.6,1)
# 3. Burns
#     - interval: 2023, doy 210-310
#     - hourly data
#     - VOD1_anom
#     - figsize=(5, 3)
#     - ylims=(0, 1)
# """
#
# authors = False
# if authors:
#     # only run, if data (time_subset) encompasses 2022-2024 data
#     if not vod_ts.index.min() < pd.to_datetime('2022-05-01', utc=True) or vod_ts.index.max() > pd.to_datetime('2024-12-31', utc=True):
#         raise ValueError("Data does not cover the required time range for author plots.")
#     # Plot in different author styles
#     plot_vod_by_author(vod_ts, 'yitong')
#     plot_vod_by_author(vod_ts, 'humphrey')
#     plot_vod_by_author(vod_ts, 'burns')