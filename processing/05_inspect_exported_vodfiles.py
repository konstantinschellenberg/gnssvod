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
mask_qq10 = vod['VOD1_anom'].quantile(0.05)
mask2 = (vod['VOD1_anom'] >= mask_qq10)
mask = mask & mask2  # combine both masks

vod_ts = vod[mask].copy()  # filter for minimum number of satellites
# for all cols, set mask to nan
for col in vod_ts.columns:
    # mustn't contain 'S' to avoid SBAS columns, not in ['hour', 'doy', 'year', 'month', 'day']
    if 'S' not in col and col not in ['hour', 'doy', 'year', 'month', 'day']:
        print(f"Setting {col} to NaN where mask is False")
        vod_ts.loc[~mask, col] = pd.NA  # set to NaN where mask is False

# -----------------------------------
# fail fast ke*d
d_ast = canopy_height - z0  # d_ast is the height of the canopy above the ground receiver

# todo: wouldn't you need to multiply by the mean path length? not the height of the canopy?
# make new columns called <band>_kevod_anom
vod_ts['VOD1_kevod_anom'] = vod_ts["VOD1_ke_anom"] * d_ast
vod_ts['VOD2_kevod_anom'] = vod_ts["VOD2_ke_anom"] * d_ast

# make the mean of VOD1_anom_bin3 and VOD1_anom_bin4
vod_ts['VOD1_anom_highbiomass'] = vod_ts[['VOD1_anom_bin3', 'VOD1_anom_bin4']].mean(axis=1)
# -----------------------------------
# subset time

if time_subset:
    # Convert time_subset to datetime if it's a string
    if isinstance(time_subset, str):
        time_subset = pd.to_datetime(time_subset, format='%Y-%m-%d', utc=True).tz_convert(visualization_timezone)
    
    # Filter vod_ts based on the time_subset
    vod_ts = vod_ts[vod_ts.index >= time_subset[0]]
    vod_ts = vod_ts[vod_ts.index <= time_subset[1]]
    

# -----------------------------------
# time series plot
# Static matplotlib plot

figsize = (7, 4)

plot = False
if plot:
    plot_vod_timeseries(vod_ts, ['VOD1'], figsize=figsize)
    plot_vod_timeseries(vod_ts, ['VOD1_anom'], figsize=figsize)
    plot_vod_timeseries(vod_ts, ['VOD1_kevod_anom'], figsize=figsize)
    plot_vod_timeseries(vod_ts, ['VOD1_ke_anom'], figsize=figsize)
    plot_vod_timeseries(vod_ts, ['VOD1_S31'], figsize=figsize)
    plot_vod_timeseries(vod_ts, ['VOD1_S33'], figsize=figsize)
    plot_vod_timeseries(vod_ts, ['VOD1_S35'], figsize=figsize)
    plot_vod_timeseries(vod_ts, ['VOD1_S'], figsize=figsize)
    plot_vod_timeseries(vod_ts, ['VOD1_anom_highbiomass'], figsize=figsize, title="VOD1 Anomaly High Biomass")


# -----------------------------------
# diurnal plot

vars = ['VOD1_anom', 'VOD2_anom']
diurnal = False
if diurnal:
    # Example usage:
    # Single variables in separate subplots
    
    """
    MOST PROMISING PLOT
    """
    plot_diurnal_cycle(vod_ts, ['VOD1_anom_highbiomass'],
                       normalize=None, ncols=1,
                       title="Daily Normalized Diurnal Cycles",
                          figsize=(4, 4))
    
    # Group variables in the same subplots
    plot_diurnal_cycle(vod_ts, [
        ('VOD1_S31', 'VOD1_S33', 'VOD1_S35'),
        ('VOD1_anom_bin0', 'VOD1_anom_bin1', 'VOD1_anom_bin2', 'VOD1_anom_bin3', "VOD1_anom_bin4"),
        ('VOD1_anom_bin0_gps', 'VOD1_anom_bin1_gps', 'VOD1_anom_bin2_gps', 'VOD1_anom_bin3_gps', "VOD1_anom_bin4_gps"),
    ], normalize='zscore', ncols=2, figsize=(8, 8), title="Standardized Diurnal Cycles by Group")


# -----------------------------------
# fingerprint plot
plot = True
if plot:
    # plot_vod_fingerprint(vod_ts, 'VOD1', title="Comparing algorithms (1/3)\n No anomaly calculation\n\n band: (L1)")
    
    # -----------------------------------
    # extinction fingerprint
    # plot_vod_fingerprint(vod_algo, 'VOD1_ke_anom_tp')
    plot_vod_fingerprint(vod_ts, 'VOD1', title="VOD")
    plot_vod_fingerprint(vod_ts, 'VOD1_anom', title="VOD")
    plot_vod_fingerprint(vod_ts, 'VOD1_anom_tp', title="VOD")
    plot_vod_fingerprint(vod_ts, 'VOD1_ke_anom', title="VOD1 Anomaly Fingerprint\n TPS Algorithm")
    plot_vod_fingerprint(vod_ts, 'VOD1_kevod_anom', title="$VOD_{rectified} = k_e * d$")
    plot_vod_fingerprint(vod_ts, 'VOD1_anom', title="VOD")
    plot_vod_fingerprint(vod_ts, 'S1_grn_S35', title="VOD")
    plot_vod_fingerprint(vod_ts, 'S1_ref_S35', title="VOD")
    plot_vod_fingerprint(vod_ts, 'VOD1_S35', title="VOD")
    plot_vod_fingerprint(vod_ts, 'S1_grn_S33', title="VOD")
    plot_vod_fingerprint(vod_ts, 'S1_ref_S33', title="VOD")
    plot_vod_fingerprint(vod_ts, 'VOD1_S33', title="VOD")
    plot_vod_fingerprint(vod_ts, 'S1_grn_S31', title="VOD")
    plot_vod_fingerprint(vod_ts, 'S1_ref_S31', title="VOD")
    plot_vod_fingerprint(vod_ts, 'VOD1_S31', title="VOD")
    plot_vod_fingerprint(vod_ts, 'Ns_t', title="VOD")
    plot_vod_fingerprint(vod_ts, 'VOD1_anom_highbiomass', title="VOD")
    plot_vod_fingerprint(vod_ts, 'VOD1_anom_bin0', title="VOD")
    plot_vod_fingerprint(vod_ts, 'VOD1_anom_bin1', title="VOD")
    plot_vod_fingerprint(vod_ts, 'VOD1_anom_bin2', title="VOD")
    plot_vod_fingerprint(vod_ts, 'VOD1_anom_bin3', title="VOD")
    plot_vod_fingerprint(vod_ts, 'VOD1_anom_bin4', title="VOD")

# -----------------------------------
# Two-variable scatter plot
plot = False
if plot:
    # With linear fit and custom settings
    # Compare polarizations using tps algorithm
    kwargs = {
        "figsize": (5, 5),
    }
    # Custom variable selection
    # plot_vod_scatter(vod_algo, x_var='VOD1_anom_tp', y_var='VOD2_anom_tps', hue='hour', only_outliers=90, **kwargs)
    # Compare algorithms for VOD1
    plot_vod_scatter(vod_algo, polarization='VOD1', algo='compare', hue='doy', only_outliers=90, **kwargs)
    
# -----------------------------------
# diurnal power
vars = ['VOD1_anom', 'VOD2_anom']
vars = ['VOD1_kevod_anom', 'VOD2_kevod_anom']
vars = ['VOD1_S31', 'VOD1_S33', 'VOD1_S35']
plot = True
if plot:
    plot_daily_diurnal_range(vod_ts, vars_to_plot=vars,
                             title="Daily Diurnal Range of VOD1 and VOD2",
                             qq99=99.5,)
    
    # plot_daily_diurnal_range(vod_algo, vars_to_plot=['VOD1_anom_tp', 'VOD1_anom_tps'],
    #                          title="Daily Diurnal Range of VOD1 and VOD2")


# -----------------------------------
# wavelet

wvlt = False
if wvlt:
    analyze_wavelets(vod_ts, 'VOD1_anom_highbiomass')
    

# -----------------------------------
# recreate plot from the literature
"""
1. Yitong:
    - interval: 05-2022 to 11-2023
    - VOD1_anom
    - figsize=(5, 3)
    - daily average (red)
    - 95% percentile shaded area (light red)
    - linewidth 1.2
    - ylims=(0.4, 0.9)
2. Humphrey:
    - interval: 05-2023 to 12-2023
    - hourly data
    - figsize=(10, 5)
    - two curves: "raw" VOD1 (grey) and "processed" VOD1_anom (black), line width 0.8
    - ylims=(0.6,1)
3. Burns
    - interval: 2023, doy 210-310
    - hourly data
    - VOD1_anom
    - figsize=(5, 3)
    - ylims=(0, 1)
"""

authors = False
if authors:
    # only run, if data (time_subset) encompasses 2022-2024 data
    if not vod_ts.index.min() < pd.to_datetime('2022-05-01', utc=True) or vod_ts.index.max() > pd.to_datetime('2024-12-31', utc=True):
        raise ValueError("Data does not cover the required time range for author plots.")
    # Plot in different author styles
    plot_vod_by_author(vod_ts, 'yitong')
    plot_vod_by_author(vod_ts, 'humphrey')
    plot_vod_by_author(vod_ts, 'burns')