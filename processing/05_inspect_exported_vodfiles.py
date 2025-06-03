#!/usr/bin/env python
# -*- coding: utf-8 -*-

from processing.inspect_vod_funs import *
from io.vodreader import VODReader
from processing.settings import *


# -----------------------------------
# Define the load mode
if load_mode == 'single_file':
    reader = VODReader(single_file_settings)
elif load_mode == 'multi_year':
    reader = VODReader(gatheryears=time_intervals)

# -----------------------------------
# Get long data
vod = reader.get_data(format='long')

if vod is None:
    raise ValueError("No VOD data found for the specified time interval.")

# -----------------------------------
# selecting only the 'tps' algorithm
vod_ts = vod[vod['algo'] == 'tps'].copy()
# get both algos to compare them
vod_algo = reader.get_data(format='wide').copy()

# -----------------------------------
# todo: filter for gnss_parameters!!

# -----------------------------------
# subset time

if time_subset:
    # Convert time_subset to datetime if it's a string
    if isinstance(time_subset, str):
        time_subset = pd.to_datetime(time_subset, format='%Y-%m-%d', utc=True).tz_convert(visualization_timezone)
    
    # Filter vod_ts based on the time_subset
    vod_ts = vod_ts[vod_ts.index >= time_subset[0]]
    vod_ts = vod_ts[vod_ts.index <= time_subset[1]]
    
    vod_algo = vod_algo[vod_algo.index >= time_subset[0]]
    vod_algo = vod_algo[vod_algo.index <= time_subset[1]]

# -----------------------------------
# -----------------------------------
# -----------------------------------
# time series plot
# Static matplotlib plot

figsize = (7, 5)

plot = False
if plot:
    plot_vod_timeseries(vod_ts, ['VOD1_anom'], figsize=figsize)
    # plot_vod_timeseries(vod_ts, ['VOD2', 'VOD2_anom'], title="VOD Time Series")

    # Interactive plotly plot
    # plot_vod_timeseries(vod_ts, ['VOD1_anom', 'VOD1_std'], interactive="interactive")
    
    # Save plot to file
    # plot_vod_timeseries(vod_ts, ['VOD2_anom'], figsize=figsize)
    
# -----------------------------------

compare_algos = False
if compare_algos:
    # compare algos
    kwargs = {
        'title': "VOD Time Series Comparison",
        'linewidth': 0.8,
        'figsize': (5, 3),
        'ylim': (-0.2, 1.3),
        'alpha': 0.4,
        'daily_average': True,
    }
    plot_vod_timeseries(vod_algo, variables=['VOD1_anom_tp', 'VOD1_anom_tps'],**kwargs)
    
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

authors = True

if authors:
    # only run, if data (time_subset) encompasses 2022-2024 data
    if not vod_ts.index.min() < pd.to_datetime('2022-05-01', utc=True) or vod_ts.index.max() > pd.to_datetime('2024-12-31', utc=True):
        raise ValueError("Data does not cover the required time range for author plots.")
    # Plot in different author styles
    plot_vod_by_author(vod_ts, 'yitong')
    plot_vod_by_author(vod_ts, 'humphrey')
    plot_vod_by_author(vod_ts, 'burns')

# -----------------------------------
# diurnal plot
plot = False
if plot:
    # Create the 2x2 diurnal plot
    # For the algorithm-suffixed dataset:
    
    # filter vod_algo (may-September)
    vod_algo_filtered = vod_algo[vod_algo['doy'].between(121, 273)]  # May to September (121 to 273)
    plot_vod_diurnal(vod_algo_filtered,
                     title="VOD Diurnal Patterns by Algorithm\n May-September (incl)",
                     filename="vod_diurnal_algo_comparison.png")
    
    # winter (November to March)
    vod_algo_filtered_winter = vod_algo[vod_algo['doy'].between(1, 120) | vod_algo['doy'].between(274, 365)]
    plot_vod_diurnal(vod_algo_filtered_winter,
                        title="VOD Diurnal Patterns by Algorithm\n November-March (incl)",
                        filename="vod_diurnal_algo_comparison_winter.png")

# -----------------------------------
# fingerprint plot
plot = False
if plot:
    plot_vod_fingerprint(vod_ts, 'VOD1', title="Comparing algorithms (1/3)\n No anomaly calculation\n\n band: (L1)")
    plot_vod_fingerprint(vod_algo, 'VOD1_anom_tp', title="Comparing algorithms (2/3)\n Anomaly calculated with Vincent's method\n (theta, psi)\n band: (L1)")
    plot_vod_fingerprint(vod_algo, 'VOD1_anom_tps', title="Comparing algorithms (3/3)\n Anomaly calculated with Konstantin's extension\n (theta, psi, sat)\n band: (L1)")
    
    # nsat
    plot_vod_fingerprint(vod_ts, 'S2_ref')

# -----------------------------------
# Two-variable scatter plot
plot = True
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
plot = False
if plot:
    plot_daily_diurnal_range(vod_ts, vars_to_plot=['VOD1_anom', 'VOD2_anom'],
                             title="Daily Diurnal Range of VOD1 and VOD2",
                             qq99=99.5,)
    
    # plot_daily_diurnal_range(vod_algo, vars_to_plot=['VOD1_anom_tp', 'VOD1_anom_tps'],
    #                          title="Daily Diurnal Range of VOD1 and VOD2")


# -----------------------------------
# wavelet

wvlt = False
if wvlt:
    analyze_wavelets(vod_ts, 'VOD1_anom')