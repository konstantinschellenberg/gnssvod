#!/usr/bin/env python
# -*- coding: utf-8 -*-

from processing.inspect_vod_funs import *
from processing.vodreader import VODReader
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
# subset time

if time_subset:
    # Convert time_subset to datetime if it's a string
    if isinstance(time_subset, str):
        time_subset = pd.to_datetime(time_subset, format='%Y-%m-%d', utc=True).tz_convert(tz)
    
    # Filter vod_ts based on the time_subset
    vod_ts = vod_ts[vod_ts.index >= time_subset[0]]
    vod_ts = vod_ts[vod_ts.index <= time_subset[1]]
    
    vod_algo = vod_algo[vod_algo.index >= time_subset[0]]
    vod_algo = vod_algo[vod_algo.index <= time_subset[1]]

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
    plot_vod_timeseries(vod_ts, ['VOD2_anom'], figsize=figsize)
    
    # compare algos
    plot_vod_timeseries(vod_algo, ['VOD1_anom_tp', 'VOD1_anom_tps'],
                        title="VOD Time Series (TPS Algorithm)")

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
# polarimetry
# Basic scatter plot colored by hour of day
plot = False
if plot:
    
    # With linear fit and custom settings
    plot_vod_scatter(
        vod_ts,
        hue='doy',
        point_size=1,
        add_linear_fit=True,
        cmap='plasma',
        figsize=(5,4),
        title='VOD Frequency Relationship',
        filename='vod_frequency_scatter.png'
    )
    
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

wvlt = True
if wvlt:
    analyze_wavelets(vod_ts, 'VOD1_anom')