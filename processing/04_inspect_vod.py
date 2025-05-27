#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd

from definitions import DATA

# Matplotlib version
from processing.inspect_vod_funs import plot_vod_diurnal, plot_vod_fingerprint, plot_vod_scatter, plot_vod_timeseries, \
    read_vod_timeseries
from processing.vodreader import VODReader

"""
What happened with the missing data in the second half of 2024?
"""

# -----------------------------------

settings = {
    'station': 'MOz',
    'time_interval': ('2024-05-01', '2024-05-30'),
    'anomaly_type': 'phi_theta_sv'
}

# Create reader with automatic file selection
reader = VODReader(settings)

# Get the data
vod_ts = reader.get_data()

if vod_ts is None:
    raise ValueError("No VOD data found for the specified time interval.")

# -----------------------------------
# show all pandas columns

pd.set_option('display.max_columns', None)
vod_ts.describe()

# -----------------------------------
# time series plot
# Static matplotlib plot

figsize = (7, 5)

plot = True
if plot:
    plot_vod_timeseries(vod_ts, ['VOD1_anom'], figsize=figsize)
    # plot_vod_timeseries(vod_ts, ['VOD2', 'VOD2_anom'], title="VOD Time Series")

    # Interactive plotly plot
    # plot_vod_timeseries(vod_ts, ['VOD1_anom', 'VOD1_std'], interactive="interactive")
    
    # Save plot to file
    plot_vod_timeseries(vod_ts, ['VOD2_anom'], figsize=figsize)

# -----------------------------------
# diurnal plot
plot = True
if plot:
    # Create the 2x2 diurnal plot
    plot_vod_diurnal(vod_ts, show_std=False,
                     figsize=(8, 6),
                     title="VOD Diurnal Patterns",
                     filename="vod_diurnal_patterns.png")

# -----------------------------------
# fingerprint plot
plot = True
if plot:
    # plot_vod_fingerprint(vod_ts, 'VOD1', title="Fingerprint Plot of VOD1")
    plot_vod_fingerprint(vod_ts, 'VOD1_anom', title="Fingerprint Plot of VOD1 (anomaly)")
    plot_vod_fingerprint(vod_ts, 'VOD2', title="Fingerprint Plot of VOD2")
    plot_vod_fingerprint(vod_ts, 'VOD2_anom', title="Fingerprint Plot of VOD2 (anomaly)")

# -----------------------------------
# polarimetry
# Basic scatter plot colored by hour of day
plot = True
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
    
    # plot_vod_scatter(
    #     vod_ts,
    #     hue='hod',
    #     point_size=1,
    #     add_linear_fit=True,
    #     cmap=cmocean.cm.balance,
    #     figsize=(5,4),
    #     title='VOD Frequency Relationship',
    #     filename='vod_frequency_scatter.png'
    # )