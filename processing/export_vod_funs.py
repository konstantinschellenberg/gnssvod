#!/usr/bin/env python
# -*- coding: utf-8 -*-

from matplotlib import dates as mdates, pyplot as plt
from matplotlib.collections import PatchCollection
import numpy as np
import pandas
import pandas as pd

from definitions import DATA, FIG
from processing.settings import *

def plot_hemi(vod, patches, var="VOD1_mean", title=None, **kwargs):
    figsize = kwargs.get('figsize', (5,5))
    transform = kwargs.get('transform', None)
    clim = kwargs.get('clim', [0.0, 1.0])
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
    # associate the mean values to the patches, join inner will drop patches with no data, making plotting slightly faster
    ipatches = pd.concat([patches, vod], join='inner', axis=1)
    # if transform is specified, apply it to the variable
    if transform is not None:
        if transform == 'log10':
            ipatches[var] = np.log10(ipatches[var])
        else:
            raise ValueError(f"Unknown transform: {transform}. Supported: 'log10'.")
    # plotting with colored patches
    pc = PatchCollection(ipatches.Patches, array=ipatches[var], edgecolor='face', linewidth=1)
    if clim != "auto":
        pc.set_clim(clim)
    ax.add_collection(pc)
    ax.set_rlim([0, 90-angular_cutoff])
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    if title:
        ax.set_title(title)
    # plot grid
    ax.grid(True, linestyle='--', linewidth=0.5)
    plt.colorbar(pc, ax=ax, location='bottom', shrink=.5, pad=0.05, label='GNSS-VOD')
    plt.tight_layout()
    plt.savefig(FIG / f"hemi_vod_{station}.png", dpi=300)
    plt.show()


def plot_satellite_polar(df, sv, station_name, snr_col='S7Q', figsize=(10, 10), show=True):
    """
    Create a polar plot of satellite data for a specific satellite and station.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the GNSS data with MultiIndex (Station, Epoch, SV)
    sv : str
        Satellite vehicle ID (e.g., 'E03')
    station_name : str
        Name of the station (e.g., 'MOz1_Grnd')
    snr_col : str, optional
        Column name for the signal-to-noise ratio to use for coloring
    figsize : tuple, optional
        Figure size (width, height) in inches
    show : bool, optional
        Whether to call plt.show() or just return the figure and axes

    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    # initialize figure with polar axes
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
    
    # subset the dataset
    subdf = df.xs(sv, level='SV').xs(station_name, level='Station')
    
    # polar plots need a radius and theta direction in radians
    radius = 90 - subdf.Elevation
    theta = np.deg2rad(subdf.Azimuth)
    
    # plot each measurement and color by signal to noise ratio
    hs = ax.scatter(theta, radius, c=subdf[snr_col])
    ax.set_rlim([0, 90])
    ax.set_theta_zero_location("N")
    plt.colorbar(hs, shrink=.5, label=f'SNR ({snr_col})')
    plt.title(station_name)
    
    if show:
        plt.show()


def plot_anomaly(vod_ts, bands, **kwargs):
    figsize = kwargs.get('figsize', (10, 5))
    fig, ax = plt.subplots(1, figsize=figsize)
    title = kwargs.get('title', 'GNSS-VOD Anomaly')
    for band in bands:
        # plot each measurement and color by signal to noise ratio
        ax.plot(vod_ts.index.get_level_values('Epoch'), vod_ts[f"{band}_anom"], label=f"{band} anomaly")
        ax.plot(vod_ts.index.get_level_values('Epoch'), vod_ts[f"{band}"], label=f"{band}")
    myFmt = mdates.DateFormatter('%d-%b')
    ax.xaxis.set_major_formatter(myFmt)
    ax.set_ylabel('GNSS-VOD (L1)')
    ax.legend()
    plt.title(title)
    plt.savefig(FIG / f"vod_anomaly_{band}_{station}.png", dpi=300)
    plt.show()
