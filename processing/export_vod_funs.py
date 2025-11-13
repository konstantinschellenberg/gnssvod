#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from matplotlib import dates as mdates, pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from definitions import FIG
from processing.settings import *

def plot_hemi(vod, patches, var="VOD1_mean", type="vod", sbas_sats=None, show_sbas=False, **kwargs):
    
    # unfold kwargs
    figsize = kwargs.get('figsize', (6,6))
    transform = kwargs.get('transform', None)
    clim = kwargs.get('clim', [0.0, 1.0])
    angular_cutoff = kwargs.get('angular_cutoff', 0)
    title = kwargs.get('title', f"Hemispherical plot of {var} at station {station}")
    title = title if not show_sbas else f"{title}\nSBAS satellite location shown as black markers"

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
    # associate the mean values to the patches, join inner will drop patches with no data, making plotting slightly faster
    ipatches = pd.concat([patches, vod], join='inner', axis=1)
    if transform is not None:
        if transform == 'log10':
            ipatches[var] = np.log10(ipatches[var])
        else:
            raise ValueError(f"Unknown transform: {transform}. Supported: 'log10'.")
    
    hist = ipatches[var].dropna()
    if clim == "auto":
        vmin, vmax = hist.min(), hist.max()
    elif clim == "perc_975":
        vmin, vmax = np.percentile(hist, 2.5), np.percentile(hist, 97.5)
    else:
        vmin, vmax = clim
        
    if type == "vod":
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        ncolors = 256
        colors_neg = plt.get_cmap('Blues')(np.linspace(0, 1, ncolors//2))
        colors_pos = plt.get_cmap('hot')(np.linspace(0, 1, ncolors//2))
        newcolors = np.vstack((colors_neg, colors_pos))
        custom_cmap = LinearSegmentedColormap.from_list('blue_hot', newcolors)
        # Use the custom colormap and norm in the PatchCollection
    elif type == "std":
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        custom_cmap = plt.get_cmap('viridis')
    else:
        raise ValueError(f"Unknown type: {type}. Supported types: 'vod', 'std'.")
    
    pc = PatchCollection(ipatches.Patches, array=ipatches[var], cmap=custom_cmap, norm=norm,
                         edgecolor='face', linewidth=1)
    ax.add_collection(pc)
    
    if show_sbas:
        from processing.settings import SBAS_IDENT  # ensure SBAS_IDENT is available
        if sbas_sats is None:
            sbas_sats = list(SBAS_IDENT.keys())
        for sat in sbas_sats:
            sat_info = SBAS_IDENT.get(sat)
            if sat_info:
                azi = np.deg2rad(sat_info["Azimuth"])
                theta = 90 - sat_info["Elevation"]
                ax.scatter(azi, theta, marker='o', color='black', s=40, label=sat)
                if sat == "S33":
                    ax.annotate("S33", xy=(azi, theta), xytext=(azi - np.deg2rad(15), theta + 10), textcoords='data',
                                arrowprops=dict(arrowstyle='->', color='black'), color='black', fontsize=15, ha='right',
                                va='top',
                                bbox=dict(facecolor='white', boxstyle='round,pad=0.2', alpha=0.6))
    ax.set_rlim([0, 90-angular_cutoff])
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    
    ax.set_title(title, fontsize=14, pad=20)
    ax.grid(True, linestyle='--', linewidth=0.5)
    plt.colorbar(pc, ax=ax, location='bottom', shrink=.5, pad=0.05, label=var)
    plt.tight_layout()
    plt.savefig(FIG / f"hemi_{var}_{station}.png", dpi=300)
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
