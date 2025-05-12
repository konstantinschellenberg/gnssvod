#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gnssvod as gv
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import matplotlib.dates as mdates

from definitions import FIG, DATA, ROOT, get_repo_root, AUX, GROUND, TOWER


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


if __name__ == '__main__':
    # as xarray
    ds = xr.open_mfdataset(str(DATA / "gather" / "*.nc"), combine='nested', concat_dim='Epoch')
    
    # to dataframe
    df = ds.to_dataframe().dropna(how='all').reorder_levels(["Station", "Epoch", "SV"]).sort_index()
    
    # print all SV in the df
    print("SV in the df\n", df.index.get_level_values('SV').unique())
    SVs = df.index.get_level_values('SV').unique().sort_values()
    for prn in SVs:
        print(f"SV {prn} has {len(df.xs(prn, level='SV'))} rows")
    
    # -----------------------------------
    # get a subset of the data
    subset = df.xs('E03', level='SV').xs('MOz1_Grnd', level='Station')
    
    #print the perc nans in the cols
    print(subset.isna().mean() * 100)
    
    mySV = 'E03'
    mystation_name = 'MOz1_Grnd'
    snr_col = 'S6C'  # S5Q, S1C, S6C
    
    
    # Example usage
    plot_satellite_polar(df, mySV, mystation_name, figsize=(5,5), snr_col=snr_col)