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
from processing.settings import *


def save_vod_timeseries(vod_ts, filename, overwrite=False):
    """
    Save VOD time series data to a NetCDF file.

    Parameters
    ----------
    vod_ts : pandas.DataFrame
        VOD time series data with 'Epoch' in the index
    filename : str
        Filename to save (without extension)
    overwrite : bool, optional
        Whether to overwrite existing files

    Returns
    -------
    path : Path
        Path to the saved file or None if save failed
    """
    import os
    
    # Set default directory if not specified
    directory = DATA / "timeseries"
    directory.mkdir(parents=True, exist_ok=True)
    
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    try:
        # Create a copy of the dataframe to avoid modifying the original
        df = vod_ts.copy()
        
        # Reset index to get 'Epoch' as a column
        df = df.reset_index()
        
        # Rename 'Epoch' to 'datetime'
        df = df.rename(columns={'Epoch': 'datetime'})
        
        # Set 'datetime' as the index again
        df = df.set_index('datetime')
        
        # Generate output filename based on the date range
        start = df.index.min()
        end = df.index.max()
        outname = f"{filename}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}"
        
        # Add .nc extension if not already present
        if not outname.endswith('.nc'):
            outname = f"{outname}.nc"
            
        # Full file path
        file_path = directory / outname
        
        # Convert to xarray dataset for NetCDF export
        ds = df.to_xarray()
        
        # Check if file exists and overwrite flag
        if file_path.exists() and not overwrite:
            print(f"File {file_path} already exists. Pass overwrite=True to overwrite.")
            return file_path
        
        # Save to NetCDF
        ds.to_netcdf(file_path)
        
        print(f"Saved VOD time series to {file_path}")
        return file_path
    
    except Exception as e:
        print(f"Error saving file {file_path}: {str(e)}")
        return None
    
def plot_hemi(vod, patches, title=None):
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    # associate the mean values to the patches, join inner will drop patches with no data, making plotting slightly faster
    ipatches = pd.concat([patches, vod], join='inner', axis=1)
    # plotting with colored patches
    pc = PatchCollection(ipatches.Patches, array=ipatches["VOD1_mean"], edgecolor='face', linewidth=1)
    pc.set_clim([-0.1, 0.5])
    ax.add_collection(pc)
    ax.set_rlim([0, 90])
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    if title:
        ax.set_title(title)
    
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
    for band in bands:
        # plot each measurement and color by signal to noise ratio
        ax.plot(vod_ts.index.get_level_values('Epoch'), vod_ts[f"{band}_anom"], label=f"{band} anomaly")
        ax.plot(vod_ts.index.get_level_values('Epoch'), vod_ts[f"{band}"], label=f"{band}")
    myFmt = mdates.DateFormatter('%d-%b')
    ax.xaxis.set_major_formatter(myFmt)
    ax.set_ylabel('GNSS-VOD (L1)')
    ax.legend()
    plt.title(f'GNSS-VOD anomaly at {station} {year}-{doy}')
    plt.savefig(FIG / f"vod_anomaly_{band}_{station}.png", dpi=300)
    plt.show()
    
# -----------------------------------
"""
Main Features:
- Calculate VOD
- Export to netCDF

Viz:
- Plot 1 sat hemi
- Plot SNR ground/tower hemi
- Plot SNR ground/tower time series
- Plot VOD time series

"""
# -----------------------------------
band_ids = list(bands.keys())
# -----------------------------------

pattern = str(DATA / "gather" / "*.nc")
# define how to associate stations together. Always put reference station first.
pairings = {station:(tower_station, ground_station)}
# define if some observables with similar frequencies should be combined together. In the future, this should be replaced by the selection of frequency bands.

vod = gv.calc_vod(pattern, pairings, bands)[station]

# print the percentage of NaN values per column
print("NaN values in VOD:")
print(vod.isna().mean() * 100)

# -----------------------------------
# hemi grid

# todo: detach this into anomaly_type == "phi_theta"
# intialize hemispheric grid
hemi = gv.hemibuild(angular_resolution)
# get patches for plotting later
patches = hemi.patches()
# classify vod into grid cells, drop azimuth and elevation afterwards as we don't need it anymore
vod = hemi.add_CellID(vod).drop(columns=['Azimuth','Elevation'])
# get average value per grid cell
vod_avg = vod.groupby(['CellID']).agg(['mean', 'std', 'count'])
# flatten the columns
vod_avg.columns = ["_".join(x) for x in vod_avg.columns.to_flat_index()]

# plot hemi
if plot:
    plot_hemi(vod_avg, patches, title=f"VOD {station} {year}-{doy}")

# -----------------------------------
# calculate anomaly

print("Anomaly calulation type", anomaly_type)

if anomaly_type == "phi_theta":
    vod_anom = vod.join(vod_avg, on='CellID')
    for band in band_ids:
        vod_anom[f"{band}_anom"] = vod_anom[band] - vod_anom[f"{band}_mean"]
    vod_ts = vod_anom.groupby(pd.Grouper(freq=f"{temporal_resolution}min", level='Epoch')).mean()
    for band in band_ids:
        vod_ts[f"{band}_anom"] = vod_ts[f"{band}_anom"] + vod_ts[f"{band}"].mean()
    if plot:
        plot_anomaly(vod_ts, band_ids, figsize=(6, 4))
elif anomaly_type == "phi_theta_sv":
    raise NotImplementedError("phi_theta_sv is not implemented yet")
else:
    raise ValueError(f"Unknown anomaly type: {anomaly_type}")

# -----------------------------------
# Save VOD time series to NetCDF file
save_vod_timeseries(
    vod_ts,
    f"vod_timeseries_{station}",
    overwrite=overwrite
)

# -----------------------------------
# legacy
# # as xarray
# ds = xr.open_mfdataset(str(DATA / "gather" / "*.nc"), combine='nested', concat_dim='Epoch')
#
# # to dataframe
# df = ds.to_dataframe().dropna(how='all').reorder_levels(["Station", "Epoch", "SV"]).sort_index()
#
# # print all SV in the df
# print("SV in the df\n", df.index.get_level_values('SV').unique())
# SVs = df.index.get_level_values('SV').unique().sort_values()
# for prn in SVs:
#     print(f"SV {prn} has {len(df.xs(prn, level='SV'))} rows")
#
# # -----------------------------------
# # get a subset of the data
# subset = df.xs('E03', level='SV').xs('MOz1_Grnd', level='Station')
#
# #print the perc nans in the cols
# print(subset.isna().mean() * 100)
#
# mySV = 'E03'
# mystation_name = 'MOz1_Grnd'
# snr_col = 'S1'  # S5Q, S1C, S6C
#
#
# # Example usage
# plot_satellite_polar(df, mySV, mystation_name, figsize=(5,5), snr_col=snr_col)