#!/usr/bin/env python
# -*- coding: utf-8 -*-

from matplotlib import dates as mdates, pyplot as plt
from matplotlib.collections import PatchCollection
import numpy as np
import pandas
import pandas as pd

from definitions import DATA, FIG
from processing.settings import *

def create_metadata():
    """
    Create a dictionary containing all the settings used for VOD processing.

    Returns
    -------
    dict
        Dictionary containing metadata
    """
    import sys
    import datetime
    
    # Get all settings from the current module (settings.py)
    # Only include settings up to line 64 as requested
    metadata = {
        # General settings
        'station': station,
        'ground_station': ground_station,
        'tower_station': tower_station,
        'overwrite': overwrite,
        'output_results_locally': output_results_locally,
        'time_selection': time_selection,
        'year': year,
        'doy': doy,
        'dates_to_skip': dates_to_skip,
        
        # Preprocessing settings
        'unzipping_run': unzipping_run,
        'binex2rinex_run': binex2rinex_run,
        'one_dataset_run': one_dataset_run,
        'both_datasets_run': both_datasets_run,
        'binex2rinex_driver': binex2rinex_driver,
        'single_station_to_be_preprocessed': single_station_to_be_preprocessed,
        'save_orbit': save_orbit,
        
        # Gather settings
        'timeintervals_periods': timeintervals_periods,
        'timeintervals_freq': timeintervals_freq,
        'timeintervals_closed': timeintervals_closed,
        
        # VOD export settings
        'bands': bands,
        'angular_resolution': angular_resolution,
        'temporal_resolution': temporal_resolution,
        'agg_func': agg_func,
        'anomaly_type': anomaly_type,
        'angular_cutoff': angular_cutoff,
    }
    
    # make a unique hexadecimal identifier for the metadata
    metadata['metadata_id'] = f"{hash(frozenset(str(metadata.items()))):x}"
    metadata['script_name'] = sys.argv[0] if len(sys.argv) > 0 else 'unknown_script'
    metadata['creation_time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metadata['python_version'] = sys.version.split()[0]  # Get major.minor version
    
    return metadata


def save_metadata(metadata, file_path):
    """
    Save metadata to a JSON file.

    Parameters
    ----------
    metadata : dict
        Dictionary containing metadata
    file_path : str or Path
        File path to save metadata to (without extension)

    Returns
    -------
    path : Path
        Path to the saved metadata file
    """
    import json
    from pathlib import Path
    
    # Add .json extension
    metadata_path = Path(str(file_path).replace('.nc', '_metadata.json'))
    
    try:
        # Save metadata to JSON file
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4, default=str)
        
        print(f"Saved metadata to {metadata_path}")
        return metadata_path
    
    except Exception as e:
        print(f"Error saving metadata file {metadata_path}: {str(e)}")
        return None


def save_vod_timeseries(vod_ts, filename, overwrite=False):
    """
    Save VOD time series data to a NetCDF file and metadata to a JSON file.

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
        
        metadata = create_metadata()

        outname = f"{filename}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}_{metadata["metadata_id"]}"
        
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
            # Still save metadata even if not overwriting the data
            metadata = create_metadata()
            save_metadata(metadata, file_path)
            return file_path
        
        # Save to NetCDF
        ds.to_netcdf(file_path)
        
        # Save metadata
        save_metadata(metadata, file_path)
        
        print(f"Saved VOD time series to {file_path}")
        return file_path
    
    except Exception as e:
        print(f"Error saving file {file_path}: {str(e)}")
        return None


def plot_hemi(vod, patches, title=None):
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))
    # associate the mean values to the patches, join inner will drop patches with no data, making plotting slightly faster
    ipatches = pd.concat([patches, vod], join='inner', axis=1)
    # plotting with colored patches
    pc = PatchCollection(ipatches.Patches, array=ipatches["VOD1_mean"], edgecolor='face', linewidth=1)
    pc.set_clim([0.0, 1.0])
    ax.add_collection(pc)
    ax.set_rlim([0, 90-angular_cutoff])
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
    plt.savefig(FIG / f"vod_anomaly_{band}_{station}_{anomaly_type}.png", dpi=300)
    plt.show()
