"""
calc_vod calculates VOD according to specified pairing rules
"""
# ===========================================================
# ========================= imports =========================
import os
import time
import glob
import datetime
import numpy as np
import pandas as pd
import xarray as xr
import warnings
from gnssvod.io.preprocess import get_filelist
import pdb
from tqdm import tqdm

from processing.filepattern_finder import filter_files_by_date


#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#----------------- CALCULATING VOD --------------------
#--------------------------------------------------------------------------


def calc_vod(parquet, pairings, bands, recover_snr=False, n_workers=15, chunk_size='100MB'):
    """
    Calculate VOD using parallel processing with better handling of non-monotonic SV dimension.
    """
    import dask.dataframe as dd
    
    try:
        data = dd.read_parquet(parquet, engine='pyarrow', chunksize=chunk_size)
    except Exception as e:
        warnings.warn(f"Failed to read parquet file {parquet}: {e}")
        return None
    
    if len(pairings) > 1:
        raise ValueError("Only one pairing is allowed at a time. Please provide a single station pairing.")
    
    # Process each pairing using Dask
    icase = list(pairings.items())[0][1]
    try:
        # Extract reference and ground stations
        iref = data[data['Station'] == icase[0]]
        igrn = data[data['Station'] == icase[1]]
        
        # Merge on common indices
        idat = iref.merge(
            igrn,
            on=['Epoch', 'SV'],
            suffixes=['_ref', '_grn']
        )
        
        # Process each band
        for ivod in bands.items():
            ivars = np.intersect1d(data.columns.to_list(), ivod[1])
            
            # Apply calculations for each variable
            for ivar in ivars:
                # Get column names
                ref_name = f"{ivar}_ref"
                grn_name = f"{ivar}_grn"
                ele_name = f"Elevation_grn"
                
                # Calculate VOD
                idat[ivar] = -np.log(np.power(10, (idat[grn_name] - idat[ref_name]) / 10)) * \
                             np.cos(np.deg2rad(90 - idat[ele_name]))
            
            # Merge different bands
            idat[ivod[0]] = np.nan
            for ivar in ivars:
                idat[ivod[0]] = idat[ivod[0]].fillna(idat[ivar])
        
        if recover_snr:
            snr_cols = [f"S{band[3:]}_ref" for band in bands.keys()] + \
                       [f"S{band[3:]}_grn" for band in bands.keys()]
            selected_cols = list(bands.keys()) + snr_cols + ['SV', 'Azimuth_ref', 'Elevation_ref']
        else:
            selected_cols = list(bands.keys()) + ['Azimuth_ref', 'Elevation_ref']
        
        idat = idat[selected_cols].rename(columns={'Azimuth_ref': 'Azimuth', 'Elevation_ref': 'Elevation'})
    except Exception as e:
        warnings.warn(f"Error processing pairing {icase}: {e}")
        return None
    return idat