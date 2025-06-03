#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import xarray as xr

def calculate_anomaly(vod_with_cells, band_ids, temporal_resolution, **kwargs):
    """
    Calculate VOD anomalies using both methods.
    
    Following standard features:
    - VOD and VOD anomalies
    - Different bands (e.g., VOD1, VOD2)
    - C/N0 output
    - Temporal aggregation to <temporal_resolution> minutes
    
    New features:
    - Ns(t): Average number of satellites per resampled epoch
    - SD(Ns(t)): Standard deviation of the number of satellites per resampled epoch
    - C(t): Percentage sky plots covered per time period
    - Ci(t): Binned percentage (by VOD percentiles) of sky plots covered per time period
    
    Aggregation methods:
    - Method 1: Vincent's method (phi_theta) for short time periods
    - Method 2: Konstantin's extension (phi_theta_sv) for short time periods
    
    
    TODO: Make this script more efficient!
    - save intermediate results and remove from memory (vod_ts_1)

    Parameters
    ----------
    vod_with_cells : pandas.DataFrame
        VOD data with cell IDs
    band_ids : list
        List of band identifiers to calculate anomalies for (e.g., ['VOD1', 'VOD2'])
    temporal_resolution : int
        Temporal resolution in minutes

    Returns
    -------
    tuple
        vod_ts_combined : pandas.DataFrame
            Combined time series of VOD anomalies from both methods
        vod_avg : pandas.DataFrame
            Average VOD values per grid cell
    """

    
    # -----------------------------------
    # Method 1: Vincent's method (phi_theta) - short tp
    # Get average value per grid cell
    vod_avg = vod_with_cells.groupby(['CellID']).agg(['mean', 'std', 'count'])
    vod_avg.columns = ["_".join(x) for x in vod_avg.columns.to_flat_index()]
    
    # Calculate anomaly
    vod_anom = vod_with_cells.join(vod_avg, on='CellID')
    for band in band_ids:
        vod_anom[f"{band}_anom"] = vod_anom[band] - vod_anom[f"{band}_mean"]
    
    # Temporal aggregation
    vod_ts_1 = vod_anom.groupby(pd.Grouper(freq=f"{temporal_resolution}min", level='Epoch')).agg('mean')
    for band in band_ids:
        vod_ts_1[f"{band}_anom"] = vod_ts_1[f"{band}_anom"] + vod_ts_1[f"{band}"].mean()
    
    # -----------------------------------
    # Method 2: Konstantin's extension (phi_theta_sv) - short tps
    vod_ts_all = []
    
    # Process each satellite vehicle separately
    for sv in vod_with_cells.index.get_level_values('SV').unique():
        # Extract data for this SV only
        vod_sv = vod_with_cells.xs(sv, level='SV')
        
        # todo: revisit this condition
        # Skip if there's not enough data for this SV
        # if len(vod_sv) < 100:
        #     print(f"Skipping SV {sv} due to insufficient data.")
        #     continue
        
        # Calculate average values per grid cell for this specific SV
        vod_avg_sv = vod_sv.groupby(['CellID']).agg(['mean', 'std', 'count'])
        vod_avg_sv.columns = ["_".join(x) for x in vod_avg_sv.columns.to_flat_index()]
        
        # Join the cell averages back to the original data
        vod_anom_sv = vod_sv.join(vod_avg_sv, on='CellID')
        
        # Calculate anomalies for each band
        for band in band_ids:
            vod_anom_sv[f"{band}_anom"] = vod_anom_sv[band] - vod_anom_sv[f"{band}_mean"]
        
        # Add SV as a column before appending
        vod_anom_sv['SV'] = sv
        vod_anom_sv = vod_anom_sv.reset_index()
        vod_ts_all.append(vod_anom_sv)
    
    # Combine all SV results
    if vod_ts_all:
        vod_ts_svs = pd.concat(vod_ts_all).set_index(['Epoch', 'SV'])
        
        # Temporal aggregation
        vod_ts_2 = vod_ts_svs.groupby(pd.Grouper(freq=f"{temporal_resolution}min", level='Epoch')).agg('mean')
        
        # Add back the mean
        for band in band_ids:
            vod_ts_2[f"{band}_anom"] = vod_ts_2[f"{band}_anom"] + vod_ts_2[f"{band}"].mean()
    else:
        vod_ts_2 = pd.DataFrame()
    
    # -----------------------------------
    # Prepare the final results
    # Combine both methods
    if not vod_ts_1.empty:
        vod_ts_1['algo'] = 'tp'  # Vincent's method
    if not vod_ts_2.empty:
        vod_ts_2['algo'] = 'tps'  # Konstantin's extension
    
    # Concatenate results
    vod_ts_combined = pd.concat([vod_ts_1, vod_ts_2], axis=0)
    
    # Drop CellID if present
    vod_ts_combined = vod_ts_combined.drop(columns=['CellID'], errors='ignore')
    
    return (vod_ts_combined, vod_avg)