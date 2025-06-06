#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import xarray as xr

from processing.export_vod_funs import plot_hemi


def ke_fun(vod, d, elevation):
    """
    vod: vegetation optical depth
    d: canopy height
    elevation: elevation angle in degrees
    ke: effective extinction coefficient
    """
    theta = 90 - elevation  # convert elevation to zenith angle
    pathlength = d / np.cos(np.deg2rad(theta))
    ke = vod / pathlength
    return ke, pathlength


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
    # make ke per band and add to the band_ids if make_ke is True
    make_ke = kwargs.get('make_ke', False)  # whether to calculate extinction coefficient
    if make_ke:
        canopy_height = kwargs.get('canopy_height', None)
        z0 = kwargs.get('z0', None)  # height of the ground receiver
        di = canopy_height - z0
        if canopy_height is None:
            raise ValueError("Canopy height must be provided if extinction coefficient is to be calculated.")
        # Calculate ke for each band and add to vod_with_cells
        for band in band_ids:
            vod_with_cells[f"{band}_ke"], _ = ke_fun(
                vod_with_cells[band],
                d=di,
                elevation=vod_with_cells["Elevation"]
            )
            # todo: add a vod_rect as ke*d to with tau values
        # append 'VOD_ke' to band_ids
        band_ids = [f"{band}_ke" for band in band_ids] + band_ids  # add ke bands to the list
        
    # -----------------------------------
    # clean df
    
    vod_with_cells = vod_with_cells.drop(columns=["Azimuth", "Elevation"], errors='ignore')
    
    # -----------------------------------
    # Method 1: Vincent's method (phi_theta) - short tp
    # Get average value per grid cell
    global vod_ts_1
    vod_avg = vod_with_cells.groupby(['CellID']).agg(['mean', 'std', 'count'])
    vod_avg.columns = ["_".join(x) for x in vod_avg.columns.to_flat_index()]
    
    # Count satellites per original epoch first
    sv_counts = vod_with_cells.groupby('Epoch').apply(lambda x: x.index.get_level_values('SV').nunique())
    
    # -----------------------------------
    
    # Calculate anomaly
    vod_anom = vod_with_cells.join(vod_avg, on='CellID')
    for band in band_ids:
        vod_anom[f"{band}_anom"] = vod_anom[band] - vod_anom[f"{band}_mean"]
    
    # Temporal aggregation
    vod_ts_1 = vod_anom.groupby(pd.Grouper(freq=f"{temporal_resolution}min", level='Epoch')).agg('mean')
    for band in band_ids:
        vod_ts_1[f"{band}_anom"] = vod_ts_1[f"{band}_anom"] + vod_ts_1[f"{band}"].mean()
    
    # -----------------------------------
    # NEW FEATURES
    # Calculate Ns(t): Average number of satellites per resampled epoch
    ns_t = sv_counts.groupby(pd.Grouper(freq=f"{temporal_resolution}min")).mean().rename('Ns_t')
    
    # ########
    # Calculate SD(Ns(t)): Standard deviation of the number of satellites per resampled epoch
    sd_ns_t = sv_counts.groupby(pd.Grouper(freq=f"{temporal_resolution}min")).std().rename('SD_Ns_t')
    
    # ########
    # Calculate C(t): Percentage of sky plots covered per time period
    # First, count total unique cells in the entire dataset
    total_possible_cells = vod_with_cells['CellID'].nunique()
    
    # cutoff already applied to hemi grid - doesn't need to be filtere here again :)
    # For each time window, calculate the percentage of sky covered
    cell_coverage = vod_with_cells.groupby(pd.Grouper(freq=f"{temporal_resolution}min", level='Epoch')).apply(
        lambda x: x['CellID'].nunique() / total_possible_cells * 100
    ).rename('C_t_perc')
    
    # ########
    # Calculate Ci(t): Binned percentage (by VOD percentiles) of sky plots covered per time period
    num_bins = kwargs.get('vod_percentile_bins', 5)
    
    for band in band_ids:
        # Get the mean VOD values for this band
        band_means = vod_avg[f"{band}_mean"]
        
        # Create bins based on VOD percentiles
        try:
            # Try to create equal-sized bins by percentile
            bins = pd.qcut(band_means, num_bins, labels=False, duplicates='drop')
        except ValueError:
            # Fall back to equal-width bins if not enough unique values
            bins = pd.cut(band_means, min(num_bins, len(band_means.unique())), labels=False)
        
        plot = False
        if plot:
            # Calculate actual bin edges for visualization
            if isinstance(bins, pd.Series):
                # Get the unique bin values and their corresponding data
                bin_groups = {}
                for bin_num in range(num_bins):
                    bin_groups[bin_num] = band_means[bins == bin_num]
                
                # Calculate min and max for each bin to use as edges
                bin_edges = [bin_groups[i].min() for i in sorted(bin_groups.keys())]
                bin_edges.append(bin_groups[max(bin_groups.keys())].max())
                
                from matplotlib import pyplot as plt
                plt.figure(figsize=(5, 3))
                band_means.hist(bins=30, alpha=0.5, label='VOD Means')
                for edge in bin_edges:
                    plt.axvline(edge, color='red', linestyle='--')
                plt.axvline(bin_edges[0], color='red', linestyle='--', label='Bin Edges')  # Add label only once
                plt.title(f"VOD Means Histogram with Bins for {band}")
                plt.xlabel('VOD Mean Values')
                plt.ylabel('Frequency')
                plt.legend()
                plt.show()
        
        # Create a mapping from CellID to bin
        cell_to_bin = pd.Series(bins.values, index=band_means.index, name='bin')
        
        # Count total cells in each bin
        total_cells_per_bin = cell_to_bin.value_counts().sort_index()
        
        # Create a temporary DataFrame with just the necessary data
        temp_df = pd.DataFrame({
            'Epoch': vod_with_cells.index.get_level_values('Epoch'),
            'CellID': vod_with_cells['CellID']
        }).drop_duplicates()
        
        # Add bin information based on CellID
        temp_df = temp_df.join(cell_to_bin, on='CellID')
        
        # Group by time window and bin, count unique cells
        bin_counts = temp_df.groupby([
            pd.Grouper(key='Epoch', freq=f"{temporal_resolution}min"),
            'bin'
        ])['CellID'].nunique().unstack(fill_value=0)
        
        # Calculate percentage of each bin covered relative to total current coverage
        bin_coverage = pd.DataFrame(index=bin_counts.index)
        # Calculate total cells covered at each time step across all bins
        total_current_coverage = bin_counts.sum(axis=1)
        for bin_idx in range(num_bins):
            if bin_idx in bin_counts.columns and bin_idx in total_cells_per_bin.index:
                # Calculate relative coverage (percentage of current total coverage this bin represents)
                bin_coverage[f"Ci_t_{band}_bin{bin_idx}_pct"] = (
                        bin_counts[bin_idx] / total_current_coverage.replace(0, np.nan) * 100
                )
        # Join the results to vod_ts_1
        vod_ts_1 = vod_ts_1.join(bin_coverage)
    
    # -----------------------------------
    # join new features to the aggregated time series
    vod_ts_1 = vod_ts_1.join(ns_t)
    vod_ts_1 = vod_ts_1.join(sd_ns_t)
    vod_ts_1 = vod_ts_1.join(cell_coverage)
    
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
