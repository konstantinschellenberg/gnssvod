#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import xarray as xr

from gnssvod import hemibuild
from processing.export_vod_funs import plot_hemi


def vod_fun(grn, ref, ele):
    return -np.log(np.power(10, (grn - ref) / 10)) * np.cos(np.deg2rad(90 - ele))

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


def calculate_extinction_coefficient(vod_with_cells, band_ids, canopy_height=None, z0=None):
    """
    Calculate extinction coefficient (ke) for each band in the VOD dataset.

    Parameters
    ----------
    vod_with_cells : pandas.DataFrame
        VOD data with Elevation information
    band_ids : list
        List of band identifiers (e.g., ['VOD1', 'VOD2'])
    canopy_height : float, optional
        Height of the canopy in meters
    z0 : float, optional
        Height of the ground receiver in meters

    Returns
    -------
    tuple
        vod_with_cells : pandas.DataFrame
            Input DataFrame with added extinction coefficient columns
        band_ids : list
            Updated list of band identifiers including ke bands
    """
    if canopy_height is None:
        raise ValueError("Canopy height must be provided if extinction coefficient is to be calculated.")
    if z0 is None:
        raise ValueError("Ground receiver height (z0) must be provided if extinction coefficient is to be calculated.")
    
    di = canopy_height - z0
    
    # Calculate ke for each band and add to vod_with_cells
    for band in band_ids:
        vod_with_cells[f"{band}_ke"], _ = ke_fun(
            vod_with_cells[band],
            d=di,
            elevation=vod_with_cells["Elevation"]
        )
        # todo: add a vod_rect as ke*d to with tau values
    
    # append 'VOD_ke' to band_ids
    updated_band_ids = [f"{band}_ke" for band in band_ids] + band_ids  # add ke bands to the list
    
    return vod_with_cells, updated_band_ids

def create_aggregation_dict(vod_with_cells, band_ids, agg_fun_ts='mean'):
    """
    Create an aggregation dictionary for groupby operations based on column types.

    Parameters
    ----------
    vod_with_cells : pandas.DataFrame
        DataFrame containing VOD data
    band_ids : list
        List of band identifiers

    Returns
    -------
    dict
        Dictionary with column names as keys and aggregation operations as values
    """
    agg_dict = {}
    # Add base band_ids (without suffixes) for mean, std, count
    for col in band_ids:
        if col in vod_with_cells.columns:
            if col.endswith('_ke'):
                agg_dict[col] = ['mean', 'std']
            else:
                agg_dict[col] = [agg_fun_ts, 'std', 'count']
    # Add _ref and _grn columns for mean, std only
    ref_grn_cols = [col for col in vod_with_cells.columns if "_ref" in col or "_grn" in col]
    for col in ref_grn_cols:
        agg_dict[col] = [agg_fun_ts, 'std']
    # Keep Azimuth and Elevation if they exist
    for col in ["Azimuth", "Elevation"]:
        if col in vod_with_cells.columns:
            agg_dict[col] = agg_fun_ts
    return agg_dict


def calculate_binned_sky_coverage(vod_with_cells, vod_avg, band_ids, temporal_resolution, num_bins=5,
                                  plot_bins=False, suffix=None):
    """
    Calculate Ci(t): Binned percentage of sky plots covered per time period, based on VOD percentiles.

    Parameters
    ----------
    df : pandas.DataFrame
        Time series DataFrame to be augmented with bin coverage information
    vod_with_cells : pandas.DataFrame
        VOD data with cell IDs
    vod_avg : pandas.DataFrame
        Average VOD values per grid cell
    band_ids : list
        List of band identifiers to calculate bin coverage for
    temporal_resolution : int
        Temporal resolution in minutes
    num_bins : int, default=5
        Number of percentile bins to create
    plot_bins : bool, default=False
        Whether to plot the bin histograms for visualization

    Returns
    -------
    pandas.DataFrame
        Time series DataFrame with added bin coverage columns
    """
    final = pd.DataFrame()
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
        
        if plot_bins:
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
        final.loc[:, f"Ci_t_{band}_bin*_pct"] = bin_coverage
        
    return final

def calculate_sv_specific_anomalies(vod_with_cells, band_ids, temporal_resolution, suffix="", **kwargs):
    """
    Calculate satellite vehicle specific VOD anomalies (Method 2: Konstantin's extension).

    Parameters
    ----------
    vod_with_cells : pandas.DataFrame
        VOD data with cell IDs
    band_ids : list
        List of band identifiers to calculate anomalies for
    temporal_resolution : int
        Temporal resolution in minutes

    Returns
    -------
    pandas.DataFrame
        Time series of VOD anomalies calculated using the SV-specific method
    """
    agg_fun_ts = kwargs.get('agg_fun_ts', 'mean')  # aggregation function for time series
    vod_ts_all = []
    suffix = f"_{suffix}" if suffix else ""
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
        vod_ts_2 = vod_ts_svs.groupby(pd.Grouper(freq=f"{temporal_resolution}min", level='Epoch')).agg(agg_fun_ts)
        
        # Add back the mean
        for band in band_ids:
            """
            Remove adding mean VOD here, as it is not needed in the final output. See method 1
            """
            vod_ts_2[f"{band}_anom{suffix}"] = vod_ts_2[f"{band}_anom"]
    else:
        vod_ts_2 = pd.DataFrame()
    
    # only return _anom cols
    return vod_ts_2[[f"{band}_anom{suffix}" for band in band_ids]]


def calculate_biomass_binned_anomalies(vod, vod_avg, band_ids, temporal_resolution, con=None, biomass_bins=5, **kwargs):
    """
    Calculate VOD anomalies for cells grouped by biomass (VOD) bins and filtered by constellation.
    Bins go from low to high biomass

    Parameters
    ----------
    vod : pandas.DataFrame
        VOD data with cell IDs
    vod_avg : pandas.DataFrame
        Average VOD values per grid cell
    band_ids : list
        List of band identifiers to calculate anomalies for
    temporal_resolution : int
        Temporal resolution in minutes
    con : list or None, default=None
        List of constellation names to include (e.g., ['GPS', 'Galileo'])
    biomass_bins : int, default=5
        Number of biomass bins to create
    **kwargs : dict
        Additional keyword arguments

    Returns
    -------
    pandas.DataFrame
        Time series of VOD anomalies for each biomass bin with Epoch as index
    """
    agg_fun_ts = kwargs.get('agg_fun_ts', 'mean')
    suffix = kwargs.get('suffix', '')
    if suffix and not suffix.startswith('_'):
        suffix = f"_{suffix}"
    
    # Constellation mapping
    constellation_ident = {
        'G': 'GPS',
        'R': 'GLONASS',
        'E': 'Galileo',
        'C': 'BeiDou',
        'S': 'SBAS'
    }
    
    # Filter by constellation if specified
    if con is not None:
        con_prefixes = []
        for c in con:
            for prefix, name in constellation_ident.items():
                if name.upper() == c.upper():
                    con_prefixes.append(prefix)
        
        if not con_prefixes:
            raise ValueError(f"No valid constellations found in {con}")
        
        # Filter VOD data
        sv_filter = vod.index.get_level_values('SV').str[0].isin(con_prefixes)
        vod_filtered = vod[sv_filter]
        
        # Print constellation filtering info
        selected_cons = [constellation_ident[prefix] for prefix in sorted(set(con_prefixes))]
        print(f"Filtering data for constellations: {', '.join(selected_cons)}")
        print(
            f"Selected {len(vod_filtered)} out of {len(vod)} observations ({len(vod_filtered) / len(vod) * 100:.1f}%)")
    else:
        vod_filtered = vod
    
    # Create biomass bins based on VOD averages for the reference band
    reference_band = band_ids[0]  # Use first band for biomass binning
    band_means = vod_avg[f"{reference_band}_mean"]
    
    # Create bins based on VOD percentiles
    try:
        bins = pd.qcut(band_means, biomass_bins, labels=False, duplicates='drop')
    except ValueError:
        actual_bins = min(biomass_bins, len(band_means.unique()))
        bins = pd.cut(band_means, actual_bins, labels=False)
        print(f"Warning: Not enough unique values for {biomass_bins} bins. Using {actual_bins} bins instead.")
    
    # Create a mapping from CellID to bin
    cell_to_bin = pd.Series(bins.values, index=band_means.index, name='biomass_bin')
    
    # Add bin information to VOD data
    vod_with_bins = vod_filtered.join(cell_to_bin, on='CellID')
    
    # Initialize results DataFrame
    combined_results = pd.DataFrame()
    
    # Process each band
    for band in band_ids:
        # Join band means to the data
        vod_with_means = vod_with_bins.join(vod_avg[[f"{band}_mean"]], on='CellID')
        
        # Calculate anomalies for each bin
        for bin_num in range(biomass_bins):
            # Filter data for this bin
            bin_data = vod_with_means[vod_with_means['biomass_bin'] == bin_num]
            
            if len(bin_data) < 10:  # Skip bins with very little data
                print(f"Skipping bin {bin_num} for {band} due to insufficient data ({len(bin_data)} observations)")
                continue
            
            # Calculate anomalies
            bin_data[f"{band}_anom"] = bin_data[band] - bin_data[f"{band}_mean"]
            
            # Temporal aggregation
            bin_ts = bin_data.groupby(pd.Grouper(freq=f"{temporal_resolution}min", level='Epoch'))[f"{band}_anom"].agg(
                agg_fun_ts)
            
            # Add to results with appropriate column name
            combined_results[f"{band}_anom_bin{bin_num}{suffix}"] = bin_ts
    
    return combined_results

def calculate_anomaly(vod, band_ids, temporal_resolution, **kwargs):
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
    vod : pandas.DataFrame
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
    # -------------------from dask.distributed import Client----------------
    # make ke per band and add to the band_ids if make_ke is True
    make_ke = kwargs.get('make_ke', False)  # whether to calculate extinction coefficient
    agg_fun_ts = kwargs.get('agg_fun_ts', 'mean')  # aggregation function for time series
    biomass_bins = kwargs.get('biomass_bins', 5)  # bins for biomass if needed
    
    if not isinstance(band_ids, list):
        raise ValueError("band_ids must be a list of band identifiers.")
    
    # -----------------------------------
    # calculate extinction coefficient if requested
    if make_ke:
        vod, band_ids = calculate_extinction_coefficient(
            vod,
            band_ids,
            canopy_height=kwargs.get('canopy_height', None),
            z0=kwargs.get('z0', None)
        )
    
    # -----------------------------------
    # Get average value per grid cell
    agg_dict = create_aggregation_dict(vod, band_ids)
    
    # Group by CellID and apply the specific aggregations
    vod_avg = vod.groupby(['CellID']).agg(agg_dict)
    
    # Flatten the MultiIndex for Azimuth and Elevation (rename mean to '')
    vod_avg.columns = ["_".join(x) for x in vod_avg.columns.to_flat_index()]
    
    # remove suffixes for Azimuth and Elevation
    vod_avg = vod_avg.rename(columns={
        'Azimuth_mean': 'Azimuth',
        'Elevation_mean': 'Elevation'
    })
    # Count satellites per original epoch first
    sv_counts = vod.reset_index().groupby('Epoch')['SV'].nunique()
    
    # -----------------------------------
    # Method 1: Vincent's method (phi_theta) - short tp
    # Calculate anomaly
    
    band_ids_tp = ["VOD1"]
    vod_drop = vod.drop(columns=['Azimuth', 'Elevation'], errors='ignore')
    vod_anom = vod_drop.join(vod_avg, on='CellID')[band_ids_tp + [f"{band}_mean" for band in band_ids_tp]]
    for band in band_ids:
        if band == "VOD1":
            vod_anom[f"{band}_anom_tp"] = vod_anom[band] - vod_anom[f"{band}_mean"]
    vod_anom = vod_anom[[f"{band}_anom_tp" for band in band_ids_tp] + band_ids_tp]
    
    """
    MAJOR CHANGE: ADD MEAN VOD IN A LATER SCIPRT:
    
    Background:
    - GNSS time series is too large to be processed at once (memory overflow)
    - Thus, VOD mean cannot be calculated for the entire dataset but only for chunks, which is wrong.
    - Solution: Calculate mean VOD later and VOD_anom = VOD_mean + VOD_anom
    
    """
    # Temporal aggregation
    vod_ts_1 = vod_anom.groupby(pd.Grouper(freq=f"{temporal_resolution}min", level='Epoch')).agg(agg_fun_ts)
    # -----------------------------------
    # Method 2: Konstantin's extension (phi_theta_sv) - short tps
    vod_ts_2 = calculate_sv_specific_anomalies(vod, band_ids, temporal_resolution, **kwargs)
  
    """
    Additional constellation combination:
    1. GPS (highest revisit of similar locations)
    2. GPS + GALILEO (best constellations)
    
    constellation_ident=
    {'G??': 'GPS',
     'R??': 'GLONASS',
     'E??': 'Galileo',
     'C??': 'BeiDou',
     'S??': 'SBAS'}
    """
    
    # filter where SV index starts with 'G' (GPS)
    vod_con1 = vod[vod.index.get_level_values('SV').str.startswith('G')]
    vod_ts_con1 = calculate_sv_specific_anomalies(vod_con1, band_ids, temporal_resolution, suffix="gps", **kwargs)
    vod_con2 = vod[vod.index.get_level_values('SV').str.startswith(('G', 'E'))]  # GPS + GALILEO
    vod_ts_con2 = calculate_sv_specific_anomalies(vod_con2, band_ids, temporal_resolution, suffix="gps+gal", **kwargs)
  
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
    total_possible_cells = vod['CellID'].nunique()
    
    # cutoff already applied to hemi grid - doesn't need to be filtere here again :)
    # For each time window, calculate the percentage of sky covered
    cell_coverage = vod.groupby(pd.Grouper(freq=f"{temporal_resolution}min", level='Epoch')).apply(
        lambda x: x['CellID'].nunique() / total_possible_cells * 100
    ).rename('C_t_perc')
    
    # #######
    # Calculate Ci(t): Binned percentage (by VOD percentiles) of sky plots covered per time period
    num_bins = kwargs.get('vod_percentile_bins', 5)
    # vod_ts_1 = calculate_binned_sky_coverage(vod_with_cells, vod_avg, band_ids, temporal_resolution,
    #                                             num_bins=num_bins, plot_bins=False, suffix="tp")
    # vod_ts_2 = calculate_binned_sky_coverage(vod_with_cells, vod_avg, band_ids, temporal_resolution,
    #                                          num_bins=num_bins, plot_bins=False, suffix="tps")
    #
    
    # -----------------------------------
    # extract SBAS
    
    # SBAS VOD was not calculated because azimuth and elevation were missing
    vod_sbas = vod[vod.index.get_level_values('SV').str.startswith('S')]

    for band in band_ids:
        freq = band[-1]  # get frequency from band name
        if "ke" in band:
            continue
        grn = vod_sbas[f"S{freq}_grn"]
        ref = vod_sbas[f"S{freq}_ref"]
        ele = vod_sbas["Elevation"]
        vod_sbas.loc[:, f"VOD{freq}"] = vod_fun(grn, ref, ele)
        
    # delete all col with all nans
    vod_sbas = vod_sbas.dropna(axis=1, how='all').drop(columns=['Azimuth', 'Elevation', 'CellID'], errors='ignore')
    vod_ts_sbas = vod_sbas.groupby(
        [pd.Grouper(freq=f"{temporal_resolution}min", level='Epoch'), pd.Grouper(level='SV')]).agg(agg_fun_ts)
    
    # widen the SV col, add SV as suffix to each col
    vod_ts_sbas = vod_ts_sbas.unstack(level='SV')

    # flatten the MultiIndex columns
    vod_ts_sbas.columns = ["_".join(col).strip() for col in vod_ts_sbas.columns.values]
    
    # add the col VOD{freq}_S as the mean of all VOD{freq}_S{sv} columns
    for band in band_ids:
        if f"VOD{band[-1]}_S" not in vod_ts_sbas.columns:
            vod_ts_sbas[f"VOD{band[-1]}_S"] = vod_ts_sbas.filter(like=f"VOD{band[-1]}_S").mean(axis=1)
    
    # -----------------------------------
    # Add biomass-binned anomalies if requested
    if kwargs.get('calculate_biomass_bins', True):
        # GPS only
        vod_ts_biomass_gps = calculate_biomass_binned_anomalies(
            vod, vod_avg, band_ids, temporal_resolution,
            con=['GPS'], biomass_bins=biomass_bins, suffix='gps', **kwargs
        )
        
        # All constellations
        vod_ts_biomass_all = calculate_biomass_binned_anomalies(
            vod, vod_avg, band_ids, temporal_resolution,
            con=None, biomass_bins=biomass_bins, **kwargs
        )
    # -----------------------------------
    # join new features to the aggregated time series
    ds3 = pd.DataFrame({
        'Ns_t': ns_t,
        'SD_Ns_t': sd_ns_t,
        'C_t_perc': cell_coverage
    })
    
    vod_ds_combined = (vod_ts_1.join(vod_ts_2).
                       join(ds3).
                       join(vod_ts_sbas).
                       join(vod_ts_con1).
                       join(vod_ts_con2).
                       join(vod_ts_biomass_gps).
                       join(vod_ts_biomass_all))
    
    
    # -----------------------------------
    ploting_hemi = False
    if ploting_hemi:
        hemi = hemibuild(2, 10)
        plot_hemi(vod_avg, hemi.patches(), var="VOD1_count", clim="auto", title="CellID Observation Counts", )
        
    return (vod_ds_combined, vod_avg)
