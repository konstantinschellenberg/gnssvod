#!/usr/bin/env python
# -*- coding: utf-8 -*-
import warnings

import numpy as np
import pandas as pd

from analysis.aux_plotting import plot_sv_observation_counts


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
    agg_fun_vodoffset = kwargs.get('agg_fun_vodoffset', 'mean')  # aggregation function for VOD offset
    agg_fun_satincell = kwargs.get('agg_fun_satincell', 'mean')  # aggregation function for satellite in cell
    eval_num_obs_tps = kwargs.get('eval_num_obs_tps', False)  # whether to evaluate number of observations per SV
    
    vod_ts_all = []
    suffix = f"_{suffix}" if suffix else ""
    
    # get all satellite vehicles (SVs) from the index
    svs = vod_with_cells.index.get_level_values('SV').unique()
    
    # Process each satellite vehicle separately
    for sv in svs:
        # Extract data for this SV only
        vod_sv = vod_with_cells.xs(sv, level='SV')
        
        # todo: revisit this condition
        # Skip if there's not enough data for this SV
        # if len(vod_sv) < 100:
        #     print(f"Skipping SV {sv} due to insufficient data.")
        #     continue
        
        aggregation_dict_over_cells = {band: agg_fun_satincell for band in band_ids}
        aggregation_dict_over_cells.update({'CellID': 'size'})  # get length of vector
        
        # Calculate average values per grid cell for this specific SV
        vod_avg_sv = vod_sv.groupby(['CellID']).agg(aggregation_dict_over_cells)
        # rename CELLid to n
        vod_avg_sv = vod_avg_sv.rename(columns={'CellID': 'n'})
        
        # -----------------------------------
        # EVALUATION OF MINIMUM NUMBER OF OBSERVATIONS IN PER CELL PER SATELLITE
        # print(f"Total number of cells for SV {sv}: {vod_avg_sv['n'].nunique()}")
        plot = False
        if plot:
            print(f"Total number of cells for SV {sv}: {vod_avg_sv['n'].nunique()}")
            # plot the histogram of n
            vod_avg_sv['n'].hist(bins=50, alpha=0.5, figsize=(6, 4))
            from matplotlib import pyplot as plt
            plt.xlabel('Number of Observations per Cell')
            plt.ylabel('Frequency')
            plt.title(f"Histogram of Observations per Cell for SV {sv}")
            plt.show()
        # -----------------------------------
        
        # Join the cell averages back to the original data
        vod_anom_sv = vod_sv.join(vod_avg_sv, on='CellID', rsuffix='_mean')

        # Calculate anomalies for each band
        for band in band_ids:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                offset = vod_anom_sv[f"{band}"].agg(agg_fun_vodoffset)  # median offset for the band
                vod_anom_sv[f"{band}_anom"] = vod_anom_sv[band] - vod_anom_sv[f"{band}_mean"] + offset
        
        # Add SV as a column before appending
        vod_anom_sv['SV'] = sv
        vod_anom_sv = vod_anom_sv.reset_index()
        vod_ts_all.append(vod_anom_sv)
    
    # Combine all SV results
    if vod_ts_all:
        vod_ts_svs = pd.concat(vod_ts_all).set_index(['Epoch', 'SV'])
        
        # Evaluate number of observations per SV if requested
        if eval_num_obs_tps:
            plot_sv_observation_counts(vod_ts_svs, min_threshold=10)
            
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
    pd.options.mode.chained_assignment = None
    agg_fun_ts = kwargs.get('agg_fun_ts', 'mean')
    suffix = kwargs.get('suffix', '')
    plot_bins = kwargs.get('plot_bins', False)  # whether to plot the bin histograms for visualization
    
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

    
    # Initialize results DataFrame
    combined_results = pd.DataFrame()
    
    # Process each band
    for band in band_ids:
        # Create biomass bins based on VOD averages for the reference band
        band_means = vod_avg[f"{band}_mean"]
        
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
        
        # Adding the sky sector mean (does not need to be bin-wise sky sectors fall into bins automatically)
        vod_with_means = vod_with_bins.join(vod_avg[[f"{band}_mean"]], on='CellID')
        
        bins = {}
        # -----------------------------------
        # Calculate anomalies for each bin
        for bin_num in range(biomass_bins):
            # Filter data for this bin
            bin_data = vod_with_means[vod_with_means['biomass_bin'] == bin_num]
            bin_mean = bin_data[band].mean()
            
            # Subtracts long-term mean of sky-sector from the binned VOD values
            # mute SettingWithCopyWarning
            bin_data.loc[:, f"{band}_anom"] = bin_data[band] - bin_data[f"{band}_mean"]
                
            # populate the bins dict with the anomalies
            bins[bin_num] = pd.DataFrame({
                'Epoch': bin_data.index.get_level_values('Epoch'),
                f"{band}_anom": bin_data[f"{band}_anom"],
                f"{band}": bin_data[band],
            })
            
            # adding bin-internal mean to the anomaly data
            bin_data.loc[:, f"{band}_anom"] = bin_data[f"{band}_anom"] + bin_mean
                
            # Temporal aggregation
            bin_ts = bin_data.groupby(pd.Grouper(freq=f"{temporal_resolution}min", level='Epoch'))[f"{band}_anom"].agg(
                agg_fun_ts)
            
            # Add to results with appropriate column name
            combined_results[f"{band}_anom_bin{bin_num}{suffix}"] = bin_ts
            
        # After processing individual bins, add a combined high-biomass bin (3-5)
        # Collect dataframes from bins 3-5 (higher biomass)
        high_biomass_bins = pd.concat([bins[i] for i in range(3, min(5 + 1, biomass_bins))])
        high_biomass_mean = high_biomass_bins[band].mean()
        high_biomass_bins[f"{band}_anom_mean"] = high_biomass_bins[f"{band}_anom"] + high_biomass_mean
        
        # Temporal aggregation for the combined high biomass bins
        high_biomass_ts = high_biomass_bins[f"{band}_anom_mean"].groupby(
            pd.Grouper(freq=f"{temporal_resolution}min", level='Epoch')
        ).agg(agg_fun_ts)
        
        # Add to results with appropriate column name
        combined_results[f"{band}_anom_bin3-5{suffix}"] = high_biomass_ts
        
        if plot_bins:
            # Calculate actual bin edges for visualization
            if isinstance(bins, pd.Series):
                # Get the unique bin values and their corresponding data
                bin_groups = {}
                for bin_num in range(biomass_bins):
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

    return combined_results


