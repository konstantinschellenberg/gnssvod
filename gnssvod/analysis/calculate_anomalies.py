#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd

from analysis.calculate_anomaly_functions import calculate_biomass_binned_anomalies, calculate_extinction_coefficient, \
    calculate_sv_specific_anomalies, \
    create_aggregation_dict, vod_fun
from gnssvod import hemibuild
from processing.export_vod_funs import plot_hemi


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
    make_ke = kwargs.get('make_ke', False)  # whether to calculate extinction coefficient
    agg_fun_ts = kwargs.get('agg_fun_ts', 'mean')  # aggregation function for time series
    biomass_bins = kwargs.get('biomass_bins', 5)  # bins for biomass if needed
    agg_fun_vodoffset = kwargs.get('agg_fun_vodoffset', 'mean')  # aggregation function for VOD offset
    calculate_biomass_bins = kwargs.get('calculate_biomass_bins', True)
    
    # print kwargs in a readable format
    print("Calculating anomalies with the following parameters:")
    for key, value in kwargs.items():
        print(f"  {key}: {value}")
    # -----------------------------------

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
            offset = vod_anom[f"{band}"].agg(agg_fun_vodoffset)  # mean offset for VOD1
            vod_anom[f"{band}_anom_tp"] = vod_anom[band] - vod_anom[f"{band}_mean"] + offset
    vod_anom = vod_anom[[f"{band}_anom_tp" for band in band_ids_tp] + band_ids_tp]
    
    # Temporal aggregation
    vod_ts_1 = vod_anom.groupby(pd.Grouper(freq=f"{temporal_resolution}min", level='Epoch')).agg(agg_fun_ts)
    # -----------------------------------
    # Method 2: Konstantin's extension (phi_theta_sv) - short tps
    vod_ts_2 = calculate_sv_specific_anomalies(vod, band_ids, temporal_resolution, **kwargs)
    
    # -----------------------------------
    # Calculate VOD anomalies for GPS and GALILEO separately
    
    # filter where SV index starts with 'G' (GPS)
    vod_con1 = vod[vod.index.get_level_values('SV').str.startswith('G')]
    kwargs["eval_num_obs_tps"] = False  # do not evaluate number of observations per SV for this method
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
    # todo: requires reimplementation
    
    # num_bins = kwargs.get('vod_percentile_bins', 5)
    # vod_ts_1 = calculate_binned_sky_coverage(vod_with_cells, vod_avg, band_ids, temporal_resolution,
    #                                             num_bins=num_bins, plot_bins=False, suffix="tp")
    # vod_ts_2 = calculate_binned_sky_coverage(vod_with_cells, vod_avg, band_ids, temporal_resolution,
    #                                          num_bins=num_bins, plot_bins=False, suffix="tps")
    
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
        if band.endswith('_ke'):
            continue
        if f"VOD{band[-1]}_S" not in vod_ts_sbas.columns:
            vod_ts_sbas[f"VOD{band[-1]}_S"] = vod_ts_sbas.filter(like=f"VOD{band[-1]}_S").mean(axis=1)
    
    # -----------------------------------
    # Add biomass-binned anomalies if requested
    if calculate_biomass_bins:
        # GPS only
        vod_ts_biomass_gps = calculate_biomass_binned_anomalies(
            vod, vod_avg, band_ids, temporal_resolution,
            con=['GPS'], biomass_bins=biomass_bins, suffix='gps', **kwargs
        )
        
        # All constellations
        kwargs['plot_bins'] = True  # set to True to plot the bin histograms for visualization
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
                       join(vod_ts_con2))
    
    if calculate_biomass_bins:
        vod_ds_combined = vod_ds_combined.join(vod_ts_biomass_gps).join(vod_ts_biomass_all)
    
    # -----------------------------------
    ploting_hemi = False
    if ploting_hemi:
        hemi = hemibuild(2, 10)
        plot_hemi(vod_avg, hemi.patches(), var="VOD1_count", clim="auto", title="CellID Observation Counts", )
        
    return (vod_ds_combined, vod_avg)
