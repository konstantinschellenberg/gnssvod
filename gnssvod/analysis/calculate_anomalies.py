#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd

from analysis.calculate_anomaly_functions import calc_anomaly_ak, calc_anomaly_ksak, calc_anomaly_vh, \
    calculate_biomass_binned_anomalies, \
    calculate_extinction_coefficient, \
    calc_anomaly_ks, \
    create_aggregation_dict, vod_fun
from gnssvod import hemibuild
from processing.export_vod_funs import plot_hemi
from processing.settings import canopy_height, z0


def _calculate_anomaly(vod, band_ids, cfg, show=False, **kwargs):
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
    
    
    # -----------------------------------
    # Statistical metrics
    
    sv_counts = vod.reset_index().groupby('Epoch')['SV'].nunique()

    # -----------------------------------
    # calculate extinction coefficient if requested
    if cfg.make_ke:
        vod, band_ids = calculate_extinction_coefficient(
            vod,
            band_ids,
            canopy_height=canopy_height,
            z0=z0
        )
    
    # -----------------------------------
    # -----------------------------------
    # Anomaly calculations
    kwargs_anom = {
        'temporal_resolution': cfg.temporal_resolution,
        'agg_fun_vodoffset': cfg.agg_fun_vodoffset,
        'agg_fun_satincell': cfg.agg_fun_satincell,
        'eval_num_obs_tps': cfg.eval_num_obs_tps
    }
    
    # -----------------------------------
    # Method 1: Vincent's method (phi_theta) - short tp
    # -----------------------------------
    vod_ts_1, vod_avg = calc_anomaly_vh(vod, band_ids, suffix="vh", show=show, **kwargs_anom)

    # -----------------------------------
    # Method 2: Konstantin's extension >(phi_theta_sv) - short tps
    # -----------------------------------
    vod_ts_2 = calc_anomaly_ks(vod, band_ids, suffix="ks", show=show, **kwargs_anom)
    
    # -----------------------------------
    # Method 3: Alex' approach
    # -----------------------------------
    vod_ts_3 = calc_anomaly_ak(vod, band_ids, timedelta=cfg.anom_ak_timedelta, suffix="ak", **kwargs_anom)
    
    # -----------------------------------
    # Method 4: Konstantin and Alex combined approach
    vod_ts_4 = calc_anomaly_ksak(vod, band_ids, timedelta=cfg.anom_ak_timedelta, suffix="ksak", **kwargs_anom)
    
    
    # -----------------------------------
    # Calculate VOD anomalies for GPS and GALILEO separately
    
    vod_gps_gal = vod[vod.index.get_level_values('SV').str.startswith(('G', 'E'))]
    if not vod_gps_gal.empty:
        vod_ts_con2 = calc_anomaly_ks(vod_gps_gal, band_ids, suffix="gps+gal", **kwargs_anom)
  
    # end anomaly calculations
    # -----------------------------------
    # -----------------------------------
    # NEW FEATURES
    # Calculate Ns(t): Average number of satellites per resampled epoch
    ns_t = sv_counts.groupby(pd.Grouper(freq=f"{cfg.temporal_resolution}min")).mean().rename('Ns_t')
    
    # Calculate SD(Ns(t)): Standard deviation of the number of satellites per resampled epoch
    sd_ns_t = sv_counts.groupby(pd.Grouper(freq=f"{cfg.temporal_resolution}min")).std().rename('SD_Ns_t')
    
    # Calculate C(t): Percentage of sky plots covered per time period
    # First, count total unique cells in the entire dataset
    total_possible_cells = vod['CellID'].nunique()
    
    # cutoff already applied to hemi grid - doesn't need to be filtere here again :)
    # For each time window, calculate the percentage of sky covered
    cell_coverage = vod.groupby(pd.Grouper(freq=f"{cfg.temporal_resolution}min", level='Epoch')).apply(
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
        [pd.Grouper(freq=f"{cfg.temporal_resolution}min", level='Epoch'), pd.Grouper(level='SV')]).agg(cfg.agg_fun_ts)
    
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
    if cfg.calculate_biomass_bins:
        # GPS only
        vod_ts_anom_biomass_gps = calculate_biomass_binned_anomalies(
            vod, vod_avg, band_ids, con=['GPS', 'Galileo'], biomass_bins=cfg.biomass_bins, suffix='gps+gal', **kwargs
        )

    # -----------------------------------
    # join new features to the aggregated time series
    ds3 = pd.DataFrame({
        'Ns_t': ns_t,
        'SD_Ns_t': sd_ns_t,
        'C_t_perc': cell_coverage
    })
    
    # -----------------------------------
    # MERGE ALL TIME SERIES
    vod_ds_combined = (vod_ts_1.join(vod_ts_2).join(vod_ts_4).
                       join(vod_ts_3).
                       join(ds3).
                       join(vod_ts_sbas))
    
    if not vod_gps_gal.empty:
        vod_ds_combined = vod_ds_combined.join(vod_ts_con2)
    
    if cfg.calculate_biomass_bins:
        vod_ds_combined = vod_ds_combined.join(vod_ts_anom_biomass_gps)
        
    # -----------------------------------
    # if param cols don't exist, make them here
    # param_cols = ['angular_resolution', 'angular_cutoff', 'temporal_resolution']
    #
    # vod_ds_combined[param_cols] = pd.DataFrame(
    #     [[cfg.angular_resolution, cfg.angular_cutoff, cfg.temporal_resolution]],
    #     index=vod_ds_combined.index
    # )
    
    return (vod_ds_combined, vod_avg)
