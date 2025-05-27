#!/usr/bin/env python
# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
import gnssvod as gv

from definitions import DATA
from processing.export_vod_funs import plot_anomaly, plot_hemi, save_vod_timeseries
from processing.filepattern_finder import create_time_filter_patterns
from processing.settings import *

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

# suffix = ".*nc"
pattern = str(DATA / "gather" / f"{create_time_filter_patterns(time_interval)['nc']}")
# Filter to exact date range

# define how to associate stations together. Always put reference station first.
pairings = {station:(tower_station, ground_station)}
# define if some observables with similar frequencies should be combined together. In the future, this should be replaced by the selection of frequency bands.

vod = gv.calc_vod(pattern, pairings, bands, time_interval)[station]

# print the percentage of NaN values per column
print("NaN values in VOD:")
print(vod.isna().mean() * 100)

# -----------------------------------
# hemi grid

# todo: detach this into anomaly_type == "phi_theta"
# intialize hemispheric grid
hemi = gv.hemibuild(angular_resolution, angular_cutoff)
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
    # plot_hemi(vod_avg, patches, title=f"VOD {station} {year}-{doy}")
    pass

# -----------------------------------
# calculate anomaly

print("Anomaly calulation type", anomaly_type)

if anomaly_type == "phi_theta":
    vod_anom = vod.join(vod_avg, on='CellID')
    for band in band_ids:
        vod_anom[f"{band}_anom"] = vod_anom[band] - vod_anom[f"{band}_mean"]
    vod_ts = vod_anom.groupby(pd.Grouper(freq=f"{temporal_resolution}min", level='Epoch')).agg(agg_func)
    for band in band_ids:
        vod_ts[f"{band}_anom"] = vod_ts[f"{band}_anom"] + vod_ts[f"{band}"].mean()
    if plot:
        plot_anomaly(vod_ts, band_ids, figsize=(6, 4), title=f"VOD Anomalies per sat {station} {year}-{doy}\n Vincent's method")
elif anomaly_type == "phi_theta_sv":
    """
    1. Calculate anomaly for each SV separately
    2. Temporally aggregate the anomalies for all SVs
    
    """
    # Initialize an empty DataFrame to store the final results
    vod_ts_all = []
    
    # Process each satellite vehicle separately
    for sv in vod.index.get_level_values('SV').unique():
        """ Loop through each SV in the VOD data"""
        # Extract data for this SV only
        vod_sv = vod.xs(sv, level='SV')
        
        # Skip if there's not enough data for this SV
        if len(vod_sv) < 100:  # Adjust threshold as needed
            print(f"Skipping SV {sv} - insufficient data points ({len(vod_sv)})")
            continue
        
        # Calculate average values per grid cell for this specific SV
        vod_avg_sv = vod_sv.groupby(['CellID']).agg(['mean', 'std', 'count'])
        # Flatten the columns
        vod_avg_sv.columns = ["_".join(x) for x in vod_avg_sv.columns.to_flat_index()]

        # Join the cell averages back to the original data
        vod_anom_sv = vod_sv.join(vod_avg_sv, on='CellID')
        
        # Calculate anomalies for each band by subtracting the mean
        for band in band_ids:
            vod_anom_sv[f"{band}_anom"] = vod_anom_sv[band] - vod_anom_sv[f"{band}_mean"]

        # # Add SV as a column before appending to the combined results
        vod_anom_sv['SV'] = sv
        vod_anom_sv = vod_anom_sv.reset_index()
        vod_ts_all.append(vod_anom_sv)

    # Combine all SV results
    if vod_ts_all:
        """2. Temporally aggregate the anomalies for all SVs"""
        vod_ts_svs = pd.concat(vod_ts_all).set_index(['Epoch', 'SV'])
        
        # disregards SV as expected
        vod_ts = vod_ts_svs.groupby(pd.Grouper(freq=f"{temporal_resolution}min", level='Epoch')).agg(agg_func)
        
        # Add back the mean to make anomalies comparable
        for band in band_ids:
            vod_ts[f"{band}_anom"] = vod_ts[f"{band}_anom"] + vod_ts[f"{band}"].mean()
        
        # Plot the combined anomalies
        if plot:
            plot_anomaly(vod_ts, band_ids, figsize=(6, 4), title=f"VOD Anomalies per sat {station} {year}-{doy}\n Konstantin's extension")
    else:
        print("No valid data for any SV")
        vod_ts = pd.DataFrame()  # Empty DataFrame as fallback
else:
    raise ValueError(f"Unknown anomaly type: {anomaly_type}")

# -----------------------------------

print(vod_ts.describe())

# -----------------------------------
# Save VOD time series to NetCDF file

filename = f"vod_timeseries_{station}_{year}_{doy}"

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


# elif anomaly_type == "phi_theta_sv":
#     """
#     Rationale:
#     - Calculate anomaly for each SV separately
#     """
#     for band in band_ids:
#         for sv in vod.index.get_level_values('SV').unique():
#             vod_sv = vod.xs(sv, level='SV')
#             if plot:
#                 plt.figure(figsize=(10, 5))
#                 vod_sv[band].plot(title=f"VOD {station} {year}-{doy} SV {sv} {band}")
#                 plt.show()
#             vod_avg_sv = vod_sv.groupby('CellID').agg(agg_func)
#             # print unique cellids
#             print(f"SV {sv} has {len(vod_avg_sv)} unique CellIDs")
#
#             # flatten the columns
#             vod_sv_anom = vod.xs(sv, level='SV').copy().reset_index().set_index(['CellID', 'Epoch'])
#             # Use loc for assignment to be explicit about where data is being modified
#             vod_sv_anom.loc[:, f"{band}_anom"] = vod_sv_anom[band] - vod_avg_sv[band]
#             # reset cellid index
#             vod_sv_anom = vod_sv.reset_index().set_index("Epoch")
#             # drop CellID col
#             vod_sv_anom = vod_sv_anom.drop(columns=['CellID'])
#             vod_ts = vod_sv_anom.groupby(pd.Grouper(freq=f"{temporal_resolution}min", level='Epoch')).agg(agg_func)
#         vod_ts[f"{band}_anom"] = vod_ts[f"{band}_anom"] + vod_ts[f"{band}"].mean()
#         if plot:
#             plot_anomaly(vod_ts, band_ids, figsize=(6, 4))
#
#     vod_anom = vod.join(vod_avg, on='CellID')