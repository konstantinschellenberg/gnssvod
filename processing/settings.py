#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd

from definitions import ARD, ENVDATA
from gnssvod.geodesy.coordinate import ell2cart
from processing.helper import create_time_intervals

"""
THESE SETTINGS ARE USED FOR USING ONE SITE AND ONE ANTENNA PAIR
"""

# -----------------------------------
# SETTINGS (user) - general

station = 'MOz'
ground_station = f"{station}1_Grnd"
tower_station = f"{station}1_Twr"

overwrite = False  # overwrite existing files
output_results_locally = False  # save results locally
time_selection = "all_per_year"  # or "one_day" or "all_per_year" or "all_time"

# example year
year = 2025
# example file
doy = 160

dates_to_skip = [(2024, 344), (2024, 174), (2024, 173), (2021, 365)]
# period_anomaly = [("2024-06-12 23:00:00", "2024-06-23 00:00:00"),
#                   ("2024-06-25 23:00:00", "2024-07-11 17:00:00")]
period_anomaly = [("2024-06-12 23:00:00", "2024-07-11 17:00:00")]
# -----------------------------------
# SETTINGS (user) - 01_run_preprocessing.py
# -----------------------------------

# script
unzipping_run = False
binex2rinex_run = False
one_dataset_run = False
both_datasets_run = True

# options
binex2rinex_driver = "teqc"  # or "convbin"
single_station_to_be_preprocessed = ground_station  # or tower_station
save_orbit = True  # save orbit files

# what variables should be kept
keepvars = ['S?', 'S??']
moflux_coordinates = {"lat": 38.7441, "lon": 360 - 92.2, "h": 219}
pos = ell2cart(**moflux_coordinates)

# -----------------------------------
# SETTINGS (user) – 02_gather_stations.py

timeintervals_periods = None  # None or int. If None, it will be calculated. If int, it will be used as is.
timeintervals_freq = '1h'  # hourly file saving is more efficient
timeintervals_closed = 'left'

# -----------------------------------
# SETTINGS (user) – 03_export_vod.py

# general settings
batch_run = True  # run all years in batch mode
time_intervals = create_time_intervals('2024-04-01', '2024-10-31', 4)
"""
Runs:

time_intervals = create_time_intervals('2022-04-03', '2025-05-19', 2)
time_intervals = create_time_intervals('2024-04-01', '2024-10-31', 4)

"""

# set to True to iterate over parameters (angular_resolution, angular_cutoff, temporal_resolution)
iterate_parameters = False

# quick plot of the results
plot_results = True
overwrite_vod_processing = False  # overwrite existing VOD processing files
overwrite_anomaly_processing = True  # overwrite existing anomaly processing files
add_sbas_position_manually = True  # add SBAS position to VOD files

# todo: settings on constellation

# parameters, L5 currently not really implemented, but should work when debugged
bands = {'VOD1':['S1','S1X','S1C'], 'VOD2':['S2','S2X','S2C']} ## 'VOD3':['S3','S3X','S3C'], 'VOD4':['S4','S4X','S4C'], 'VOD5':['S5','S5X','S5C'],
            # 'VOD6':['S6','S6X','S6C'], 'VOD7':['S7','S7X','S7C'], 'VOD8':['S8','S8X','S8C'], 'VOD9':['S9','S9X','S9C'], 'VOD10':['S10','S10X','S10C']}
visualization_timezone = "etc/GMT+6"

# for ke calculation:
canopy_height = 20.0  # meters
z0 = 1.0  # height of the ground receiver
make_ke = False  # whether to calculate ke

# for VOD calculation (must be lists)
angular_resolution = 2  # degrees
temporal_resolution = 30  # minutes  # change from 30
angular_cutoff = 30 # changed from 30
agg_fun_vodoffset = "median"  # aggregation function for VOD offset added to the anomaly, can be "mean" or "median"
agg_fun_ts = "median"   # aggregation function for time series
agg_fun_satincell = "median"  # Konstantin's aggregation function for satellite in cell, can be "mean" or "median"
eval_num_obs_tps = True

gnss_parameters_iteratable = {
    'angular_resolution': angular_resolution,  # must be a list
    'angular_cutoff': angular_cutoff,
    'temporal_resolution': temporal_resolution,
}

gnss_parameters = {
    'make_ke': make_ke,  # whether to calculate ke
    'canopy_height': canopy_height,
    'z0': z0,
    'overwrite': overwrite_anomaly_processing,  # overwrite existing VOD processing files
    "agg_fun_vodoffset": agg_fun_vodoffset,
    "agg_fun_ts": agg_fun_ts,  # aggregation function for time series
    "agg_fun_satincell": agg_fun_satincell,  # Konstantin's aggregation function for satellite in cell
    "eval_num_obs_tps": eval_num_obs_tps,  # whether to evaluate number of observations for TPS
}

# ALWAYS CALC BOTH
# anomaly_type = "phi_theta"  # or "phi_theta_sv" or "phi_theta"

# -----------------------------------
# SETTINGS (user) – 04_merge_years.py

# settings for merging years (these criteria must match all datasets)
angular_resolution = 2
temporal_resolution = 30
angular_cutoff = 30
search_agg_fun_ts = "median"

# general time series settings (manually set)
filter_anomalies = False  # filter anomalies in time series

# VOD "optimized" settings
precip_quantile = 1.0  # NOT USED WHEN `filter_anomalies`. cutoff for precipitation quantile to filter dip-artifacts, e.g. 0.05 for 5% quantile
minimum_nsat = 13  #  def: 15. minimum number of satellites in view on average in a time interval to be considered valid
min_vod_quantile = 0.05  # def: 0.05. cutoff for VOD1_anom to filter dip-artifacts, e.g. 0.05 for 5% quantile
loess_frac = 0.1  # def: 0.1. fraction of data used for LOESS smoothing, e.g. 0.05 for 5% of data
detrend_weekly = False

filepath_environmentaldata = ENVDATA / "tb_interval_20250529_160122.csv"  # path to environmental data file

# -----------------------------------
# SETTINGS (user) – 05_inspect_exported_vodfiles.py

# todo: will be rename to be specific
vod_file = ARD / "combined_vod_data_MOz_2024_to_2024.parquet"

# combined_vod_data_MOz_2024_to_2024.parquet

# subset must be in tz
# time_subset = ('2023-05-15', "2024-12-30")  # ("2024-01-01", "2024-12-30")

# -----------------------------------
# settings (static)
# -----------------------------------

# todo: redo based on create_time_intervals
# search pattern needs to be glob-compatible
all_per_year = f"SEPT???[a-z].{year % 100:02d}"
one_day = f"SEPT{doy:03d}[a-z].{year % 100:02d}"
all_time = f"SEPT???[a-z].[0-9][0-9]"

# in %Y%m%d%H%M%S format from year and doy
one_day_gather = f"{station}_{ pd.to_datetime(f"{year}-{doy}", format='%Y-%j').strftime("%Y%m%d*")}"
all_year_gather = f"{station}_{year}"
all_time_gather = f"{station}_*"

date = pd.to_datetime(f"{year}-{doy}", format='%Y-%j')
if timeintervals_periods == None:
    # get the period of time intervals time_range = periods * freq
    if time_selection == "one_day":
        # get datetime from doy
        startdate = pd.to_datetime(f"{year}-{doy}", format='%Y-%j')
        timedelta = pd.Timedelta(days=1)
    elif time_selection == "all_per_year":
        startdate = pd.to_datetime(f"{year}-01-01")
        timedelta = pd.to_datetime(f"{year}-12-31") - startdate
    elif time_selection == "all_time":
        startdate = pd.to_datetime("2022-01-01")
        timedelta = pd.to_datetime("2024-12-31") - startdate
    periods = timedelta.total_seconds() / pd.Timedelta(timeintervals_freq).total_seconds()
    timeintervals_periods = int(periods)
    
# create time intervals
timeintervals = pd.interval_range(start=startdate,
                                  periods=timeintervals_periods,
                                  freq=timeintervals_freq,
                                  closed=timeintervals_closed)

search_horizont = {
    "level_0": {
        "all_per_year": all_per_year,
        "one_day": one_day,
        "all_time": all_time,
    },
    "level_1": {
        "all_per_year": all_year_gather,
        "one_day": one_day_gather,
        "all_time": all_time_gather,
    }
}

# -----------------------------------
# SBAS Identifiers
# note that the position of the satellite for now need to be manually looked up...

sbas_ident = {
    "S31": {
        "system": "WAAS",
        "Azimuth": 216.5-360,  # Adjusted for azimuth range [-180, 180]
        "Elevation": 31.1,
        "PRN": "131",
    },
    "S33": {
        "system": "WAAS",
        "Azimuth": 230.3-360,
        "Elevation": 38.3,
        "PRN": "133",
    },
    "S35": {
        "system": "WAAS",
        "Azimuth": 225.8-360,
        "Elevation": 33.7,
        "PRN": "135",
    },
    "S48": {
        "system": "Algerian SBAS",
        "Azimuth": 104.6,
        "Elevation": 8.9,  # too low!
        "PRN": "148",
    },
}
