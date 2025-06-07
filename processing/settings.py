#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from matplotlib import rcParams

from gnssvod.geodesy.coordinate import ell2cart

# -----------------------------------q
# set default matplotlib settings figsize 5,3
rcParams['figure.figsize'] = (4,4)
# set tight layout as default
rcParams['figure.autolayout'] = True
# standard point size is 3
rcParams['lines.markersize'] = .5


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

# -----------------------------------
# SETTINGS (user) – 02_gather_stations.py

timeintervals_periods = None  # None or int. If None, it will be calculated. If int, it will be used as is.
timeintervals_freq = '1h'  # hourly file saving is more efficient
timeintervals_closed = 'left'

# -----------------------------------
# SETTINGS (user) – 03_export_vod.py

# general settings
batch_run = False  # run all years in batch mode
iterate_parameters = False  # set to True to iterate over parameters
plot = True  # I think this option is dead
overwrite_vod_processing = False  # overwrite existing VOD processing files
overwrite_anomaly_processing = True  # overwrite existing anomaly processing files
add_sbas_position_manually = True  # add SBAS position to VOD files

# todo: settings on constellation

# parameters
bands = {'VOD1':['S1','S1X','S1C'], 'VOD2':['S2','S2X','S2C']} ## 'VOD3':['S3','S3X','S3C'], 'VOD4':['S4','S4X','S4C'], 'VOD5':['S5','S5X','S5C'],
            # 'VOD6':['S6','S6X','S6C'], 'VOD7':['S7','S7X','S7C'], 'VOD8':['S8','S8X','S8C'], 'VOD9':['S9','S9X','S9C'], 'VOD10':['S10','S10X','S10C']}
single_file_interval = ('2023-04-15', "2023-07-30")
visualization_timezone = "etc/GMT+6"

# for ke calculation:
canopy_height = 20.0  # meters
z0 = 1.0  # height of the ground receiver

# for VOD calculation
angular_resolution = [1]  # degrees
temporal_resolution = [10]  # minutes  # change from 30
angular_cutoff = [30] # changed from 30
# agg_func = "mean"  # or "median"

gnss_parameters = {
    'angular_resolution': angular_resolution,  # must be a list
    'angular_cutoff': angular_cutoff,
    'temporal_resolution': temporal_resolution,
    'make_ke': True,  # whether to calculate ke
    'canopy_height': canopy_height,
    'z0': z0,
    'overwrite': overwrite_anomaly_processing,  # overwrite existing VOD processing files
}

# ALWAYS CALC BOTH
# anomaly_type = "phi_theta"  # or "phi_theta_sv" or "phi_theta"
time_intervals = [
    ('2022-04-03', '2022-12-31'),
    ('2023-01-01', '2023-06-30'),
    ('2023-07-01', '2023-12-31'),
    ('2024-01-01', '2024-06-30'),
    ('2024-07-01', '2024-12-31'),
    ('2025-01-01', '2025-05-19'),
]

# -----------------------------------
# SETTINGS (user) – 04_merge_years.py

# multiparameter (iterate_parameters)?
multiple_parameters = True  # set to True to iterate over parameters

# -----------------------------------
# SETTINGS (user) – 05_inspect_exported_vodfiles.py

"""
Use this time interface for time series selection
"""

# subset must be in tz
time_subset = ('2023-05-15', "2024-12-30")  # ("2024-01-01", "2024-12-30")

load_mode = 'single_file'  # 'multi_year' or single_file or supply_path

# single file
single_file_settings = {
    'station': 'MOz',
    'time_interval':  single_file_interval,
    # 'anomaly_type': 'unknown',  # or 'phi_theta' or 'phi_theta_sv'
}

final_vod_path = "combined_vod_data_MOz_2022_to_2025.csv"

# -----------------------------------
# settings (static)
# -----------------------------------

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
# what variables should be kept
keepvars = ['S?', 'S??']

# non-sane coords
moflux_coordinates = {"lat": 38.7441,
                      "lon": 360 - 92.2,
                      "h": 219}
pos = ell2cart(**moflux_coordinates)




