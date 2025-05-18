#!/usr/bin/env python
# -*- coding: utf-8 -*-
import calendar
from datetime import datetime

import pandas as pd

from gnssvod.geodesy.coordinate import ell2cart

# -----------------------------------
# SETTINGS (user) - general

station = 'MOz'
ground_station = f"{station}1_Grnd"
tower_station = f"{station}1_Twr"

overwrite = False  # overwrite existing files
output_results_locally = False  # save results locally
time_selection = "all_per_year"  # or "one_day" or "all_per_year"

# example year
year = 2024
# example file
doy = 150

date_to_skip = [(2024, 344), (2024, 174), (2024, 173)]

# -----------------------------------
# SETTINGS (user) - 01_run_preprocessing.py
# -----------------------------------

# script
unzipping_run = False
binex2rinex_run = False
one_dataset_run = True
both_datasets_run = False

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

bands = {'VOD1':['S1','S1X','S1C'], 'VOD2':['S2','S2X','S2C']}
angular_resolution = 2  # degrees
temporal_resolution = 30  # minutes
anomaly_type = "phi_theta"  # or "phi_theta_sv"
plot = True

# -----------------------------------
# settings (static)
# -----------------------------------

# search pattern needs to be glob-compatible
all_per_year = f"SEPT???[a-z].{year % 100:02d}"
one_day = f"SEPT{doy:03d}[a-z].{year % 100:02d}"

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
    periods = timedelta.total_seconds() / pd.Timedelta(timeintervals_freq).total_seconds()
    timeintervals_periods = int(periods)
    
    
# create time intervals
timeintervals = pd.interval_range(start=startdate,
                                  periods=timeintervals_periods,
                                  freq=timeintervals_freq,
                                  closed=timeintervals_closed)

search_horizont = {
    "all_per_year": all_per_year,
    "one_day": one_day,
}
# what variables should be kept
keepvars = ['S?', 'S??']

# non-sane coords
moflux_coordinates = {"lat": 38.7441,
                      "lon": 360 - 92.2,
                      "h": 219}
pos = ell2cart(**moflux_coordinates)




