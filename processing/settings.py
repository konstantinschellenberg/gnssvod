#!/usr/bin/env python
# -*- coding: utf-8 -*-
import calendar
from datetime import datetime

import pandas as pd

from gnssvod.geodesy.coordinate import ell2cart

def get_doys_of_month(year: int, month: int) -> list[int]:
    """
    Get all Days of Year (DOYs) for a given month.

    Parameters
    ----------
    year : int
        The year for which to calculate DOYs.
    month : int
        The month for which to calculate DOYs.

    Returns
    -------
    list[int]
        A list of DOYs for the given month.
    """
    doys = []
    for day in range(1, calendar.monthrange(year, month)[1] + 1):
        date = datetime(year, month, day)
        doys.append(date.timetuple().tm_yday)
    return doys


# -----------------------------------
# -----------------------------------
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
save_orbit = False  # save orbit files
output_results_locally = False
time_selection = "all_per_year"  # or "one_day" or "all_per_year"
overwrite = True

# example file
year = 2024
doy = 122

# -----------------------------------
# SETTINGS (user) – 02_gather_stations.py

timeintervals_periods = 1
timeintervals_freq = 'D'
timeintervals_closed = 'left'

# -----------------------------------
# settings (static)
# -----------------------------------

# search pattern needs to be glob-compatible
all_per_year = f"SEPT???[a-z].{year % 100:02d}"
one_day = f"SEPT{doy:03d}[a-z].{year % 100:02d}"

date = pd.to_datetime(f"{year}-{doy}", format='%Y-%j')
timeintervals = pd.interval_range(start=date,
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




