#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gnssvod as gv
import pandas as pd
from definitions import DATA, ROOT, GROUND, TOWER

import glob
from pathlib import Path

def search_and_print_files(pattern: str):
    """
    Search for files matching the given pattern and print them nicely.

    Parameters
    ----------
    pattern : str
        The glob pattern to search for files.
    """
    files = glob.glob(pattern)
    if files:
        printit(f"Found {len(files)} file(s) matching the pattern '{pattern}':")
        for file in files:
            printit(f"{Path(file).name}", end=', ')
    else:
        printit(f"No files found matching the pattern '{pattern}'.")



if __name__ == '__main__':
    
    doy = 122
    year = 2022
    # #convert doy to date in year
    date = pd.to_datetime(f"{year}-{doy}", format='%Y-%j')
    timeintervals = pd.interval_range(start=date, periods=1, freq='D', closed='left')
    
    all_per_year = f"SEPT???[a-z].{year % 100:02d}.nc"
    one_day = f"SEPT{doy:03d}[a-z].{year % 100:02d}.nc"
    
    search_horizont = {
        "all_per_year": all_per_year,
        "one_day": one_day,
    }
    
    pattern_ground = str(DATA / GROUND / one_day)
    pattern_tower = str(DATA / TOWER / one_day)
    
    pattern = {'MOz1_Grnd': pattern_ground,
             'MOz1_Twr': pattern_tower}
    outputdirs = {'MOz': str(DATA / 'gather')}
    Path(outputdirs['MOz']).mkdir(exist_ok=True)
    
    # print all files matching the pattern using glob
    # Example usage
    printit = False
    if printit:
        search_and_print_files(pattern_ground)
        search_and_print_files(pattern_tower)

    # define how to make pairs, always give reference station first, matching the dictionary keys of 'pattern'
    pairings = {'MOz': ('MOz1_Twr', 'MOz1_Grnd')}
    keepvars = ['S*', 'Azimuth', 'Elevation']
    
    args = {'filepattern': pattern,
            'pairings': pairings,
            'timeintervals': timeintervals,
            'outputdir': outputdirs,
            'keepvars': keepvars,
            'outputresult': True,
    }
    
    # run function
    out = gv.gather_stations(**args)
    
