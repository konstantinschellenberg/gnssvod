#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gnssvod as gv
import pandas as pd
import glob
from pathlib import Path

from definitions import DATA, ROOT, GROUND, TOWER
from processing.settings import *

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
        print(f"Found {len(files)} file(s) matching the pattern '{pattern}':")
        for file in files:
            print(f"{Path(file).name}", end=', ')
    else:
        print(f"No files found matching the pattern '{pattern}'.")
        
if __name__ == "__main__":

    # Gather stations
    suffix = ".*nc"
    pattern_ground = str(DATA / GROUND / f"{search_horizont[time_selection]}{suffix}")
    pattern_tower = str(DATA / TOWER / f"{search_horizont[time_selection]}{suffix}")
    
    pattern = {ground_station: pattern_ground,
               tower_station: pattern_tower}
    
    outputdirs = {'MOz': str(DATA / 'gather')}
    Path(outputdirs['MOz']).mkdir(exist_ok=True)
    
    # -----------------------------------
    # print all files matching the pattern using glob
    printit = False
    if printit:
        search_and_print_files(pattern_ground)
        search_and_print_files(pattern_tower)
    
    # -----------------------------------
    # define how to make pairs, always give reference station first, matching the dictionary keys of 'pattern'
    pairings = {'MOz': (tower_station, ground_station)}
    keepvars = ['S*', 'Azimuth', 'Elevation']
    
    args = {'filepattern': pattern,
            'pairings': pairings,
            'timeintervals': timeintervals,
            'outputdir': outputdirs,
            'keepvars': keepvars,
            'outputresult': True if output_results_locally else False,
    }
    
    # -----------------------------------
    # run function
    out = gv.gather_stations(mergebands=True, **args)
    
    # -----------------------------------
    # merge subbands
    
    
    moz = out["MOz"].copy()
    
    # make a hist of S1 and S2
    moz.hist(column=['S1', 'S2'], bins=50, alpha=0.5, figsize=(6, 4))
    from matplotlib import pyplot as plt
    plt.legend(title='Station', loc='upper right')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()
    
    
    
    # -----------------------------------
    # inspect
    
    if output_results_locally:
        
        # -----------------------------------
        # nan in cols
        # print percentage of NaN values per column and station (ignored Epoch and SV in indices)
        df = out["MOz"].copy().reset_index().set_index(['Station'])
        df = df.reindex(sorted(df.columns), axis=1)
    
        
        # print unique SVs
        svs = df['SV'].unique()
        svs = [sv for sv in svs if isinstance(sv, str)]
        print("SVs in data:\n", sorted(svs))
        
        # etract first digit and add as col: "con"
        df['con'] = df['SV'].str.extract(r'([a-zA-Z])')[0]
        
        # sort alphabetically
        df.select_dtypes(include='number').aggregate(lambda x: x.isna().sum() / len(df) * 100, axis=0).plot(kind='bar', figsize=(6, 4))
        from matplotlib import pyplot as plt
        plt.show()
        
        # calculate the percentage of NaN values per column
        nan_percentage = df.isna().mean() * 100
        # print the percentage of NaN values
        print("Percentage of NaN values per column:")
        for col, percentage in nan_percentage.items():
            print(f"{col}: {percentage:.2f}%")
        
        bands = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8']
        
        
        
        # average all SV
        out = out["MOz"].groupby(level=[0, 1]).mean()
        
        # make histogram of all cols, hue=station
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        out.hist(ax=ax, bins=50, alpha=0.5)
        plt.legend(title='Station', loc='upper right')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()
        
    print("+" * 50)
    print("Finished gathering stations.")
    print("+" * 50)
    
    
