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
    
    print(f"Pattern for ground station: {pattern_ground}")
    print(f"Pattern for tower station: {pattern_tower}")
    
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
            'n_workers': 17,
    }
    
    # -----------------------------------
    # run function
    out = gv.gather_stations(mergebands=True, **args)

    # -----------------------------------
    # inspect
    
    if output_results_locally:
        moz = out["MOz"].copy()
        moz.hist(column=['S1', 'S2'], bins=50, alpha=0.5, figsize=(6, 4))
        from matplotlib import pyplot as plt
        
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()
        
    print("+" * 50)
    print("Finished gathering stations.")
    print("+" * 50)
    
    
