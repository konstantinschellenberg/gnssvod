#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from pathlib import Path

import gnssvod as gv
from definitions import FIG, DATA, ROOT, get_repo_root, AUX, GROUND, TOWER
from gnssvod.geodesy.coordinate import ell2cart

from datetime import datetime
import calendar

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

def main():
    
    # -----------------------------------
    # unzipping
    zip_archive = ROOT / 'zip_archive'

    testdir = DATA / "test"
    
    # if not (DATA / GROUND).exists():
    #     unpack_gz_files(search_dir=zip_archive / GROUND, out_dir=DATA / GROUND)
    # if not (DATA / TOWER).exists():
    #     unpack_gz_files(search_dir=zip_archive / TOWER, out_dir=DATA / TOWER)
    
    # -----------------------------------
    # binex to rinex
    
    """
    Use RTKLIB to convert BINEX to RINEX
    
    Installation:
    wget https://github.com/tomojitakasu/RTKLIB/archive/refs/tags/2.4.3.b34L-pre0.tar.gz
    tar -xzf 2.4.3.b34L-pre0.tar.gz
    
    cd app/convbin
    make
    sudo make install
    
    convbin is now an executable.
    """
    
    # TOWER
    # bin2rin(search_dir=DATA / TOWER, out_dir=DATA / TOWER, overwrite=False, num_workers=18)
    # GROUND
    # bin2rin(search_dir=DATA / GROUND, out_dir=DATA / GROUND, overwrite=False, num_workers=18)
    
    # -----------------------------------
    # indexing
    
    # example file
    year = 2022
    doy = 122
    
    # month = 5
    # doys = get_doys_of_month(year, month)
    
    # search pattern needs to be glob-compatible
    all_per_year = f"SEPT???[a-z].{year % 100:02d}.obs"
    one_day = f"SEPT{doy:03d}[a-z].{year % 100:02d}.obs"
    
    search_horizont = {
        "all_per_year": all_per_year,
        "one_day": one_day,
    }
    
    # what variables should be kept
    keepvars = ['S?', 'S??']
    
    moflux_coordinates = {"lat": 38.7441,
                          "lon": 360 - 92.2,
                          "h": 219}
    # non-sane coords
    pos = ell2cart(**moflux_coordinates)
    # (-191222.2171873083, -4977655.015253694, 3970336.885343948)
    
    # -----------------------------------
    # Process 1 dataset
    
    one_dataset = True
    if one_dataset:
        # testdir = DATA / "test"
        station = "MOz1_Twr"
        filepattern = {station: str(DATA / GROUND / search_horizont["one_day"])}
        outpattern = {station: str(DATA / GROUND)}
        
        args = {
            'filepattern': filepattern,
            'interval': '15s',
            'keepvars': keepvars,
            'outputdir': outpattern,
            'overwrite': True,
            'approx_position': pos,
            'aux_path': str(AUX),
            'outputresult': True,
            'num_workers': 15,
        }
        result = gv.preprocess(**args)
        
        # and show data frame
        res = result[station][0].observation
        res.columns
        
        res['S1C']
        #print percentage of NaN values per column
        print(res.isna().mean() * 100)
        
        # sort all columns alphabetically
        res = res.reindex(sorted(res.columns), axis=1)
        
        #make a barplot of means per col
        res.mean().plot(kind='bar', figsize=(10, 6))
        from matplotlib import pyplot as plt
        plt.show()
        

    # -----------------------------------
    # batch processing

    both_datasets = False
    if both_datasets:
        pattern = {'MOz1_Grnd': str(DATA / GROUND / one_day),
                   'MOz1_Twr': str(DATA / TOWER / one_day)}
        outputdir = {'MOz1_Grnd': str(DATA / GROUND),
                     'MOz1_Twr': str(DATA / TOWER)}
        
        arg = {
            'filepattern': pattern,
            'interval': '15s',
            'keepvars': keepvars,
            'outputdir': outputdir,
            'overwrite': False,
            'approx_position': pos,
            'aux_path': str(AUX),
            'outputresult': True,
        }
        
        res = gv.preprocess(**arg)
        len(res['MOz1_Grnd'])
        len(res['MOz1_Twr'])
        twr_obs = res['MOz1_Twr'][0]
        
        # inspect the data
        twr_obs.to_xarray()
        
        # as pandas
        twr = res['MOz1_Twr'][0].observation
        grnd = res['MOz1_Grnd'][0].observation

        twr.columns
        grnd.columns
        
        # print all columns that are in common
        common_cols = set(twr.columns).intersection(set(grnd.columns))
        print("cols in common\n", common_cols)
        
        # print all columns that are not in common
        twr_only = set(twr.columns).difference(set(grnd.columns))
        grnd_only = set(grnd.columns).difference(set(twr.columns))
        print("twr only\n", twr_only)
        print("grnd only\n", grnd_only)
        
if __name__ == '__main__':
    main()