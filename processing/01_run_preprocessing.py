#!/usr/bin/env python
# -*- coding: utf-8 -*-
from download import ping_ftp
import gnssvod as gv
from definitions import DATA, ZIP, AUX, GROUND, TOWER
from gnssvod.io.bin2rin import bin2rin
from gnssvod.io.unpack_zip import unpack_gz_files

from processing.settings import *

# -----------------------------------
suffix = "*.obs"
# -----------------------------------
# unzipping

ping_ftp()

if unzipping_run:
    if not (DATA / GROUND).exists():
        unpack_gz_files(search_dir=ZIP / GROUND, out_dir=DATA / GROUND)
    if not (DATA / TOWER).exists():
        unpack_gz_files(search_dir=ZIP / TOWER, out_dir=DATA / TOWER)

# -----------------------------------
# binex to rinex

if binex2rinex_run:
    # GROUND
    bin2rin(search_dir=DATA / GROUND, driver=binex2rinex_driver, out_dir=DATA / GROUND, overwrite=False, num_workers=18)
    # TOWER
    bin2rin(search_dir=DATA / TOWER, driver=binex2rinex_driver, out_dir=DATA / TOWER, overwrite=False, num_workers=18)


# -----------------------------------
# Process 1 dataset
# -----------------------------------

if one_dataset_run:
    # testdir = DATA / "test"
    if single_station_to_be_preprocessed == ground_station:
        subpath = GROUND
        path = str(DATA / GROUND / f"{search_horizont[time_selection]}{suffix}")
    elif single_station_to_be_preprocessed == tower_station:
        subpath = TOWER
        path = str(DATA / TOWER / f"{search_horizont[time_selection]}{suffix}")
    else:
        raise ValueError(f"Unknown station: {single_station_to_be_preprocessed}")
    
    filepattern = {single_station_to_be_preprocessed: str(path)}
    outpattern = {single_station_to_be_preprocessed: str(DATA / subpath)}
    
    args = {
        'filepattern': filepattern,
        'interval': '15s',
        'keepvars': keepvars,
        'outputdir': outpattern,
        'overwrite': True if overwrite else False,
        'approx_position': pos,
        'aux_path': str(AUX) if save_orbit else None,
        'outputresult': True if output_results_locally else False,
    }
    # run preprocessing
    result = gv.preprocess(**args)
    
    if output_results_locally:

        # and show data frame
        res = result[station][0].observation
        res.columns
        
        res['S1']
        #print percentage of NaN values per column
        print(res.isna().mean() * 100)
        
        # sort all columns alphabetically
        res = res.reindex(sorted(res.columns), axis=1)
        
        #make a barplot of means per col
        res.mean().plot(kind='bar', figsize=(6, 4))
        from matplotlib import pyplot as plt
        plt.tight_layout()
        plt.show()
        

# -----------------------------------
# batch processing of both datasets
# -----------------------------------

if both_datasets_run:
    pattern = {ground_station: str(DATA / GROUND / f"{search_horizont[time_selection]}{suffix}"),
               tower_station: str(DATA / TOWER / f"{search_horizont[time_selection]}{suffix}")}
    outputdir = {ground_station: str(DATA / GROUND),
                 tower_station: str(DATA / TOWER)}
    
    arg = {
        'filepattern': pattern,
        'interval': '15s',
        'keepvars': keepvars,
        'outputdir': outputdir,
        'overwrite': True if overwrite else False,
        'approx_position': pos,
        'aux_path': str(AUX) if save_orbit else None,
        'outputresult': True if output_results_locally else False,
    }
    
    res = gv.preprocess(**arg)
    if output_results_locally:
        len(res[ground_station])
        len(res[tower_station])
        twr_obs = res[tower_station][0]
        
        # inspect the data
        twr_obs.to_xarray()
        
        # as pandas
        twr = res[tower_station][0].observation
        grnd = res[ground_station][0].observation

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

print("+" * 50)
print("Finished preprocessing.")
print("+" * 50)