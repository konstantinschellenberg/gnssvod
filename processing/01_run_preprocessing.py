#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from pathlib import Path

import gnssvod as gv
from definitions import FIG, DATA, ROOT, get_repo_root, AUX
from gnssvod.io.unpack_zip import unpack_gz_files
from gnssvod.io.bin2rin import bin2rin

def main():

    # -----------------------------------
    # unzipping
    zip_archive = ROOT / 'zip_archive'
    ground = 'subcanopy/MOz1_Grnd'
    tower = 'tower/MOz2_Twr'
    testdir = DATA / "test"
    
    # if not (DATA / ground).exists():
    #     unpack_gz_files(search_dir=zip_archive / ground, out_dir=DATA / ground)
    # if not (DATA / tower).exists():
    #     unpack_gz_files(search_dir=zip_archive / tower, out_dir=DATA / tower)
    
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
    
    # Tower
    # bin2rin(search_dir=DATA / tower, out_dir=DATA / tower, overwrite=False, num_workers=18)
    # Ground
    bin2rin(search_dir=DATA / ground, out_dir=DATA / ground, overwrite=False, num_workers=18)
    exit()
    # -----------------------------------
    # Process 1 file
    
    filepath = Path('/home/konsch/Documents/5-Repos/gnssvod/data/tower/LOG1_15sec_BINEX/23149/SEPT149a.23.obs')
    testdir = DATA / "test"
    filepattern = {'MOz1_Grnd': str(filepath)}
    outpattern = {'MOz1_Grnd': str(testdir)}
    
    all_columns = ['C1', 'C2', 'C5', 'C6', 'C7', 'C8', 'L1', 'L2', 'L5', 'L6', 'L7', 'L8',
       'P1', 'P2', 'Azimuth', 'Elevation']
    keepvars = ['S?', 'S??']
    result = gv.preprocess(filepattern=filepattern,
                           interval='15s',
                           keepvars=keepvars,
                           outputresult=True,
                           overwrite=True,
                           aux_path=str(AUX),
                           outputdir=outpattern)
    
    # and show data frame
    result['MOz1_Grnd'][0].observation
    result['MOz1_Grnd'][0].observation.columns
    
    result['MOz1_Grnd'][0].observation['S1C']
    #print percentage of NaN values per column
    print(result['MOz1_Grnd'][0].observation.isna().mean() * 100)
    
    # make a plot of nan/non-nan values of all cols
    result['MOz1_Grnd'][0].observation.isna().sum().plot(kind='bar')
    # make a plot of nan/non-nan values of all cols
    from matplotlib import pyplot as plt
    plt.show()
    
    res = result['MOz1_Grnd'][0].observation
    # unfold multi-index
    res.reset_index(inplace=True)
    # make a scatterplot x=datetime, y=col (nan/non-nan)
    
    # Create a DataFrame indicating NaN (1) or non-NaN (0) for each column
    nan_status = res.isna().astype(int)
    
    # Plot each column's NaN status against the datetime
    for col in nan_status.columns:
        plt.scatter(res['datetime'], nan_status[col], label=col, alpha=0.5)
    
    plt.xlabel('Datetime')
    plt.ylabel('NaN Status (1=NaN, 0=Non-NaN)')
    plt.title('Scatterplot of NaN/Non-NaN Values Over Time')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    # -----------------------------------
    
    """
    - 2024 first
    - choose cols to stay (S?)
    
    """
    # batch processing
    
    pattern = {'Dav2_Twr': 'data_RINEX2.11/Dav2_Twr/rinex/*.*O',
               'Dav1_Grnd': 'data_RINEX2.11/Dav1_Grnd/rinex/*.*O'}
    outputdir = {'Dav2_Twr': 'data_RINEX2.11/Dav2_Twr/nc/',
                 'Dav1_Grnd': 'data_RINEX2.11/Dav1_Grnd/nc/'}
    # what variables should be kept
    keepvars = ['S?', 'S??']
    
    gv.preprocess(pattern, interval='15s', keepvars=keepvars, outputdir=outputdir)

if __name__ == '__main__':
    main()