#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd

from definitions import DATA
from processing.settings import time_intervals
from processing.vodreader import VODReader


if __name__ == "__main__":
    

    settings = {
        'station': 'MOz',
        'anomaly_type': 'unknown',  # or 'phi_theta' or 'phi_theta_sv'
    }
    
    year_range = list(time_intervals.keys())
    startyear = year_range[0]
    endyear = year_range[-1]
    
    # -----------------------------------
    # read in all vod data and append rows
    vod_data = time_intervals.copy()
    for year, time_interval in time_intervals.items():
        settings['time_interval'] = time_interval
        print(f"Processing VOD data for {year}...")
        
        # Create reader with automatic file selection
        reader = VODReader(settings)
        
        # Get the data
        vod = reader.get_data(format='long')
        
        if vod is not None:
            # assign to dict[year]
            vod_data[year] = vod
        else:
            print(f"No VOD data found for {year}.")
            
    # -----------------------------------
    print("VOD data exists for:")
    for year, vod in vod_data.items():
        if isinstance(vod, pd.DataFrame):
            print(f"  {year}: {vod.shape[0]} rows")
        else:
            print(f"  {year}: No data found")
            vod_data[year] = None  # Ensure we have a consistent structure
            
    # -----------------------------------
    # creat an outname based on the years with data
    vod_years_with_data = [year for year, vod in vod_data.items() if isinstance(vod, pd.DataFrame)]

    outname = f"combined_vod_data_{settings['station']}_{'_'.join(vod_years_with_data)}.csv"

    # -----------------------------------
    # Concatenate all dataframes into one
    if vod_data:
        vod_data_list = [vod for vod in vod_data.values() if isinstance(vod, pd.DataFrame)]
        vod_combined = pd.concat(vod_data_list, ignore_index=False)
        print(f"Combined VOD data shape: {vod_combined.shape}")
        
        # Save the combined data to a CSV file
        _dir = DATA / 'ard'
        _dir.mkdir(exist_ok=True, parents=True)
        vod_combined.to_csv(_dir / outname, index=True)
        print(f"Combined VOD data saved to {_dir / outname}.")
    else:
        print("No VOD data was collected.")
        
    # -----------------------------------
    # print head
    