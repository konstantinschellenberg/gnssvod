#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd

from definitions import DATA
from processing.settings import time_intervals
from gnssvod.io.vodreader import VODReader

if __name__ == "__main__":
    
    """
    Process VOD data using time intervals from settings.py
    """
    
    settings = {
        'station': 'MOz',
    }
    
    # Extract unique years from intervals for display
    years = set()
    for start_date, end_date in time_intervals:
        start_year = pd.to_datetime(start_date).year
        end_year = pd.to_datetime(end_date).year
        for year in range(start_year, end_year + 1):
            years.add(year)
    
    year_range = sorted(list(years))
    startyear = year_range[0]
    endyear = year_range[-1]
    print(f"Processing data from {startyear} to {endyear}")
    
    # -----------------------------------
    # read in all vod data for each interval
    vod_data = {}
    for interval in time_intervals:
        start_date, end_date = interval
        interval_key = f"{start_date}_to_{end_date}"
        settings['time_interval'] = interval
        print(f"Processing VOD data for interval {interval_key}...")
        
        # Create reader with automatic file selection
        reader = VODReader(settings)
        
        # Get the data
        vod = reader.get_data(format='long')
        
        if vod is not None:
            # assign to dict with interval key
            vod_data[interval_key] = vod
        else:
            print(f"No VOD data found for interval {interval_key}.")
    
    # -----------------------------------
    print("VOD data exists for:")
    for interval_key, vod in vod_data.items():
        if isinstance(vod, pd.DataFrame) and not vod.empty:
            print(f"  {interval_key}: {vod.shape[0]} rows")
        else:
            print(f"  {interval_key}: No data found")
            vod_data[interval_key] = None  # Ensure we have a consistent structure
    
    # -----------------------------------
    # create an outname based on the years with data
    year_range_str = f"{startyear}_to_{endyear}"
    outname = f"combined_vod_data_{settings['station']}_{year_range_str}.csv"
    
    # -----------------------------------
    # Concatenate all dataframes into one
    if vod_data:
        vod_data_list = [vod for vod in vod_data.values() if isinstance(vod, pd.DataFrame) and not vod.empty]
        if vod_data_list:
            vod_combined = pd.concat(vod_data_list, ignore_index=False)
            print(f"Combined VOD data shape: {vod_combined.shape}")
            
            # Check for duplicate timestamps
            # if isinstance(vod_combined.index, pd.MultiIndex):
            #     duplicate_count = vod_combined.index.duplicated().sum()
            #     if duplicate_count > 0:
            #         print(f"Warning: Found {duplicate_count} duplicate timestamps. Keeping first occurrence.")
            #         vod_combined = vod_combined[~vod_combined.index.duplicated(keep='first')]
            
            # Save the combined data to a CSV file
            _dir = DATA / 'ard'
            _dir.mkdir(exist_ok=True, parents=True)
            vod_combined.to_csv(_dir / outname, index=True)
            print(f"Combined VOD data saved to {_dir / outname}.")
        else:
            print("No valid VOD data was found in any interval.")
    else:
        print("No VOD data was collected.")