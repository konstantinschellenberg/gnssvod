#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from pandas.io.sas.sas_constants import sas_date_formats

from definitions import DATA
from processing.inspect_vod_funs import characterize_daily_vod, characterize_weekly_trends, \
    create_optimal_estimator, create_satellite_mask, \
    create_vod_percentile_mask, \
    create_vod_trend, filter_vod_columns, process_diurnal_vod
from processing.metadata import create_vod_metadata
from processing.settings import *
from gnssvod.io.VODReader import VODReader

if __name__ == "__main__":
    
    """
    Process VOD data using time intervals from settings.py
    """
    
    # settings that all datasets need to match
    settings = {
        'station': 'MOz',
        "angular_resolution": angular_resolution,  # must be a list
        "angular_cutoff": angular_cutoff,
        "temporal_resolution": temporal_resolution,
        "search_agg_fun_ts": search_agg_fun_ts,
    }
    
    # -----------------------------------
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
        print("+"*50)
        print(f"Processing VOD data for interval {interval_key}...")
        
        # Create reader with automatic file selection
        reader = VODReader(settings, transform_time=False)
        
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
            
            # todo: Use precipitation data to filter VOD data
            
            # -----------------------------------
            # PREPARATIONS BEFORE MERGE
            
            # -----------------------------------
            # MERGE
            vod_merged = pd.concat(vod_data_list, ignore_index=False)
            print(f"Combined VOD data shape: {vod_merged.shape}")
            
            # add the mean of each subset
            metadata = create_vod_metadata()
            
            # -----------------------------------
            # POSTPROCESSING OF ADDITIONAL METRICS
            
            # filters low-availability data
            total_row = vod_merged.shape[0]
            nsat_mask = create_satellite_mask(vod_merged, min_satellites=minimum_nsat, satellite_col="Ns_t")
            satellite_mask_columns = filter_vod_columns(vod_merged, exclude_metrics=True, exclude_sbas=True)
            
            # apply mask to nan to the cols in satellite_mask_columns but keep all cols
            vod_merged[satellite_mask_columns] = vod_merged[satellite_mask_columns].where(nsat_mask, np.nan)
            # sum of nan values innsat_mask
            non_nan = nsat_mask.sum()
            print(f"Filtered VOD data to {vod_merged.shape[0]} rows with at least {minimum_nsat} satellites.")
            print(
                f"Total NaN values before filtering: {total_row}, after filtering: {non_nan}, filtered out: {total_row - non_nan}")
            print("+" * 50)
            
            # -----------------------------------
            # filter dip-artifacts
            total_row = vod_merged.shape[0]
            percvod_mask = create_vod_percentile_mask(vod_merged, vod_column="VOD1_anom", min_percentile=min_vod_quantile,
                                                      loess_frac=loess_frac)
            vod_percentile_columns1 = filter_vod_columns(vod_merged, column_type='binned anom', exclude_sbas=True,
                                                         is_binned=True)
            vod_percentile_columns2 = filter_vod_columns(vod_merged, column_type='anom', exclude_sbas=True, is_binned=False)
            vod_perc_cols = vod_percentile_columns1 + vod_percentile_columns2
            # apply filter
            vod_merged[vod_perc_cols] = vod_merged[vod_perc_cols].where(percvod_mask, np.nan)
            # sum of nan values in percvod_mask
            total_nan_prior = vod_merged.isna().sum().sum()
            non_nan = percvod_mask.sum()
            print(f"Filtered VOD data to {vod_merged.shape[0]} rows with VOD1_anom >= {min_vod_quantile} quantile.")
            print(f"Total NaN values before filtering: {total_row}, after filtering: {non_nan}, filtered out: {total_row - non_nan}")
            print("+" * 50)
            
            # -----------------------------------
            # HIGHBIOMASS
            # make the mean of VOD1_anom_bin3 and VOD1_anom_bin4
            
            print(f"Calculating high biomass VOD anomalies...")
            vod_merged['VOD1_anom_highbiomass'] = vod_merged[['VOD1_anom_bin2', 'VOD1_anom_bin3', 'VOD1_anom_bin4']].mean(
                axis=1)
            
            # -----------------------------------
            # 1. Characterize precipitation patterns using VOD1_anom
            
            # todo: use wetness data to filter precipitation patterns
            print("Characterizing precipitation patterns...")
            
            """
            - gps+gal
            - top 60% biomass
            
            TODOs now:
            - cut precip from all data
            - better quantile for dips
            - diurnal receives trend! should not
            
            """
            daily = characterize_daily_vod(vod_merged, dataset_col='VOD1_anom_highbiomass', offset_col='VOD1',
                                           precip_quantile=precip_quantile,
                                           min_hours_per_day=12)
            
            daily_old = characterize_precipitation(vod_merged, dataset_col='VOD1_anom_gps+gal', precip_quantile=precip_quantile,
                                                 min_hours_per_day=12)
            
            from matplotlib import pyplot as plt
            daily.plot(title="VOD Daily Mean", ylabel="VOD", xlabel="Time");  plt.show()
            
            trend1 = create_vod_trend(vod_merged, 'VOD1_anom_gps+gal')
            trend2 = create_vod_trend(vod_merged, 'VOD2_anom_gps+gal')
            
            print("Characterizing weekly trends...")
            weekly = characterize_weekly_trends(vod_merged, sbas_bands=['VOD1_S33', 'VOD1_S35'], detrend=detrend_weekly)
            
            print("Processing diurnal VOD patterns...")
            diurnal = process_diurnal_vod(vod_merged, diurnal_col='VOD1_anom_highbiomass',
                                          window_hours=6, polyorder=2, apply_loess=True,
                                          loess_frac=0.1)
            
            print("Creating optimal VOD estimator...")
            # Combine the results
            optimal_vod = "VOD_optimal_zscore"
            vod_optimal = create_optimal_estimator(
                pd.DataFrame({
                    'VOD1_daily': trend1,
                    'VOD1_S_weekly': weekly['VOD1_S_weekly'],
                    'VOD1_diurnal': diurnal['VOD1_diurnal']
                }, index=vod_merged.index)
            )
            
            # adding "best" parameter from iterative arithmetics search (spearman correlation against branch water potential)
            # VOD1_daily + VOD2_ke_anom
            
            # todo: doesn't work anymore after VOD1_daily contains nan values
            # vod_optimal["VOD_bestspearman"] = precips['VOD1_daily'] + vod_merged['VOD2_ke_anom']
            
            # Add the optimal estimator back to the original dataframe
            vod_optimal['VOD_optimal'] = vod_optimal[optimal_vod]
            
            # join on index with vod_ts
            intermediate_steps = pd.DataFrame({
                'VOD1_daily': daily['VOD1_daily'],
                'VOD1_S_weekly': weekly['VOD1_S_weekly'],
                'VOD1_diurnal': diurnal['VOD1_diurnal']
            }, index=vod_merged.index
            )
            
            print("Joining VOD data...")
            
            # Only join columns that don't already exist in vod_ts
            vod_merged = vod_merged.join(vod_optimal[[col for col in vod_optimal.columns if col not in vod_merged.columns]])
            vod_merged = vod_merged.join(daily[[col for col in daily.columns if col not in vod_merged.columns]])
            vod_merged = vod_merged.join(weekly[[col for col in weekly.columns if col not in vod_merged.columns]])
            vod_merged = vod_merged.join(diurnal[[col for col in diurnal.columns if col not in vod_merged.columns]])
            vod_merged = pd.concat([vod_merged, trend1, trend2], axis=1)
            
            # sort columns by alphanumeric order
            vod_merged = vod_merged.reindex(sorted(vod_merged.columns), axis=1)
            
            # set nan to all values < -100
            vod_merged[vod_merged < -100] = np.nan
            
            # -----------------------------------
            # last checks
            # make sure the datetime index is monotonic and without gaps
            
            # filter anomalies
            if filter_anomalies:
                print("Filtering specific anomaly periods...")
                
                # Create a mask for rows to keep (outside anomaly periods)
                mask = pd.Series(True, index=vod_merged.index)
                for start, end in period_anomaly:
                    period_mask = (vod_merged.index >= start) & (vod_merged.index <= end)
                    mask = mask & ~period_mask
                
                # set the mask to NaN where the mask is False
                vod_merged.loc[~mask, vod_merged.columns] = np.nan
                print(f"Filtered out {(~mask).sum()} rows in anomaly periods.")
                
                
            # Ensure datetime index is monotonic and without gaps
            print("Ensuring datetime index is monotonic and without gaps...")
            
            # First check if index is datetime type
            if not isinstance(vod_merged.index, pd.DatetimeIndex):
                print("Converting index to DatetimeIndex...")
                vod_merged.index = pd.to_datetime(vod_merged.index)
            
            # Check if index is monotonic
            if not vod_merged.index.is_monotonic_increasing:
                print("Index is not monotonic, sorting...")
                vod_merged = vod_merged.sort_index()
            
            # Try to infer frequency from the sorted index
            inferred_freq = pd.infer_freq(vod_merged.index)
            
            if inferred_freq is None:
                # If inference fails, calculate the most common time difference
                diff = vod_merged.index.to_series().diff().dropna()
                most_common_diff = diff.value_counts().index[0]
                
                # Convert timedelta to a pandas frequency string
                seconds = most_common_diff.total_seconds()
                if seconds % 86400 == 0:  # Daily
                    inferred_freq = f"{int(seconds / 86400)}D"
                elif seconds % 3600 == 0:  # Hourly
                    inferred_freq = f"{int(seconds / 3600)}h"
                elif seconds % 60 == 0:  # Minutely
                    inferred_freq = f"{int(seconds / 60)}min"
                else:  # Secondly
                    inferred_freq = f"{int(seconds)}S"
                
                print(f"Frequency inference failed. Using most common time difference: {inferred_freq}")
            else:
                print(f"Inferred frequency: {inferred_freq}")
            
            # Create a complete datetime range
            start_time = vod_merged.index.min()
            end_time = vod_merged.index.max()
            
            try:
                complete_idx = pd.date_range(start=start_time, end=end_time, freq=inferred_freq)
                
                # Check if there are gaps
                missing_dates = complete_idx.difference(vod_merged.index)
                if len(missing_dates) > 0:
                    print(f"Found {len(missing_dates)} gaps in the datetime index. Reindexing...")
                    
                    # Reindex with the complete range, filling gaps with NaN
                    vod_merged = vod_merged.reindex(complete_idx)
                    
                    # Print some stats about the gaps
                    print(f"Added NaN values for {len(missing_dates)} missing timestamps.")
                else:
                    print("No gaps found in the datetime index.")
            except Exception as e:
                print(f"Error creating complete date range with frequency {inferred_freq}: {e}")
                print("Skipping gap filling, but index is still sorted.")

                
            # -----------------------------------
            print("Saving combined VOD data...")
            
            # Save the combined data to a CSV file
            _dir = DATA / 'ard'
            _dir.mkdir(exist_ok=True, parents=True)
            
            # save also as pyarrow
            vod_merged.to_parquet(_dir / outname.replace('.csv', '.parquet'), index=True)
            # vod_merged.to_csv(_dir / outname, index=True)
            
            print(f"Combined VOD data saved to {_dir / outname}.")
        else:
            print("No valid VOD data was found in any interval.")
    else:
        print("No VOD data was collected.")