# !/usr/bin/env python
# -*- coding: utf-8 -*-

from definitions import FIG
from analysis.VODProcessor import VODProcessor
from processing.settings import *

if __name__ == "__main__":
    if batch_run:
        print(f"Running in batch mode for {len(time_intervals)} time intervals")
        
        from tqdm import tqdm
        
        for interval in tqdm(time_intervals, desc="Processing intervals"):
            start_date, end_date = interval
            print(f"Processing interval: {start_date} to {end_date}")
            
            # Initialize the processor for this time interval
            processor = VODProcessor(station='MOz', plot=plot, time_interval=interval)
            
            # Process VOD with the current time interval (same for both modes)
            processor.process_vod(local_file=False, overwrite=overwrite_vod_processing)
            
            if iterate_parameters:
                print(f"Iterating over parameter combinations for interval {start_date} to {end_date}")
                for param, values in gnss_parameters_iterative.items():
                    print(f"  - {param}: {values}")
                
                # Process with multiple parameter combinations
                processor.process_anomaly(**gnss_parameters_iterative)
                # No plotting for batch parameter iteration
            else:
                # Process with default parameters for this interval
                processor.process_anomaly(**gnss_parameters)
                
                # Plot results for this interval
                processor.plot_results(
                    gnssband="VOD1",
                    algo="tps",
                    save_dir=FIG,
                    figsize=(7, 4),
                    y_limits={'VOD1': (0.0, 1.2)},
                    time_zone=visualization_timezone,
                )
    
    else:
        # Non-batch processing remains unchanged
        # Initialize the processor for the default case (non-batch)
        processor = VODProcessor(station='MOz', plot=plot, time_interval=single_file_interval)
        
        if not iterate_parameters:
            print("Processing VOD with default parameters")
            
            # Process VOD with default parameters
            processor.process_vod(local_file=False, overwrite=overwrite_vod_processing)
            
            # Run anomaly detection
            processor.process_anomaly(**gnss_parameters)
            
            # Plot the results
            processor.plot_results(
                gnssband="VOD1",
                algo="tps",
                save_dir=FIG,
                figsize=(7, 4),
                y_limits={'VOD1': (0.0, 1.2)},
                time_zone=visualization_timezone,
            )
        
        if iterate_parameters:
            print(f"Iterating over parameter combinations")
            for i, k in gnss_parameters_iterative.items():
                print(f"Processing VOD with {i}={k}")
            
            # Process VOD with multiple parameter combinations
            processor.process_vod(local_file=False, overwrite=overwrite_vod_processing)
            
            # Run anomaly detection
            processor.process_anomaly(**gnss_parameters_iterative)
            
            # Plot by parameter
            zoom_in_interval = ('2024-04-01', '2024-06-15')  # Adjust this as needed
            processor.plot_by_parameter(
                gnssband="VOD1",
                new_interval=zoom_in_interval,
                algo="tps",
                save_dir=FIG,
                figsize=(9, 5),
                y_limits={'VOD1': (0.3, 0.9)},
                time_zone=visualization_timezone,
            )