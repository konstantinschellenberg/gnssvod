#!/usr/bin/env python
# -*- coding: utf-8 -*-

from definitions import FIG
from analysis.VODProcessor import VODProcessor
from processing.helper import print_color
from processing.settings import *
from tqdm import tqdm


def process_single_interval(interval, iterate_params=False):
    """Process a single time interval with given parameters"""
    start_date, end_date = interval
    print("" + "="*50)
    print_color(f"Processing interval: {start_date} to {end_date}")
    
    gnss_parameters_merged = gnss_parameters.copy()
    gnss_parameters_merged.update(gnss_parameters_iteratable)
    
    processor = VODProcessor(station=station, plot=plot_results, time_interval=interval)
    processor.process_vod(local_file=output_results_locally, overwrite=overwrite_vod_processing)
    processor.process_anomaly(**gnss_parameters_merged)
    
    if not iterate_params and plot_results:
        processor.plot_results(
            gnssband="VOD1", algo="tps", save_dir=FIG, figsize=(7, 4),
            y_limits={'VOD1': (0., 0.7)}, time_zone=visualization_timezone
        )
    elif not batch_run and plot_results:
        zoom_interval = ('2024-04-01', '2024-06-15')
        processor.plot_by_parameter(
            gnssband="VOD1", new_interval=zoom_interval, algo="tps",
            save_dir=FIG, figsize=(9, 5), y_limits={'VOD1': (0.3, 0.9)},
            time_zone=visualization_timezone
        )
        
    # delete processor to free memory
    del processor


if __name__ == "__main__":
    if batch_run:
        print_color(f"Running in batch mode for {len(time_intervals)} time intervals")
        for interval in tqdm(time_intervals, desc="Processing intervals"):
            if iterate_parameters:
                print(f"Iterating parameters for {interval[0]} to {interval[1]}")
                for param, values in gnss_parameters_iteratable.items():
                    print(f"  - {param}: {values}")
            process_single_interval(interval, iterate_parameters)
    else:
        process_single_interval(single_file_interval, iterate_parameters)