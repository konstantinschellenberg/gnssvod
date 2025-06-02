# !/usr/bin/env python
# -*- coding: utf-8 -*-

from analysis.VODProcessor import VODProcessor
from definitions import FIG
from processing.settings import *

if __name__ == "__main__":
    # Initialize the processor
    processor = VODProcessor(station='MOz', plot=True)
    
    if not iterate_options:
        print("Processing VOD with default parameters")
        
        # Process VOD with default parameters
        results = processor.process_vod(
            angular_resolutions=[angular_resolution],
            angular_cutoffs=[angular_cutoff],
            temporal_resolutions=[temporal_resolution]
        )
        
        # Access the combined results as an xarray dataset
        print(results)
        
        # Plot the results
        processor.plot_results(
            gnssband="VOD1",
            algo="tps",
            save_dir=FIG,
            figsize=(7, 4),
            y_limits={'VOD1': (0.4, 0.8)}
        )
    
    if iterate_options:
        print(f"Iterating over parameter combinations")
        for i, k in iterate_options_parameters.items():
            print(f"Processing VOD with {i}={k}")
           
        # Process VOD with multiple parameter combinations
        results = processor.process_vod(**iterate_options_parameters)
        
        # Access the combined results as an xarray dataset
        print(results)
        
        # by parameter
        processor.plot_by_parameter(
            gnssband="VOD1",
            algo="tps",
            save_dir=FIG,
            figsize=(10, 5),
            y_limits={'VOD1': (0.3, 0.9)}
        )
