#!/usr/bin/env python
# -*- coding: utf-8 -*-
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from dask.distributed import Client

from definitions import DATA
import gnssvod as gv
from processing.filepattern_finder import create_time_filter_patterns
from processing.settings import bands, station
from gnssvod.analysis.calculate_anomalies import calculate_anomaly

class VODProcessor:
    def __init__(self, station=station, bands=bands, time_interval=None,
                 pairings=None, recover_snr=True, plot=False, overwrite=False):
        """
        Initialize the VOD processor.

        Parameters
        ----------
        station : str
            Station identifier
        bands : dict
            Dictionary mapping band names to frequencies
        time_interval : tuple
            Start and end date for data processing
        pairings : dict, optional
            Dictionary mapping stations to (tower, ground) pairs
        recover_snr : bool, default=True
            Whether to recover SNR values
        plot : bool, default=False
            Whether to generate plots
        overwrite : bool, default=False
            Whether to overwrite existing files
        """
        self.vod_processing_params = None
        self.vod_filename = None
        self.station = station
        self.bands = bands
        self.band_ids = list(bands.keys())
        self.time_interval = time_interval
        self.plot = plot
        self.overwrite = overwrite
        self.recover_snr = recover_snr
        
        self._init_interval()

        # Define station pairings if not provided
        if pairings is None:
            self.pairings = {station: (f"{station}1_Twr", f"{station}1_Grnd")}
        else:
            self.pairings = pairings
        
        # Create pattern for file search
        self.pattern = str(DATA / "gather" / f"{create_time_filter_patterns(time_interval)['nc']}")
        
        # Initialize data containers
        self.vod = None
        self.hemi = None
        self.patches = None
        self.results = None
        
        
    def _init_interval(self):
        # self.time_interval must be a tuple of strings (start_date, end_date)
        if not isinstance(self.time_interval, tuple) or len(self.time_interval) != 2:
            raise ValueError(f"time_interval must be a tuple of two strings (start_date, end_date) but is {self.time_interval}")
        # Ensure both dates are strings in the format 'YYYY-MM-DD'
        for date in self.time_interval:
            if not isinstance(date, str) or not pd.to_datetime(date, errors='coerce'):
                raise ValueError("Both start_date and end_date must be valid date strings in 'YYYY-MM-DD' format.")
        
    def _concat_nc_files(self, files):
        pass
    
    def process_vod(self, local_file=False, overwrite=False, **kwargs):
        """
        Process VOD data and save to disk in temp folder to avoid memory overflow.

        Parameters
        ----------
        local_file : bool, default=False
            Whether to return the processed VOD data
        overwrite : bool, default=False
            Whether to overwrite existing files
        **kwargs : dict
            Additional keyword arguments, such as start_date and end_date

        Returns
        -------
        pandas.DataFrame or None
            VOD data if local_file=True, otherwise None
        """
        # Store processing parameters for metadata
        self.vod_processing_params = {
            "local_file": local_file,
            "overwrite": overwrite
        }
        self.vod_processing_params.update(kwargs)
        
        print(f"Processing VOD data for {self.station}")
        
        # Create a unique filename based on parameters
        start_date = pd.to_datetime(self.time_interval[0]).strftime('%Y%m%d')
        end_date = pd.to_datetime(self.time_interval[1]).strftime('%Y%m%d')
        # Create abbreviations for parameters
        band_abbr = f"bd{len(self.bands)}"  # Number of bands
        snr_abbr = f"rs{int(self.recover_snr)}"  # SNR recovery (1=True, 0=False)
        # Create filename with parameter abbreviations
        self.vod_filename = f"vod_{self.station}_{start_date}_{end_date}_{band_abbr}_{snr_abbr}.pkl"
        
        # Create temp directory if it doesn't exist
        temp_dir = DATA / 'temp'
        temp_dir.mkdir(exist_ok=True, parents=True)
        
        vod_file_path = temp_dir / self.vod_filename
        
        # Check if file already exists and overwrite flag
        if vod_file_path.exists() and not overwrite:
            print(f"VOD data file already exists at {vod_file_path}. Loading from disk.")
            # Don't load into memory yet, just confirm it exists
            if local_file:
                return pd.read_pickle(vod_file_path)
            return None
        
        # -----------------------------------
        # concat all files
        
        self._concat_nc_files()
        
        # Calculate VOD
        vod = gv.calc_vod(
            self.pattern,
            self.pairings,
            self.bands,
            self.time_interval,
            recover_snr=self.recover_snr
        )[self.station]
        
        # print("NaN values in VOD:")
        # print(vod.isna().mean() * 100)
        
        # Save to disk
        if not vod.empty:
            vod.to_pickle(vod_file_path)
            
            print(f"Saved VOD data to {vod_file_path}")
            if local_file:
                return vod
        else:
            print("No VOD data found for the specified time interval.")
            return None
        
        return None
    
    def process_anomaly(self, angular_resolution, angular_cutoff, temporal_resolution, max_workers=15, overwrite=False,
                        **kwargs):
        """
        Calculate VOD anomalies with multiple parameter combinations in parallel.
        Uses saved files if they exist and overwrite=False.

        Parameters
        ----------
        angular_resolution : list or float
            Angular resolution values to test
        angular_cutoff : list or float
            Angular cutoff values to test
        temporal_resolution : list or int
            Temporal resolution values to test
        max_workers : int, optional
            Maximum number of parallel workers
        overwrite : bool, default=False
            Whether to overwrite existing results files
        **kwargs : dict
            Additional keyword arguments

        Returns
        -------
        xarray.Dataset
            Combined results from all parameter combinations
        """
        
        # Transform all parameters to lists if they are not already
        if not isinstance(angular_resolution, list):
            angular_resolution = [angular_resolution]
        if not isinstance(angular_cutoff, list):
            angular_cutoff = [angular_cutoff]
        if not isinstance(temporal_resolution, list):
            temporal_resolution = [temporal_resolution]
            
        # Store anomaly parameters for metadata
        self.anomaly_params = {
            "angular_resolution": angular_resolution,
            "angular_cutoff": angular_cutoff,
            "temporal_resolution": temporal_resolution,
            "max_workers": max_workers,
            "overwrite": overwrite
        }
        self.anomaly_params.update(kwargs)

        print(
            f"Processing VOD data for {self.station} with {len(angular_resolution) * len(angular_cutoff) * len(temporal_resolution)} parameter combinations")
        
        # Check if VOD data has been processed
        if not hasattr(self, 'vod_filename'):
            print("VOD data has not been processed. Running process_vod first.")
            self.process_vod(overwrite=self.overwrite)
        
        # Create temp directory if it doesn't exist
        temp_dir = DATA / 'temp'
        temp_dir.mkdir(exist_ok=True, parents=True)
        
        # Check for existing results files and load them if available
        result_paths = []
        param_combinations_to_process = []
        
        start_date = pd.to_datetime(self.time_interval[0]).strftime('%Y%m%d')
        end_date = pd.to_datetime(self.time_interval[1]).strftime('%Y%m%d')
        time_abbr = start_date if start_date == end_date else f"{start_date}_{end_date}"
        
        for ar in angular_resolution:
            for ac in angular_cutoff:
                for tr in temporal_resolution:
                    # Create filename for this parameter combination
                    # Extract year and month from time interval for compact representation
                    filename = f"vod_anomaly_{self.station}_{time_abbr}_ar{ar}_ac{ac}_tr{tr}.pkl"
                    file_path = temp_dir / filename
                    # Check if file exists and overwrite flag
                    if file_path.exists() and not overwrite:
                        print(f"Found existing results for AR={ar}, AC={ac}, TR={tr}. Loading from disk.")
                        result_paths.append(str(file_path))
                    else:
                        # Process this parameter combination immediately
                        print(f"Processing combination: AR={ar}, AC={ac}, TR={tr}")
                        params = {
                            'angular_resolution': ar,
                            'angular_cutoff': ac,
                            'temporal_resolution': tr,
                            'vod_anomaly_filename': filename,
                        }
                        # Update with kwargs
                        params.update(kwargs)
                        
                        # Process the parameter combination directly
                        result = self._run_parameter_combination(params)
                        if result is not None:
                            result_paths.append(result)
        
        # Combine results into a single xarray dataset
        self.results = self._combine_results(result_paths)
        
        # Save as VOD timeseries
        if self.results is not None:
            self._save_vod_timeseries(
                filename=f"vod_timeseries_{self.station}",
                overwrite=self.overwrite,
            )
        
        return self.results
    
    def _save_vod_timeseries(self, filename, overwrite=False):
        """
        Save VOD time series data to a NetCDF file and metadata to a JSON file.

        Parameters
        ----------
        vod_ts : pandas.DataFrame
            VOD time series data with 'Epoch' in the index
        filename : str
            Filename to save (without extension)
        overwrite : bool, optional
            Whether to overwrite existing files

        Returns
        -------
        path : Path
            Path to the saved file or None if save failed
        """
        import os
        
        # Set default directory if not specified
        directory = DATA / "timeseries"
        directory.mkdir(parents=True, exist_ok=True)
        
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        try:
            # Create a copy of the dataframe to avoid modifying the original
            df = self.results.to_dataframe().reset_index().copy()
            
            # Rename 'Epoch' to 'datetime'
            df = df.rename(columns={'Epoch': 'datetime'})
            
            # Set 'datetime' as the index again
            df = df.set_index('datetime')

            metadata = self._create_metadata()
            
            start, end = pd.to_datetime(self.time_interval[0]), pd.to_datetime(self.time_interval[1])
            
            outname = f"{filename}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}_{metadata["metadata_id"]}"
            
            # Add .nc extension if not already present
            if not outname.endswith('.nc'):
                outname = f"{outname}.nc"
            
            # Full file path
            file_path = directory / outname
            
            # Convert to xarray dataset for NetCDF export
            ds = df.to_xarray()
            
            # Check if file exists and overwrite flag
            if file_path.exists() and not overwrite:
                print(f"File {file_path} already exists. Pass overwrite=True to overwrite.")
                return file_path
            
            # Save to NetCDF
            ds.to_netcdf(file_path)
            
            # Save metadata
            self._save_metadata(metadata, file_path)
            
            print(f"Saved VOD time series to {file_path}")
            return file_path
        
        except Exception as e:
            print(f"Error saving file {file_path}: {str(e)}")
            return None
    
    def _save_metadata(self, metadata, file_path):
        """
        Save metadata to a JSON file.

        Parameters
        ----------
        metadata : dict
            Dictionary containing metadata
        file_path : str or Path
            File path to save metadata to (without extension)

        Returns
        -------
        path : Path
            Path to the saved metadata file
        """
        import json
        from pathlib import Path
        
        # Add .json extension
        metadata_path = Path(str(file_path).replace('.nc', '_metadata.json'))
        
        try:
            # Save metadata to JSON file
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4, default=str)
            
            print(f"Saved metadata to {metadata_path}")
            return metadata_path
        
        except Exception as e:
            print(f"Error saving metadata file {metadata_path}: {str(e)}")
            return None
    
    def _create_metadata(self):
        """
        Create metadata dictionary with all essential settings.

        Returns
        -------
        dict
            Dictionary containing metadata about the VOD processing
        """
        # Create basic metadata dictionary
        metadata = {
            "processing_timestamp": pd.Timestamp.now().isoformat(),
            "station": self.station,
            "time_interval": {
                "start_date": pd.to_datetime(self.time_interval[0]).isoformat(),
                "end_date": pd.to_datetime(self.time_interval[1]).isoformat()
            },
            "file_info": {
                "vod_filename": self.vod_filename,
                "pattern": self.pattern,
            },
            "initialization": {
                "bands": {k: v for k, v in self.bands.items()},  # Convert to dict if needed
                "band_ids": self.band_ids,
                "pairings": self.pairings,
                "recover_snr": self.recover_snr,
                "overwrite": self.overwrite,
                "plot": self.plot,
            }
        }
        
        # Add processing parameters if they exist
        if hasattr(self, 'vod_processing_params'):
            metadata["vod_processing"] = self.vod_processing_params
        
        # Add anomaly parameters if they exist
        if hasattr(self, 'anomaly_params'):
            metadata["anomaly_processing"] = self.anomaly_params
            # add a key "multi_parameter" to indicate that multiple parameter combinations were processed
            # check if any of the parameters are lists of len > 1
            multi_parameter = any(isinstance(v, list) and len(v) > 1 for v in self.anomaly_params.values())
            metadata["anomaly_processing"]["multi_parameter"] = multi_parameter
        else:
            # If processing hasn't run, but we have the parameters as attributes
            anomaly_params = {}
            for param in ['angular_resolution', 'angular_cutoff', 'temporal_resolution']:
                if hasattr(self, param):
                    anomaly_params[param] = getattr(self, param)
            
            if anomaly_params:
                metadata["anomaly_processing"] = anomaly_params
        
        # Add results information if available
        if hasattr(self, 'results') and self.results is not None:
            # Just add info about the results, not the full data
            metadata["results"] = {
                "variables": list(self.results.variables),
                "dimensions": {dim: len(self.results[dim]) for dim in self.results.dims},
                "coordinates": {coord: list(self.results[coord].values)
                                for coord in self.results.coords
                                if len(self.results[coord].values) < 100}  # Only include small coordinate lists
            }
            
        # make a hash for the metadata
        metadata["metadata_id"] = f"{hash(frozenset(str(metadata.items()))):x}"
        
        return metadata
    
    def _run_parameter_combination(self, params):
        """
        Process a single parameter combination and save results to disk.
        """
        try:
            print(f"Processing combination: {params}")
            
            # Load VOD data from disk for this process
            temp_dir = DATA / 'temp'
            anomaly_file_path = temp_dir / params['vod_anomaly_filename']
            vod_file_path = temp_dir / self.vod_filename
            
            # Load VOD data
            vod = pd.read_pickle(vod_file_path)
            
            # Build hemispheric grid
            hemi = gv.hemibuild(params['angular_resolution'], params['angular_cutoff'])
            
            # Classify VOD into grid cells
            vod_with_cells = hemi.add_CellID(vod.copy())
            
            # Calculate anomalies using memory-efficient approach
            anomaly_params = {
                "vod_with_cells": vod_with_cells,  # Pass pandas DataFrame directly
                "band_ids": self.band_ids,
                "time_chunk_size": 500,  # Process 500 time steps at once
                "sv_batch_size": 5,  # Process 5 satellites at once
                "memory_limit": "4GB"  # Limit memory usage
            }
            # Update with params
            anomaly_params.update(params)
            
            # Calculate anomalies (now accepts pandas DataFrame directly)
            vod_ts_combined, vod_avg = calculate_anomaly(**anomaly_params)
            
            # Add parameter columns
            vod_ts_combined['angular_resolution'] = params['angular_resolution']
            vod_ts_combined['angular_cutoff'] = params['angular_cutoff']
            vod_ts_combined['temporal_resolution'] = params['temporal_resolution']
            
            # Save to disk
            vod_ts_combined.to_pickle(anomaly_file_path)
            print(f"Saved anomaly results to {anomaly_file_path}")
            
            return str(anomaly_file_path)
        
        except Exception as e:
            print(f"Error processing combination {params}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _combine_results(self, result_paths):
        """
        Combine results from different parameter combinations into an xarray dataset,
        loading each from disk to save memory.

        Parameters
        ----------
        result_paths : list
            List of paths to temporary result files

        Returns
        -------
        xarray.Dataset
            Combined results
        """
        if not result_paths:
            return None
        
        # Create empty list to hold dataframes
        dfs = []
        
        # Load each result file and append to list
        for path in result_paths:
            if path is not None:
                try:
                    df = pd.read_pickle(path)
                    dfs.append(df)
                    
                    # Remove temporary file after loading
                    # Path(path).unlink(missing_ok=True)
                except Exception as e:
                    print(f"Error loading results from {path}: {str(e)}")
        
        if not dfs:
            return None
        
        # Combine all DataFrames
        combined_df = pd.concat(dfs)
        
        # Convert to xarray
        ds = combined_df.to_xarray()
        
        return ds
    
    def plot_results(self, gnssband="VOD1para", algo="tps", figsize=(12, 10), save_dir=None, show=True,
                     time_zone='etc/GMT+6', y_limits=None):
        """
        Plot results for all parameter combinations, showing time series and diurnal cycles.

        Parameters
        ----------
        figsize : tuple, default=(12, 10)
            Figure size in inches
        save_dir : str or Path, optional
            Directory to save plots. If None, plots won't be saved
        show : bool, default=True
            Whether to display plots
        time_zone : str, default='etc/GMT+6'
            Time zone for diurnal cycle calculation
        y_limits : dict, optional
            Dictionary mapping variable names to (min, max) tuples for y-axis limits
        """
        if self.results is None:
            print("No results available. Run process_vod first.")
            return
        
        # Convert xarray dataset to dataframe for easier manipulation
        df = self.results.to_dataframe().reset_index()
        
        # Filter columns that contain the algo value in their name
        algo_cols = [col for col in df.columns if algo in col]
        df_tps = df[algo_cols]
        # strip suffix from column names
        df_tps.columns = [col.replace(f"_{algo}", "") for col in df_tps.columns]
        # merge back to df
        df_tps = df.merge(df_tps, left_index=True, right_index=True)
        
        if df_tps.empty:
            print(f"No results with algo='{algo} found.")
            return
        
        # Get unique parameter combinations
        param_cols = ['angular_resolution', 'angular_cutoff', 'temporal_resolution']
        param_combinations = df_tps[param_cols].drop_duplicates()
        
        # Create save directory if needed
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True, parents=True)
        
        # Get the bands to plot (those ending with '_anom')
        band_cols = [col for col in df_tps.columns if col.endswith('_anom')]
        
        # Convert datetime to timezone-aware for diurnal cycle
        df_tps = df_tps.copy()  # Create an explicit copy to avoid SettingWithCopyWarning
        df_tps['datetime'] = pd.to_datetime(df_tps['Epoch'])
        df_tps['datetime_tz'] = df_tps['datetime'].dt.tz_localize('UTC').dt.tz_convert(time_zone)
        df_tps['hod'] = df_tps['datetime_tz'].dt.hour + df_tps['datetime_tz'].dt.minute / 60
        
        # Plot each parameter combination
        for _, params in param_combinations.iterrows():
            ar = params['angular_resolution']
            ac = params['angular_cutoff']
            tr = params['temporal_resolution']
            
            # Filter data for this parameter combination
            param_data = df_tps[
                (df_tps['angular_resolution'] == ar) &
                (df_tps['angular_cutoff'] == ac) &
                (df_tps['temporal_resolution'] == tr)
                ]
            
            # Create figure with subplots
            fig, axes = plt.subplots(1, 2, figsize=figsize,
                                     gridspec_kw={'width_ratios': [2, 1]})
            
            # Set the title with parameter information
            fig.suptitle(f"VOD Analysis - {self.station}\n"
                         f"AR: {ar}°, AC: {ac}°, TR: {tr}min, Algorithm: tps",
                         fontsize=14, y=0.98)
            
            # 1. Time series plot (2/3 of figure height)

            
            # select bands based on the provided band (list provided in settings)
            if isinstance(gnssband, str):
                band_cols = [f"{gnssband}_anom"]
            elif isinstance(gnssband, list):
                band_cols = [f"{b}_anom" for b in gnssband]
            
            for band in band_cols:
                param_data.plot(x='datetime', y=band, ax=axes[0], label=band)
            
            axes[0].set_xlabel('Date')
            axes[0].set_ylabel('VOD Anomaly')
            axes[0].set_title('Time Series')
            axes[0].grid(True, linestyle='--', alpha=0.7)
            axes[0].legend()
            
            # Apply y-limits if provided
            if y_limits is not None:
                for band in band_cols:
                    band_name = band.split('_')[0]
                    if band_name in y_limits:
                        axes[0].set_ylim(y_limits[band_name])
            
            # 2. Diurnal cycle plot (1/3 of figure height)
            # Group by hour of day and calculate mean
            diurnal_data = param_data.groupby('hod')[band_cols].mean()
            
            # Plot diurnal cycle
            for band in band_cols:
                axes[1].plot(diurnal_data.index, diurnal_data[band], label=band)
            
            axes[1].set_xlabel('Hour of Day (Local Time)')
            axes[1].set_ylabel('VOD Anomaly')
            axes[1].set_title('Diurnal Cycle')
            axes[1].set_xlim(0, 24)
            axes[1].grid(True, linestyle='--', alpha=0.7)
            
            # Apply y-limits if provided
            if y_limits is not None:
                for band in band_cols:
                    band_name = band.split('_')[0]
                    if band_name in y_limits:
                        axes[1].set_ylim(y_limits[band_name])
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure if requested
            if save_dir is not None:
                filename = f"vod_{self.station}_ar{ar}_ac{ac}_tr{tr}.png"
                plt.savefig(save_dir / filename, dpi=300, bbox_inches='tight')
            
            # Show or close figure
            if show:
                plt.show()
            else:
                plt.close()
        
        print(f"Generated plots for {len(param_combinations)} parameter combinations.")
    
    def plot_by_parameter(self, new_interval=None, gnssband="VOD1", algo="tps", figsize=(12, 6), save_dir=None, show=True,
                          time_zone='etc/GMT+6', y_limits=None):
        """
        Plot results grouped by parameter type, showing how different parameter values
        affect the time series and diurnal cycles using pandas plotting.

        Parameters
        ----------
        gnssband : str or list, default="VOD1"
            GNSS band(s) to plot
        figsize : tuple, default=(12, 6)
            Figure size in inches
        save_dir : str or Path, optional
            Directory to save plots. If None, plots won't be saved
        show : bool, default=True
            Whether to display plots
        time_zone : str, default='etc/GMT+6'
            Time zone for diurnal cycle calculation
        y_limits : dict, optional
            Dictionary mapping variable names to (min, max) tuples for y-axis limits
        """
        
        print(f"Plotting results for {self.station} with algo='{algo}' and gnssband='{gnssband}'")
        
        if self.results is None:
            print("No results available. Run process_vod first.")
            return
        
        # Convert xarray dataset to dataframe for easier manipulation
        df = self.results.to_dataframe().reset_index()
        
        # Filter to only include algo='tps'
        df_tps = df[df['algo'] == algo]
        
        if df_tps.empty:
            print(f"No results with algo={algo} found.")
            return
        
        # Create save directory if needed
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True, parents=True)
        
        # Select bands based on the provided parameter
        if isinstance(gnssband, str):
            band_cols = [f"{gnssband}_anom"]
        elif isinstance(gnssband, list):
            band_cols = [f"{b}_anom" for b in gnssband]
        else:
            print("Invalid gnssband parameter. Please provide a string or list of strings.")
            return
        
        if new_interval:
            # in UTC
            # Update time interval if provided
            print(f"Updating time interval to {new_interval}")
            start, end = pd.to_datetime(new_interval[0]), pd.to_datetime(new_interval[1])
            df_tps = df_tps[(df_tps['Epoch'] >= start) & (df_tps['Epoch'] <= end)]
            
        # Convert datetime to timezone-aware for diurnal cycle
        df_tps['datetime'] = pd.to_datetime(df_tps['Epoch'])
        df_tps['datetime_tz'] = df_tps['datetime'].dt.tz_localize('UTC').dt.tz_convert(time_zone)
        df_tps['hod'] = df_tps['datetime_tz'].dt.hour + df_tps['datetime_tz'].dt.minute / 60
        
        # Define the parameters to analyze and their default values
        parameters = ['angular_resolution', 'angular_cutoff', 'temporal_resolution']
        default_values = {
            'angular_resolution': 0.5,
            'angular_cutoff': 10,
            'temporal_resolution': 60
        }

        # Create plots for each parameter
        for param in parameters:
            # Get unique values for this parameter
            unique_values = sorted(df_tps[param].unique())
            
            if len(unique_values) <= 1:
                print(f"Only one value for {param}, skipping this parameter plot.")
                continue
            
            # Create a viridis colormap based on the number of unique values
            cmap = plt.cm.get_cmap('viridis', len(unique_values))
            colors = cmap(np.linspace(0, 1, len(unique_values)))
            
            # Create figure with subplots
            fig, axes = plt.subplots(1, 2, figsize=figsize,
                                     gridspec_kw={'width_ratios': [2, 1]})
            
            # Filter data to use default values for other parameters
            fixed_params = {p: default_values[p] for p in parameters if p != param}
            filtered_df = df_tps.copy()
            for p, val in fixed_params.items():
                filtered_df = filtered_df[filtered_df[p] == val]
            
            # Get the title with all parameter information
            param_info = "\n".join([f"{p} = {v}" for p, v in fixed_params.items()])
            
            # Set the title with parameter information
            fig.suptitle(f"VOD Analysis - {self.station}\n"
                         f"Effect of {param} on VOD anomalies (Algorithm: tps)\n"
                         f"Fixed parameters:\n{param_info}",
                         fontsize=14, y=0.98)
            
            # For each unique parameter value, create grouped data and plot
            for i, value in enumerate(unique_values):
                # Filter data for this parameter value
                param_data = filtered_df[filtered_df[param] == value].copy()
                
                # Skip if no data
                if param_data.empty:
                    continue
                
                # Tag data with parameter value for better legends
                param_data['param_value'] = f"{param}={value}"
                
                # Time series plot using pandas
                for band in band_cols:
                    # Group by datetime to get mean values
                    ts_data = param_data.groupby('datetime')[band].mean().reset_index()
                    ts_data.plot(x='datetime', y=band, ax=axes[0],
                                 label=f"{param}={value}", color=colors[i], linewidth=0.6)
                
                # Diurnal cycle plot using pandas
                for band in band_cols:
                    # Group by hour of day to get mean values
                    diurnal_data = param_data.groupby('hod')[band].mean().reset_index()
                    diurnal_data.plot(x='hod', y=band, ax=axes[1],
                                      label=f"{param}={value}", color=colors[i], linewidth=1.5)
            
            # Configure time series plot
            axes[0].set_xlabel('Date')
            axes[0].set_ylabel(f'{gnssband} Anomaly')
            axes[0].set_title('Time Series')
            axes[0].grid(True, linestyle='--', alpha=0.7)
            axes[0].legend(loc='upper left')
            
            # Configure diurnal cycle plot
            axes[1].set_xlabel('Hour of Day (Local Time)')
            axes[1].set_ylabel(f'{gnssband} Anomaly')
            axes[1].set_title('Diurnal Cycle')
            axes[1].set_xlim(0, 24)
            axes[1].grid(True, linestyle='--', alpha=0.7)
            # remove  legend from axes[1]
            axes[1].legend().remove()
            
            # Apply y-limits if provided
            if y_limits is not None and gnssband in y_limits:
                axes[0].set_ylim(y_limits[gnssband])
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure if requested
            if save_dir is not None:
                filename = f"vod_{self.station}_{param}_comparison.png"
                plt.savefig(save_dir / filename, dpi=300, bbox_inches='tight')
            
            # Show or close figure
            if show:
                plt.show()
            else:
                plt.close()
        
        print(f"Generated parameter comparison plots.")