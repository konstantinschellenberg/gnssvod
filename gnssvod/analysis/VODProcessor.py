#!/usr/bin/env python
# -*- coding: utf-8 -*-
from concurrent.futures import ProcessPoolExecutor
from datetime import time
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from dask.distributed import Client
from tqdm import tqdm
import os
import xarray as xr

from definitions import DATA
import gnssvod as gv
from gnssvod.io.preprocess import get_filelist
from processing.filepattern_finder import create_time_filter_patterns, filter_files_by_date
from processing.helper import print_color
from processing.settings import bands, SBAS_IDENT, station, moflux_coordinates, add_sbas_position_manually
from analysis.calculate_anomalies import _calculate_anomaly


# Process files individually and concatenate the DataFrames
def process_file(file_path):
    try:
        # Verify file exists and has nonzero size
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            print(f"File {file_path} doesn't exist or is empty")
            return pd.DataFrame()
        
        # Open single file
        ds = xr.open_dataset(file_path)
        df = ds.to_dataframe()
        ds.close()  # Explicitly close dataset
        return df.reset_index().sort_values(by=['Station', 'Epoch', 'SV']).set_index(['Epoch'])
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return pd.DataFrame()


def _replace_sbas_coordinates(vod, sbas_ident):
    """
    Replace Azimuth and Elevation values for SBAS satellites with fixed values.

    Add the positions of the geostationary satellites
    - sbas: ['S58', 'S35', 'S33', 'S31', 'S48']
      probably corresponds to the following:
      135, 133, 131, 148, 158
    - Azi/Ele are in 0.1 resolution
    
    Tasks:
    - Find the corresponding PRN for check databases for sbas
    - Add the positions of the geostationary satellites based on the MOFLUX location
    - `moflux_coordinates`
    
    sbas_ident:
    {'S31': {'system': 'WAAS', 'Azimuth': 216.5, 'Elevation': 31.1, 'PRN': '131'},
     'S33': {'system': 'WAAS', 'Azimuth': 230.3, 'Elevation': 38.3, 'PRN': '133'},
     'S35': {'system': 'WAAS', 'Azimuth': 225.8, 'Elevation': 33.7, 'PRN': '135'},
     'S48': {'system': 'WAAS', 'Azimuth': 104.6, 'Elevation': 8.9, 'PRN': '148'}}
     
    Parameters
    ----------
    vod : pandas.DataFrame
        DataFrame containing GNSS VOD data with SV in the index
    sbas_ident : dict
        Dictionary mapping SBAS SVs to their fixed position information

    Returns
    -------
    pandas.DataFrame
        DataFrame with updated Azimuth and Elevation values for SBAS satellites
    """
    # List of SBAS satellites to process
    sbas_list = ['S31', 'S33', 'S35', 'S48']
    print(f"Replacing Azimuth and Elevation for SBAS satellites: {sbas_list}")
    
    # Create a copy to avoid warnings
    vod_updated = vod.copy()
    
    # Reset index to make SV a column for easier filtering
    vod_reset = vod_updated.reset_index()
    
    # For each SBAS satellite
    for sv in sbas_list:
        if sv in sbas_ident:
            # Get the fixed values from sbas_ident
            fixed_azi = np.float16(sbas_ident[sv]['Azimuth'])
            fixed_ele = np.float16(sbas_ident[sv]['Elevation'])
            
            # Create mask for this SV
            mask = vod_reset['SV'] == sv
            
            if mask.any():
                # Replace values
                vod_reset.loc[mask, 'Azimuth'] = fixed_azi
                vod_reset.loc[mask, 'Elevation'] = fixed_ele
                print(f"  Updated {mask.sum()} rows for {sv}: Azimuth={fixed_azi}, Elevation={fixed_ele}")
    
    # Set index back to original structure
    vod_updated = vod_reset.set_index(['Epoch', 'SV'])
    
    return vod_updated

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
        self.vod_file_temp = None
        self.anomaly_params = None
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
        self.temp_dir = DATA / 'temp'  # Directory for temporary files
        
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
        
    def _print_interval(self):
        """
        Print the time interval in a human-readable format.
        """
        start_date = pd.to_datetime(self.time_interval[0])
        end_date = pd.to_datetime(self.time_interval[1])
        print(f"Processing data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
    def _concat_nc_files(self, filepattern, interval, n_workers=15):
        """
        WARNING: This function cannot be debugged currently because of Dask client initialization.
        """
        from dask.distributed import Client, LocalCluster
        import concurrent.futures
        import os.path
        
        # Set up a local Dask cluster with multiple workers
        cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1)
        client = Client(cluster)
        print(f"Dask dashboard available at: {client.dashboard_link}")
        
        if interval:
            files_ = get_filelist({'': filepattern})
            files = {'': filter_files_by_date(files_[""], interval)}
        else:
            files = get_filelist({'': filepattern})
        
        print("Number of files found: ", len(files['']))
        print("Average filesize: ", np.mean([os.path.getsize(x) / 1e6 for x in files['']]).round(2), "MB")
        print("Total size: ", np.sum([os.path.getsize(x) / 1e6 for x in files['']]).round(2), "MB")
        
        # Process files in parallel using ThreadPoolExecutor
        print("Processing files individually in parallel...")
        valid_dfs = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(process_file, file): file for file in files['']}
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(files['']), desc="Reading files"):
                file = futures[future]
                try:
                    df = future.result()
                    if not df.empty:
                        valid_dfs.append(df)
                except Exception as e:
                    print(f"Exception processing {file}: {e}")
        
        # Check if we have any valid dataframes
        if not valid_dfs:
            raise ValueError("No valid data could be read from any of the input files")
        
        # Concatenate all valid dataframes
        data = pd.concat(valid_dfs, axis=0)
        
        # Clean up memory
        del valid_dfs
        
        # close the Dask client
        client.close()
        
        return data


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
        
        print_color(f"(1/2) Processing VOD data for {self.station}", "magenta")
        
        # Create a unique filename based on parameters
        start_date = pd.to_datetime(self.time_interval[0]).strftime('%Y%m%d')
        end_date = pd.to_datetime(self.time_interval[1]).strftime('%Y%m%d')
        # Create abbreviations for parameters
        band_abbr = f"bd{len(self.bands)}"  # Number of bands
        snr_abbr = f"rs{int(self.recover_snr)}"  # SNR recovery (1=True, 0=False)
        # Create filename with parameter abbreviations
        self.vod_filename = f"vod_{self.station}_{start_date}_{end_date}_{band_abbr}_{snr_abbr}"
        
        # Create temp directory if it doesn't exist
        self.vod_file_temp = self.temp_dir / (self.vod_filename + ".parquet")
        self.vod_file_temp.parent.mkdir(exist_ok=True, parents=True)
        
        # Check if file already exists and overwrite flag
        if self.vod_file_temp.exists() and not overwrite:
            print(f"VOD data file already exists at {self.vod_file_temp}. Loading from disk.")
            return None
        
        # -----------------------------------
        # concat all files
        """
        Can't debug after this point because of dask client initialization
        –> Consider commenting it out in self._concat_nc_files
        """

        print_color(f"Concatenating NC files matching pattern: {self.pattern}", "yellow")
        # This will return a Dask DataFrame
        df_combind_ncs = self._concat_nc_files(
            filepattern=self.pattern,
            interval=self.time_interval,
            n_workers=15,  # Adjust as needed
        )
        
        # -----------------------------------
        # Calculate VOD
        
        print_color(f"Calculating VOD, pairing:{self.pairings}, bands:{self.bands}", "yellow")
        vod = gv.calc_vod(
            df_combind_ncs,
            self.pairings,
            self.bands,
            recover_snr=self.recover_snr
        )
        
        # Save to disk
        try:
            # compute vod while saving as parquet
            vod.compute(optimize_graph=True).to_parquet(self.vod_file_temp, engine='pyarrow')
        except Exception as e:
            print(f"Error saving VOD data to {self.vod_file_temp}: {e}")
            return None
        return None
    
    
    def process_anomaly(self, angular_resolution, angular_cutoff, temporal_resolution, overwrite=False, **kwargs):
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
        
        ar = angular_resolution
        ac = angular_cutoff
        tr = temporal_resolution
        
        # Transform all parameters to lists if they are not already
        if not isinstance(ar, int):
            raise ValueError("angular_resolution must be int")
        if not isinstance(ac, int):
            raise ValueError("angular_cutoff must be int")
        if not isinstance(tr, int):
            raise ValueError("temporal_resolution must be int")
            
        # Store anomaly parameters for metadata
        self.anomaly_params = kwargs.copy()
        self.anomaly_params.update({"band_ids": self.band_ids})

        # if not len(angular_resolution) * len(angular_cutoff) * len(temporal_resolution) == 1 –> raise
        if ar*ac*tr == 0:
            raise ValueError("At least one parameter must be provided for angular_resolution, ")
        
        self.anomaly_params.update({"angular_resolution": ar,
                                    "angular_cutoff": ac,
                                    "temporal_resolution": tr})
        # Check for existing results files and load them if available
        result_paths = []
        
        # Create a unique filename based on parameters
        start_date = pd.to_datetime(self.time_interval[0]).strftime('%Y%m%d')
        end_date = pd.to_datetime(self.time_interval[1]).strftime('%Y%m%d')
        time_abbr = start_date if start_date == end_date else f"{start_date}_{end_date}"
        
        # Extract year and month from time interval for compact representation
        filename = f"vod_anomaly_{self.station}_{time_abbr}_ar{ar}_ac{ac}_tr{tr}.pkl"
        file_path = self.temp_dir / filename
        
        # Check if file exists and overwrite flag
        if not file_path.exists() and overwrite:
            """Process VOD anew"""
            print(f"Processing combination: AR={ar}, AC={ac}, TR={tr}")
            params = self.anomaly_params.copy()
            currrent_updates = {
                'angular_resolution': ar,
                'angular_cutoff': ac,
                'temporal_resolution': tr,
                'vod_anomaly_filename': filename,
            }
            # Update with kwargs
            params.update(currrent_updates)
            
            print(f"Processing parameter combination: {params}")
            # Load VOD data from disk for this process
            temp_dir = DATA / 'temp'
            anomaly_file_path = temp_dir / params['vod_anomaly_filename']
            vod_file_path = temp_dir / (self.vod_filename + ".parquet")
            
            # read parquet as pandas DataFrame
            vod = pd.read_parquet(vod_file_path)
            vod.set_index('SV', inplace=True, append=True)  # Ensure 'Epoch' is the index
            
            # add position of sbas satellites (why did they got lost?)
            if add_sbas_position_manually:
                print("Replacing SBAS coordinates with fixed values...")
                vod = _replace_sbas_coordinates(vod, SBAS_IDENT)
            
            # Build hemispheric grid
            hemi = gv.hemibuild(params['angular_resolution'], params['angular_cutoff'])
            
            # Classify VOD into grid cells
            print("Adding CellID to VOD data...")
            vod_with_cells = hemi.add_CellID(vod.copy(), drop=True)  # Drop position without data
            
            # Calculate anomalies (now accepts pandas DataFrame directly)
            print("Calculating anomalies...")
            vod_ts_combined, vod_avg = _calculate_anomaly(vod=vod_with_cells, **params)
            
            # Add parameter columns
            vod_ts_combined['angular_resolution'] = params['angular_resolution']
            vod_ts_combined['angular_cutoff'] = params['angular_cutoff']
            vod_ts_combined['temporal_resolution'] = params['temporal_resolution']
            
            # Save to disk
            vod_ts_combined.to_pickle(anomaly_file_path)
            print(f"Saved anomaly results to {anomaly_file_path}")
            result_paths.append(anomaly_file_path)
        else:
            """Loading existing results"""
            print(f"Found existing results for AR={ar}, AC={ac}, TR={tr}, Interval={self.time_interval}, loading from disk.")
            result_paths.append(str(file_path))
            
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