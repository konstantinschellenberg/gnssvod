import xarray as xr
import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, Optional, Union, List

from processing.settings import *


class VODReader:
    """
    Reader class for VOD (Vegetation Optical Depth) time series data.

    This class can read VOD time series data from NetCDF files and their associated
    metadata from JSON files. It provides methods to access the data and metadata.
    """
    
    def __init__(self, file_path_or_settings: Optional[Union[str, Path, Dict]] = None,
                 gatheryears: Dict[str, tuple] = None,
                 transform_time: bool = True):
        """
        Initialize the VOD reader with optional file path, settings dictionary, or list of years.

        Parameters
        ----------
        file_path_or_settings : str, Path, or Dict, optional
            Either:
            - Path to the NetCDF file to load
            - Dictionary of settings to search for matching files
        gatheryears : list of str, optional
            List of years to gather and concatenate datasets for.
        """
        self.file_path = None
        self.metadata_path = None
        self.data = None
        self.metadata = None
        self.loaded = False
        self.transform_time = transform_time  # Whether to transform time to a specific timezone
        
        # Load default settings from settings.py
        self._load_default_settings()
        
        # Handle gatheryears if provided
        if gatheryears:
            self._load_from_years(gatheryears)
        elif file_path_or_settings is not None:
            if isinstance(file_path_or_settings, dict):
                self.load_from_settings(file_path_or_settings)
            else:
                self.load_file(file_path_or_settings)
    
    def _load_from_years(self, gatheryears: Dict[str, tuple]):
        """
        Load and concatenate datasets for the specified years.

        Parameters
        ----------
        gatheryears : list of str
            List of years to gather datasets for.
        """
        
        combined_data = []
        for year in gatheryears:
            if year not in time_intervals:
                print(f"Warning: No time interval defined for year {year}. Skipping.")
                continue
            
            settings = {
                'station': self.default_settings.get('station', 'MOz'),
                'time_interval': time_intervals[year],
                'anomaly_type': self.default_settings.get('anomaly_type', 'unknown'),
            }
            
            # Load data for the year
            reader = VODReader(settings)
            year_data = reader.get_data(format='long')
            if year_data is not None:
                combined_data.append(year_data)
        
        if combined_data:
            self.data = pd.concat(combined_data, ignore_index=False)
            print(f"Successfully loaded and concatenated data for years: {', '.join(gatheryears)}")
            self._prep_data()
            self.loaded = True
        else:
            print("No data found for the specified years.")
            self.data = None
    
    def load_from_settings(self, settings: Dict):
        """
        Search for and load VOD files matching the given settings by checking metadata files.

        Parameters
        ----------
        settings : Dict
            Dictionary containing search parameters like station, time_interval,
            angular_resolution, temporal_resolution, angular_cutoff, etc.

        Returns
        -------
        self : VODReader
            Returns self for method chaining
        """
        from definitions import DATA
        import glob
        import json
        import os
        
        # Extract key parameters from settings
        station = settings.get('station', self.default_settings.get('station', 'MOz'))
        time_interval = settings.get('time_interval', self.default_settings.get('time_interval'))
        angular_resolution = settings.get('angular_resolution', 2)
        temporal_resolution = settings.get('temporal_resolution', 30)
        angular_cutoff = settings.get('angular_cutoff', 30)
        
        # Get the base directory for VOD time series files
        base_dir = DATA / "timeseries"
        
        # Search for all VOD metadata files for this station
        pattern = str(base_dir / f"vod_timeseries_{station}_*_metadata.json")
        all_metadata_files = glob.glob(pattern)
        
        if not all_metadata_files:
            print(f"No metadata files found for station: {station}")
            self._show_available_files(settings)
            return self
        
        matching_files = []
        file_info_list = []
        
        # Check each metadata file for matching criteria
        for metadata_file in all_metadata_files:
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Create file info for potential display
                file_info = {
                    "file": Path(metadata_file).name.replace('_metadata.json', '.nc'),
                    "station": metadata.get("station", "unknown"),
                    "time_start": metadata.get("time_interval", {}).get("start_date", "unknown"),
                    "time_end": metadata.get("time_interval", {}).get("end_date", "unknown"),
                }
                
                # Check if this file matches our criteria
                if metadata.get("station") == station:
                    # Check time interval if provided
                    if time_interval:
                        file_start = metadata.get("time_interval", {}).get("start_date")
                        file_end = metadata.get("time_interval", {}).get("end_date")
                        
                        # Skip if time intervals don't overlap
                        if file_start and file_end:
                            file_start_dt = pd.to_datetime(file_start)
                            file_end_dt = pd.to_datetime(file_end)
                            req_start_dt = pd.to_datetime(time_interval[0])
                            req_end_dt = pd.to_datetime(time_interval[1])
                            
                            if file_end_dt < req_start_dt or file_start_dt > req_end_dt:
                                continue
                    
                    # Look for angular resolution, temporal resolution, and cutoff parameters
                    # Try to find these parameters in different locations in the metadata
                    
                    # First check in "results" variables if they contain these parameters
                    found_resolution_params = False
                    if "results" in metadata and "variables" in metadata["results"]:
                        if ("angular_resolution" in metadata["results"]["variables"] and
                                "temporal_resolution" in metadata["results"]["variables"] and
                                "angular_cutoff" in metadata["results"]["variables"]):
                            # The file contains these parameters as data variables
                            # We consider this a match as we can check actual values after loading
                            nc_file = metadata_file.replace('_metadata.json', '.nc')
                            matching_files.append((nc_file, metadata_file))
                            found_resolution_params = True
                    
                    # If not found in results, check in anomaly_processing section
                    if not found_resolution_params:
                        ap = metadata.get("anomaly_processing", {})
                        
                        # Extract parameters, checking different possible locations
                        ar_value = ap.get("angular_resolution")
                        tr_value = ap.get("temporal_resolution")
                        ac_value = ap.get("angular_cutoff")
                        
                        # Add to file_info for display
                        file_info["angular_resolution"] = ar_value
                        file_info["temporal_resolution"] = tr_value
                        file_info["angular_cutoff"] = ac_value
                        
                        # Check if parameters match
                        if (ar_value == angular_resolution and
                                tr_value == temporal_resolution and
                                ac_value == angular_cutoff):
                            nc_file = metadata_file.replace('_metadata.json', '.nc')
                            matching_files.append((nc_file, metadata_file))
                
                file_info_list.append(file_info)
            
            except Exception as e:
                print(f"Error reading metadata file {metadata_file}: {e}")
        
        # If no matching files found, display available files
        if not matching_files:
            print("No files matching the specified criteria were found.")
            self._display_available_files_table(file_info_list, settings)
            return self
        
        # Sort matching files by date, newest first
        matching_files.sort(key=lambda x: os.path.getmtime(x[0]), reverse=True)
        
        # Load the newest matching file
        newest_file, metadata_file = matching_files[0]
        print(f"Found matching file: {Path(newest_file).name}")
        print(f"Loading file with metadata: {Path(metadata_file).name}")
        
        return self.load_file(newest_file)
    
    def _display_available_files_table(self, file_info_list, requested_settings):
        """
        Display a table of available files with their settings.

        Parameters
        ----------
        file_info_list : list
            List of dictionaries containing file information
        requested_settings : dict
            The settings that were requested but not found
        """
        if not file_info_list:
            print("No files found for this station.")
            return
        
        # Create a DataFrame from the file info
        df = pd.DataFrame(file_info_list)
        
        # Add requested settings for comparison
        print("\nRequested settings:")
        for key, value in requested_settings.items():
            if key == 'time_interval' and value:
                print(f"  time_interval: {value[0]} to {value[1]}")
            else:
                print(f"  {key}: {value}")
        
        print("\nAvailable files:")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        
        # Sort by start_date (newest first)
        if 'start_date' in df.columns:
            df = df.sort_values('create_datetime_metadata', ascending=False)
        
        print(df.head(20))
        print("\nUse one of these files directly with VODReader(file_path) or adjust your settings.")
    
    def _show_available_files(self, settings):
        """Show all available files when no files are found for the station"""
        from definitions import DATA
        import glob
        
        print(f"No VOD files found for station: {settings.get('station', 'unknown')}")
        
        # Get all stations available
        base_dir = DATA / "timeseries"
        all_files = glob.glob(str(base_dir / "vod_timeseries_*.nc"))
        
        if not all_files:
            print("No VOD files found in the timeseries directory.")
            return
        
        # Extract station names
        stations = set()
        file_info = []
        
        for file_path in all_files:
            parts = Path(file_path).stem.split('_')
            if len(parts) >= 3:
                station = parts[2]
                stations.add(station)
                
                # Get basic file info
                info = {
                    "file": Path(file_path).name,
                    "station": station,
                    "modified": pd.to_datetime(Path(file_path).stat().st_mtime, unit='s')
                }
                
                # Try to extract dates from filename
                try:
                    if len(parts) >= 5:
                        info["start_date"] = pd.to_datetime(parts[3], format='%Y%m%d').strftime('%Y-%m-%d')
                        end_str = ''.join(c for c in parts[4] if c.isdigit())[:8]
                        info["end_date"] = pd.to_datetime(end_str, format='%Y%m%d').strftime('%Y-%m-%d')
                except:
                    pass
                
                file_info.append(info)
        
        print(f"\nAvailable stations: {', '.join(sorted(stations))}")
        print("\nRecent files:")
        
        # Create DataFrame and show most recent files
        df = pd.DataFrame(file_info)
        df = df.sort_values('modified', ascending=False).head(10)
        print(df[['file', 'station', 'start_date', 'end_date']])
    
    def _load_default_settings(self):
        """Load default settings from settings.py"""
        try:
            from processing.settings import (
                station, ground_station, tower_station, bands,
                angular_resolution, temporal_resolution, agg_func,
                anomaly_type, angular_cutoff, single_file_interval
            )
            
            self.default_settings = {
                'station': station,
                'ground_station': ground_station,
                'tower_station': tower_station,
                'bands': bands,
                'angular_resolution': angular_resolution,
                'temporal_resolution': temporal_resolution,
                'agg_func': agg_func,
                'anomaly_type': anomaly_type,
                'angular_cutoff': angular_cutoff,
                'time_interval': single_file_interval
            }
        except ImportError:
            print("Warning: Could not import settings from processing.settings")
            self.default_settings = {}
            
    def _prep_data(self):
        
        """
        SPACE FOR ALL THE MISCELLANEOUS VARIABLE CREATED FROM VOD DATA ALONE
        
        
        
        
        """
        
        df = self.data.copy()
        
        # Check if 'Epoch' is in the index names
        if 'Epoch' in df.index.names:
            # Rename 'Epoch' level to 'datetime'
            df.index = df.index.set_names('datetime', level='Epoch')
        
        # datetime to pd.DatetimeIndex, utc
        df.index = pd.to_datetime(df.index, utc=True)
        
        # Sort index to ensure time series is in order
        if 'datetime' in df.index.names:
            df = df.sort_index(level='datetime')
        
        # transfrom the time from utc to central us time
        if self.transform_time:
            # Convert to UTC first if not already
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            # Convert to tz
            df.index = df.index.tz_convert(visualization_timezone)
            
        # add hour-of-day as a column (subminute)
        df['hod'] = df.index.get_level_values('datetime').hour + df.index.get_level_values('datetime').minute / 60.0
        # add day-of-year as a column
        df['doy'] = df.index.get_level_values('datetime').dayofyear
        df['year'] = df.index.get_level_values('datetime').year
        
        # select only falling into any number of temporal intervals in "time_intervals" list-of-tuples
        self.data = df
    
    def load_file(self, file_path: Union[str, Path]):
        """
        Load a VOD NetCDF file and its associated metadata.

        Parameters
        ----------
        file_path : str or Path
            Path to the NetCDF file to load

        Returns
        -------
        self : VODReader
            Returns self for method chaining
        """
        # Convert to Path object
        
        filetype = Path(file_path).suffix
        self.file_path = Path(file_path)
        
        # Check if file exists
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        # Load data
        print(f"Loading VOD data from {self.file_path}")
        
        if filetype.lower() == '.nc':
            ds = xr.open_dataset(self.file_path)
            self.data = ds.to_dataframe()
        elif filetype.lower() == '.csv':
            self.data = pd.read_csv(self.file_path, index_col=0, parse_dates=True)
            
        # prep data
        self._prep_data()

        # Find and load metadata
        if filetype.lower() == '.nc':
            self.metadata_path = Path(str(self.file_path).replace('.nc', '_metadata.json'))
            if self.metadata_path.exists():
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                    print(f"Loaded metadata from {self.metadata_path}")
            else:
                print(f"Warning: No metadata file found at {self.metadata_path}")
                self.metadata = self.default_settings
        else:
            print(f"No metadata file for CSV format. Using default settings.")
            
        # Set loaded flag
        self.loaded = True
    
    def print_metadata(self):
        """Print the metadata in a readable format"""
        if not self.metadata:
            print("No metadata available")
            return
        
        print("VOD Metadata:")
        print("-" * 50)
        for key, value in sorted(self.metadata.items()):
            if isinstance(value, dict):
                print(f"{key}:")
                for k, v in sorted(value.items()):
                    print(f"  {k}: {v}")
            else:
                print(f"{key}: {value}")
        print("-" * 50)
    
    def get_data(self, format="long") -> pd.DataFrame:
        """
        Get the VOD data as a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            The VOD time series data
        """
        if format not in ["long", "wide"]:
            raise ValueError("format must be 'long' or 'wide'")
        
        if format == "long":
            return self.data
        elif format == "wide":
            
            # Identify common indices to preserve
            common_indices = ['datetime', 'hod', 'doy', 'year']
            
            # Ensure common indices exist as columns
            vod_reset =  self.data.reset_index() if any(idx not in  self.data.columns for idx in common_indices) else  self.data.copy()
            
            # Only keep common indices that actually exist
            existing_indices = [idx for idx in common_indices if idx in vod_reset.columns]
            
            # Identify value columns (all except indices and 'algo')
            value_columns = [col for col in vod_reset.columns if col not in existing_indices + ['algo']]
            
            # Reshape DataFrame to have algorithm as part of column names
            vod_wide = vod_reset.pivot_table(
                index=existing_indices,
                columns='algo',
                values=value_columns
            )
            
            # Flatten the MultiIndex columns to get "column_algo" format
            vod_wide.columns = [f"{col[0]}_{col[1]}" for col in vod_wide.columns]
            
            # Reset index to get a regular DataFrame
            vod_wide = vod_wide.reset_index()
            # set datetime as index
            return vod_wide.set_index('datetime')
        else:
            raise ValueError("Invalid format specified. Use 'long' or 'wide'.")
    
    def get_metadata(self) -> Dict:
        """
        Get the metadata as a dictionary.

        Returns
        -------
        dict
            The metadata dictionary
        """
        return self.metadata if self.metadata else self.default_settings
    
    def get_bands(self) -> List[str]:
        """
        Get the list of VOD bands in the data.

        Returns
        -------
        list
            List of band names (e.g., ['VOD1', 'VOD2'])
        """
        if self.metadata and 'bands' in self.metadata:
            return list(self.metadata['bands'].keys())
        elif self.default_settings and 'bands' in self.default_settings:
            return list(self.default_settings['bands'].keys())
        else:
            # Try to infer from column names
            if self.data is not None:
                return [col for col in self.data.columns if col.startswith('VOD') and not col.endswith('_anom')]
            return []
    
    def plot_anomalies(self, figsize=(8, 5), title=None):
        """
        Plot VOD anomalies for all bands.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size (width, height) in inches
        title : str, optional
            Title for the plot
        """
        if self.data is None:
            print("No data loaded")
            return
        
        bands = self.get_bands()
        if not bands:
            print("No VOD bands found in data")
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        
        for band in bands:
            if f"{band}_anom" in self.data.columns:
                ax.plot(self.data.index, self.data[f"{band}_anom"], label=f"{band} anomaly")
                ax.plot(self.data.index, self.data[band], label=band, alpha=0.5)
        
        ax.legend()
        ax.set_ylabel("GNSS-VOD")
        ax.set_xlabel("Date")
        ax.grid(True, linestyle='--', alpha=0.7)
        
        if title:
            plt.title(title)
        else:
            station = self.metadata.get('station', 'unknown')
            anom_type = self.metadata.get('anomaly_type', 'unknown')
            plt.title(f"VOD Anomalies - {station} ({anom_type} method)")
        
        plt.tight_layout()
        plt.show()
    
    def summary(self):
        """Print a summary of the VOD data"""
        if self.data is None:
            print("No data loaded")
            return
        
        print(f"VOD Data Summary:")
        print("-" * 50)
        print(f"File: {self.file_path}")
        print(f"Time range: {self.data.index.min()} to {self.data.index.max()}")
        print(f"Number of observations: {len(self.data)}")
        print(f"Bands: {self.get_bands()}")
        print(f"Anomaly type: {self.metadata.get('anomaly_type', 'unknown')}")
        print(f"Columns: {', '.join(self.data.columns)}")
        print("-" * 50)
        print("Basic statistics:")
        print(self.data.describe().round(3))
    
    def __repr__(self):
        """String representation of the VODReader"""
        if self.file_path:
            return f"VODReader(file='{self.file_path.name}')"
        else:
            if self.loaded:
                return "VODReader(loaded with gatheryears)"
            else:
                return "VODReader() - No file loaded yet"
        
if __name__ == "__main__":
    # Example usage
    from definitions import DATA
    
    _file = "vod_timeseries_MOz_20240501_20240505_-18b3fb34443d7656.nc"
    
    # Option 1: Load a specific file
    vod_file = DATA / "timeseries" / _file
    reader = VODReader(vod_file)
    
    # -----------------------------------
    # Option 2: Initialize first, then load
    reader = VODReader()
    reader.load_file(vod_file)
    
    # Print metadata
    reader.print_metadata()
    
    # Get the data as a DataFrame
    df = reader.get_data()
    print(df.head())
    
    # Print summary statistics
    reader.summary()
    
    # Plot anomalies
    reader.plot_anomalies(title="VOD Anomalies May 2024")
    
    # -----------------------------------
    # Option 3: Load using settings
    # Find the most recent VOD file with specific settings
    settings = {
        'station': 'MOz',
        'time_interval': ('2024-05-01', '2024-05-30'),
        'anomaly_type': 'phi_theta'
    }
    
    # Create reader with automatic file selection
    reader = VODReader(settings)
    
    # Or initialize first, then search
    reader = VODReader()
    reader.load_from_settings(settings)
    
    # Get the data
    df = reader.get_data()
