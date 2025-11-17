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
from dataclasses import asdict, fields, MISSING
from matplotlib.colors import ListedColormap
import matplotlib as mpl
import matplotlib.dates as mdates  # NEW
import re  # NEW

from definitions import DATA, FIG
import gnssvod as gv
from gnssvod.io.preprocess import get_filelist
from processing.filepattern_finder import create_time_filter_patterns, filter_files_by_date
from processing.helper import print_color, check_instance
from processing.settings import bands, SBAS_IDENT, station, moflux_coordinates, add_sbas_position_manually
from processing.export_vod_funs import plot_hemi
from processing.aux import constellation_map, constellation_colors
from analysis.calculate_anomalies import _calculate_anomaly
from analysis.config import AnomalyConfig, VodConfig
from analysis.config import VAR_LABELS  # NEW


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


def _replace_sbas_coordinates(vod, sbas_ident, debug=False):
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
    print(f"Replacing Azimuth and Elevation for SBAS satellites: {sbas_list}") if debug else None
    
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
                print(f"  Updated {mask.sum()} rows for {sv}: Azimuth={fixed_azi}, Elevation={fixed_ele}") if debug else None
    
    # Set index back to original structure
    vod_updated = vod_reset.set_index(['Epoch', 'SV'])
    
    return vod_updated

class VODProcessor:
    def __init__(self, station=station, bands=bands, time_interval=None,
                 pairings=None, recover_snr=True, plot=False, overwrite=False, debug=False):
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
        self.overwrite = overwrite
        self.recover_snr = recover_snr
        self.debug = debug
        
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


    def process_vod(self, cfg: VodConfig):
        """
        Process VOD data and save to temp parquet (memory friendly).

        Parameters
        ----------
        cfg : VodConfig
            Explicit processing configuration.
        """
        if not isinstance(cfg, VodConfig):
            raise TypeError("cfg must be a VodConfig instance")

        # Store processing parameters for metadata
        self.vod_processing_params = asdict(cfg)

        print_color(f"(1/2) Processing VOD data for {self.station}", "magenta")

        # Unique filename based on parameters and interval
        start_date = pd.to_datetime(self.time_interval[0]).strftime('%Y%m%d')
        end_date = pd.to_datetime(self.time_interval[1]).strftime('%Y%m%d')
        band_abbr = f"bd{len(self.bands)}"
        snr_abbr = f"rs{int(self.recover_snr)}"
        self.vod_filename = f"vod_{self.station}_{start_date}_{end_date}_{band_abbr}_{snr_abbr}"

        # Prepare output path
        self.vod_file_temp = self.temp_dir / (self.vod_filename + ".parquet")
        self.vod_file_temp.parent.mkdir(exist_ok=True, parents=True)

        # Reuse if present
        if self.vod_file_temp.exists() and not cfg.overwrite:
            print(f"    VOD data file already exists at {self.vod_file_temp}. Loading from disk.")
            return None

        # Concatenate input files
        print_color(f"Concatenating NC files matching pattern: {self.pattern}", "yellow")
        df_combind_ncs = self._concat_nc_files(
            filepattern=self.pattern,
            interval=self.time_interval,
            n_workers=cfg.n_workers,
        )

        # Calculate VOD
        print_color(f"Calculating VOD, pairing:{self.pairings}, bands:{self.bands}", "yellow")
        vod = gv.calc_vod(
            df_combind_ncs,
            self.pairings,
            self.bands,
            recover_snr=self.recover_snr
        )

        # Save to parquet
        try:
            vod.compute(optimize_graph=True).to_parquet(self.vod_file_temp, engine='pyarrow')
        except Exception as e:
            print(f"Error saving VOD data to {self.vod_file_temp}: {e}")
            return None
        return None
    
    
    def process_anomaly(self, cfg: AnomalyConfig):
        """
        Calculate VOD anomalies for a single, explicit configuration.

        Parameters
        ----------
        cfg : AnomalyConfig

        Returns
        -------
        xarray.Dataset
            Combined results loaded from the saved output.
        """
        if not isinstance(cfg, AnomalyConfig):
            raise TypeError("cfg must be an AnomalyConfig instance")

        print_color(f"(2/2) Processing VOD anomalies for {self.station}", "magenta")

        # Persist config for metadata (and attach band_ids for traceability)
        params = asdict(cfg)
        params.update({"band_ids": self.band_ids})

        # Canonical file naming (kept as before)
        start_date = pd.to_datetime(self.time_interval[0]).strftime('%Y%m%d')
        end_date = pd.to_datetime(self.time_interval[1]).strftime('%Y%m%d')
        time_abbr = start_date if start_date == end_date else f"{start_date}_{end_date}"
        
        # File paths
        filename = (
            f"vod_anomaly_{self.station}_{time_abbr}"
            f"_ar{cfg.angular_resolution}_ac{cfg.angular_cutoff}_tr{cfg.temporal_resolution}.pkl"
        )
        temp_dir = DATA / 'temp'
        file_path = self.temp_dir / filename
        anomaly_file_path = self.temp_dir / filename
        hemispheric_file_path = anomaly_file_path.parent / filename.replace("vod_anomaly", "vod_hemisphere")
        
        # Save the exact config in the processor for metadata
        self.anomaly_params = {**params, "vod_anomaly_filename": filename}

        # -----------------------------------
        # PROCESSING
        # -----------------------------------
        if file_path.exists() and not cfg.overwrite:
            print(f"    Found existing results for AR={cfg.angular_resolution}, AC={cfg.angular_cutoff}, TR={cfg.temporal_resolution}, Interval={self.time_interval}, loading from disk.")
            result_paths = [str(file_path)]
        else:
            # Load VOD parquet produced earlier
            
            if not self.vod_file_temp.exists():
                raise FileNotFoundError(
                    f"Expected VOD parquet not found: {self.vod_file_temp}\n"
                    f"Run process_vod() first for the same interval and station."
                )
            vod = pd.read_parquet(self.vod_file_temp)
            vod.set_index('SV', inplace=True, append=True)

            # -----------------------------------
            # Add SBAS fixed coordinates if requested globally
            if add_sbas_position_manually:
                if self.debug:
                    print("Replacing SBAS coordinates with fixed values...")
                vod = _replace_sbas_coordinates(vod, SBAS_IDENT, debug=self.debug)
            
            # -----------------------------------
            # Build hemispheric grid and add CellID
            hemi = gv.hemibuild(cfg.angular_resolution, cfg.angular_cutoff)
            if self.debug:
                print(f"Built hemisphere grid: AR={cfg.angular_resolution}, AC={cfg.angular_cutoff}")
            vod_with_cells = hemi.add_CellID(vod.copy(), drop=True)
            
            # -----------------------------------
            # Filter SV from cfg.constellations if specified
            
            if cfg.constellations is not None and len(cfg.constellations) > 0:
                letters_requested = {str(c).lower() for c in cfg.constellations}
                # constellation_map: letter -> full name
                letters = {
                    k for k, v in constellation_map.items()
                    if k.lower() in letters_requested or (isinstance(v, str) and v.lower() in letters_requested)
                }
                sv_level = vod_with_cells.index.get_level_values('SV').astype(str)
                initial_count = sv_level.nunique()
                mask = sv_level.str[0].isin(letters)
                vod_with_cells = vod_with_cells[mask]
                final_count = vod_with_cells.index.get_level_values('SV').astype(str).nunique()
                
            # -----------------------------------
            # -----------------------------------
            # Calculate anomalies
            vod_ts_combined, vod_avg = _calculate_anomaly(
                vod=vod_with_cells,
                band_ids=self.band_ids,
                cfg=cfg
            )
            # -----------------------------------
            # -----------------------------------
            # Attach configuration columns to the output for traceability
            for k, v in params.items():
                try:
                    vod_ts_combined[k] = v
                except Exception:
                    pass  # robust to index alignment variations

            # Save to disk
            vod_ts_combined.to_pickle(anomaly_file_path)
            vod_avg.to_pickle(hemispheric_file_path)
            result_paths = [str(anomaly_file_path)]

        # -----------------------------------
        # SAVE (This might be a horrible way to do it. I should just load them into memory directly?)
        # -----------------------------------
        # Save as VOD timeseries (NetCDF + metadata)
        # Combine and keep as xarray
        self.results = self._combine_results(result_paths)

        if self.results is not None:
            outfile_timeseries = self._save_vod_timeseries(
                filename=f"vod_timeseries_{self.station}",
                overwrite=self.overwrite,
            )
        else:
            raise ValueError("No results to save as time series.")
            
        print(f"    2) Saved VOD time series to {outfile_timeseries}")
        # -----------------------------------
        # Save hemispheric data as well (vod_avg, as nc)
        # read in pickle
        
        self.hemi = pd.read_pickle(hemispheric_file_path)
        hemispheric_file_path_final = outfile_timeseries.parent / filename.replace("vod_anomaly", "vod_hemisphere")
        # replace .pkl to .nc
        hemispheric_file_path_final = hemispheric_file_path_final.with_suffix('.nc')
        self.hemi.to_xarray().to_netcdf(hemispheric_file_path_final)
        print(f"    3) Saved hemisphere data to {hemispheric_file_path_final}")
        
        # BUT registered results as timeseries and hemi as hemispheric mean vod
        return None
    
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
        
        # make filename with date range and metadata id
        metadata = self._create_metadata()
        start, end = pd.to_datetime(self.time_interval[0]), pd.to_datetime(self.time_interval[1])
        outname = f"{filename}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}_{metadata['metadata_id']}"
        
        # Add .nc extension if not already present
        if not outname.endswith('.nc'):
            outname = f"{outname}.nc"
        
        # Full file path
        file_path = directory / outname
        
        try:
            # Create a copy of the dataframe to avoid modifying the original
            df = self.results.to_dataframe().reset_index().copy()
            
            # Rename 'Epoch' to 'datetime'
            df = df.rename(columns={'Epoch': 'datetime'})
            
            # Set 'datetime' as the index again
            df = df.set_index('datetime')
            
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
            
            return file_path
        
        except Exception as e:
            # throw traceback
            import traceback
            traceback.print_exc()
    
    def _var_label(self, var_name):
        """
        Get a human-readable label for a variable.

        Parameters
        ----------
        var_name : str
            Variable name

        Returns
        -------
        str
            Human-readable label
        """
        # get the string after all after 'anom_'
        import re
        suff = re.sub(r'^.*anom_', '', var_name)
        return VAR_LABELS.get(suff, var_name)
    
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
            
            print(f"    1) Saved metadata to {metadata_path}")
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
        # Helper to build a minimal schema from a dataclass
        def _schema_from_dataclass(dc):
            try:
                flds = fields(dc)
            except Exception:
                return {}
            schema = {}
            for f in flds:
                # best-effort type name
                tname = getattr(f.type, "__name__", str(f.type))
                if f.default is not MISSING:
                    default = f.default
                else:
                    # default_factory may also be present
                    default = "<required>" if getattr(f, "default_factory", MISSING) is MISSING else "<factory>"
                schema[f.name] = {"type": tname, "default": default}
            return schema

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
                "bands": {k: v for k, v in self.bands.items()},
                "band_ids": self.band_ids,
                "pairings": self.pairings,
                "recover_snr": self.recover_snr,
                "overwrite": self.overwrite,
            }
        }

        # Add processing parameters if they exist (kept for backward compatibility)
        if hasattr(self, 'vod_processing_params'):
            metadata["vod_processing"] = self.vod_processing_params

        # Add anomaly parameters if they exist (kept for backward compatibility)
        if hasattr(self, 'anomaly_params'):
            metadata["anomaly_processing"] = self.anomaly_params
            multi_parameter = any(isinstance(v, list) and len(v) > 1 for v in self.anomaly_params.values())
            metadata["anomaly_processing"]["multi_parameter"] = multi_parameter
        else:
            anomaly_params = {}
            for param in ['angular_resolution', 'angular_cutoff', 'temporal_resolution']:
                if hasattr(self, param):
                    anomaly_params[param] = getattr(self, param)
            if anomaly_params:
                metadata["anomaly_processing"] = anomaly_params

        # Results summary (unchanged)
        if hasattr(self, 'results') and self.results is not None:
            metadata["results"] = {
                "variables": list(self.results.variables),
                "dimensions": {dim: len(self.results[dim]) for dim in self.results.dims},
                "coordinates": {coord: list(self.results[coord].values)
                                for coord in self.results.coords
                                if len(self.results[coord].values) < 100}
            }

        # NEW: attach config schemas and values (automatically from the dataclasses)
        try:
            vod_schema = _schema_from_dataclass(VodConfig)
        except Exception:
            vod_schema = {}
        try:
            anom_schema = _schema_from_dataclass(AnomalyConfig)
        except Exception:
            anom_schema = {}

        config_section = {
            "schema": {
                "VodConfig": vod_schema,
                "AnomalyConfig": anom_schema,
            },
            "values": {}
        }

        # Populate used values, filtered to each config's declared fields
        if hasattr(self, 'vod_processing_params') and isinstance(vod_schema, dict) and vod_schema:
            config_section["values"]["VodConfig"] = {
                k: self.vod_processing_params.get(k, None) for k in vod_schema.keys()
            }

        if hasattr(self, 'anomaly_params') and isinstance(anom_schema, dict) and anom_schema:
            # Only keep true AnomalyConfig fields; extras (e.g., band_ids, filenames) go to "extras"
            anom_vals = {k: self.anomaly_params.get(k, None) for k in anom_schema.keys()}
            extras = {k: v for k, v in self.anomaly_params.items() if k not in anom_schema}
            config_section["values"]["AnomalyConfig"] = anom_vals
            if extras:
                config_section["values"]["AnomalyConfig_extras"] = extras

        metadata["config"] = config_section

        # Stable metadata id
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
    
    # print these statistics: Per SV and Constellation: number of total observation, fraction of obs in the total time
    # series, mode of time-of-day distribution
    def compute_sv_statistics(self, constellations=None, elevation_min=None, elevation_max=None,
                              time_zone='UTC', make=True):
        """
        Compute statistics per satellite vehicle (SV) and constellation.
        Parameters
        ----------
        constellations : list[str] | None
            Names like ['gps','glonass','galileo','beidou'] (case-insensitive).
        elevation_min, elevation_max : float | None
            Optional elevation angle filter (deg).
        time_zone : str
            Time zone for local time-of-day conversion.
        Returns
        -------
        pd.DataFrame
            DataFrame with statistics per SV and constellation.
        """
        # turn of pandas warning about setting with copy
        pd.options.mode.chained_assignment = None  # default='warn'
        
        if not make:
            return None
        
        # Ensure parquet exists
        if self.vod_file_temp is None or not self.vod_file_temp.exists():
            print("VOD parquet file not found. Run process_vod first.")
            return None
        
        # Columns of interest
        usecols = ['Epoch', 'SV', 'Elevation']
        try:
            vod = pd.read_parquet(self.vod_file_temp, columns=usecols)
        except Exception:
            vod = pd.read_parquet(self.vod_file_temp)
            vod = vod[[c for c in usecols if c in vod.columns]]
        # make epoch a col from index
        if vod.index.name == 'Epoch':
            vod = vod.reset_index()
        
        def make_stats(sub_vod):
            total_obs = len(sub_vod)
            if total_obs == 0:
                return pd.Series({
                    'Total_Observations': 0,
                    'Fraction_of_Time': 0.0,
                    'Mode_Hour_of_Day': np.nan
                })
            # Time span in days
            time_span_days = (sub_vod['Epoch'].max() - sub_vod['Epoch'].min()).total_seconds() / 86400.0
            fraction_of_time = total_obs / (time_span_days) if time_span_days > 0 else 0.0
            
            # Mode of time-of-day distribution
            sub_vod['datetime'] = pd.to_datetime(sub_vod['Epoch'])
            sub_vod['datetime_local'] = sub_vod['datetime'].dt.tz_localize('UTC').dt.tz_convert(time_zone)
            sub_vod['hour_frac'] = (sub_vod['datetime_local'].dt.hour +
                                    sub_vod['datetime_local'].dt.minute / 60.0 +
                                    sub_vod['datetime_local'].dt.second / 3600.0)
            mode_hour = sub_vod['hour_frac'].mode()
            mode_hour_value = mode_hour.iloc[0] if not mode_hour.empty else np.nan
            
            return pd.Series({
                'Total_Observations': total_obs,
                'Fraction_of_Time': fraction_of_time,
                'Mode_Hour_of_Day': mode_hour_value
            })
        
        vod = vod.dropna(subset=['SV', 'Epoch']).copy()
        vod['SV'] = vod['SV'].astype(str)
        # Constellation filtering
        if constellations is None:
            requested = {'gps', 'glonass', 'galileo', 'beidou'}
        else:
            requested = {c.lower() for c in constellations}
        letter_to_name = constellation_map  # already imported
        name_to_letter = {v.lower(): k for k, v in letter_to_name.items()}
        requested_letters = {name_to_letter[c] for c in requested if c in name_to_letter}
        vod['Constellation'] = vod['SV'].str[0].map(letter_to_name)
        vod = vod[vod['SV'].str[0].isin(requested_letters)]
        # Drop SBAS if accidentally included
        vod = vod[vod['SV'].str[0] != 'S']
        
        if elevation_min is not None:
            vod = vod[vod['Elevation'] >= elevation_min]
        if elevation_max is not None:
            vod = vod[vod['Elevation'] <= elevation_max]
        if vod.empty:
            print("No data after elevation filtering.")
            return None
        
        # -----------------------------------
        # make stats
        stats_list = []
        for const in sorted(vod['Constellation'].dropna().unique()):
            sub = vod[vod['Constellation'] == const]
            for sv in sorted(sub['SV'].unique()):
                sv_sub = sub[sub['SV'] == sv]
                stats = make_stats(sv_sub)
                stats['SV'] = sv
                stats['Constellation'] = const
                stats_list.append(stats)
        stats_df = pd.DataFrame(stats_list)
        return stats_df
    
    def plot_hemispheric(self, var, angle_res: float = 1.0,
                         angle_cutoff: float = 30.0, make=True, **kwargs):
        if not make:
            return None
        hemi = gv.hemibuild(angle_res, angle_cutoff)
        plot_hemi(self.hemi, hemi.patches(), var=var, angular_cutoff=angle_cutoff, show_sbas=False, **kwargs)
    
    def plot_hist(self, vars, bins=50, sharex=True, figsize=None, cmap='viridis',
                  density=False, combine=False, save_path=None, show=True, make=True):
        """
        Plot histograms of one or multiple variables.

        Parameters
        ----------
        vars : str | list[str]
            Variable name(s) in results. If 'X' not found, tries 'X_anom'.
        bins : int | sequence
            Number of bins or bin edges (used in non-combined mode).
        sharex : bool
            If True, share x-axis limits across subplots (or apply common limits in combined mode).
        figsize : tuple | None
            Size of a single subplot (width, height) in inches. The total figure size
            is computed as (figsize[0] * ncols, figsize[1] * nrows). In combined mode,
            figsize applies to the single axis directly.
        cmap : str | Colormap
            Matplotlib colormap name or Colormap instance to color panels/lines distinctly.
        density : bool
            If True, normalize hist to show probability density (non-combined mode only).
        combine : bool
            If True, plot all variables in a single axis as KDE density curves with low-alpha fill.
        save_path : str | Path | None
            If provided, save the figure to this path.
        show : bool
            Whether to display the figure.
        """
        if not make:
            return None
        
        if self.results is None:
            print("No results available.")
            return None

        # Normalize vars input
        if isinstance(vars, str):
            vars = [vars]

        # Flatten results to DataFrame
        df = self.results.to_dataframe().reset_index()

        # Resolve columns (support _anom fallback)
        resolved = []
        for v in vars:
            if v in df.columns:
                resolved.append(v)
            elif f"{v}_anom" in df.columns:
                resolved.append(f"{v}_anom")
            else:
                print(f"Variable {v} (or {v}_anom) not found, skipping.")
        if not resolved:
            print("No valid variables to plot.")
            return None

        n = len(resolved)
        # Colors for each panel/line
        cm = plt.get_cmap(cmap)
        colors = cm(np.linspace(0.15, 0.85, n))

        # Compute shared x-limits if requested
        xlim = None
        if sharex:
            vmin = np.inf
            vmax = -np.inf
            for v in resolved:
                s = pd.to_numeric(df[v], errors='coerce').dropna()
                if not s.empty:
                    vmin = min(vmin, s.min())
                    vmax = max(vmax, s.max())
            if np.isfinite(vmin) and np.isfinite(vmax):
                pad = 0.02 * (vmax - vmin if vmax > vmin else 1.0)
                xlim = (vmin - pad, vmax + pad)

        # Combined KDE density plot (single axis)
        if combine:
            # Lazy import seaborn to avoid hard dependency at module import
            try:
                import seaborn as sns
            except Exception:
                sns = None

            # In combined mode, figsize applies directly to the single subplot
            if figsize is None:
                comb_figsize = (max(6, 3 + 1.5 * n), 4)
            else:
                comb_figsize = figsize

            fig, ax = plt.subplots(1, 1, figsize=comb_figsize)

            for v, c in zip(resolved, colors):
                data = pd.to_numeric(df[v], errors='coerce').dropna()
                if data.empty:
                    continue
                lbl = self._var_label(v)
                if sns is not None:
                    sns.kdeplot(data, ax=ax, fill=True, alpha=0.20, color=c, linewidth=1.2, label=lbl)
                else:
                    # Fallback: draw normalized histogram as a proxy
                    ax.hist(data, bins=bins, density=True, color=c, alpha=0.20,
                            edgecolor='none', label=lbl)

            if xlim is not None:
                ax.set_xlim(*xlim)
            ax.set_title("Density (KDE) – Combined", fontsize=12)
            ax.grid(alpha=0.3, linestyle='--')
            ax.set_xlabel("Value")
            ax.set_ylabel("Density")
            ax.legend(loc='best', fontsize=8, frameon=False)

            if save_path:
                sp = Path(save_path)
                if sp.suffix == "":
                    sp = sp.with_suffix(".png")
                fig.savefig(sp, dpi=300, bbox_inches='tight')

            if show:
                plt.show()
            else:
                plt.close(fig)
            return fig

        # Multi-panel histogram mode
        # Grid: 1→1x1, 2→1x2, 3–4→2x2, 5–6→2x3, etc.
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        # Treat figsize as size per subplot; compute total figure size
        if figsize is None:
            per_subplot = (4, 3)
        else:
            per_subplot = figsize
        total_figsize = (per_subplot[0] * cols, per_subplot[1] * rows)

        fig, axes = plt.subplots(rows, cols, figsize=total_figsize, sharex='all' if sharex else False)
        axes = np.atleast_1d(axes).ravel()

        for i, (v, c) in enumerate(zip(resolved, colors)):
            ax = axes[i]
            data = pd.to_numeric(df[v], errors='coerce').dropna()
            ax.hist(data, bins=bins, color=c, alpha=0.85, density=density,
                    edgecolor='white', linewidth=0.5)
            if xlim is not None:
                ax.set_xlim(*xlim)
            ax.set_title(self._var_label(v), fontsize=10)
            ax.grid(alpha=0.3, linestyle='--')
            ax.set_xlabel("Value")
            ax.set_ylabel("Density" if density else "Count")

        # Hide any unused subplots
        for j in range(i + 1, rows * cols):
            axes[j].set_visible(False)

        # Add date range (from requested interval)
        try:
            _start_dt = pd.to_datetime(self.time_interval[0])
            _end_dt = pd.to_datetime(self.time_interval[1])
            _span = (_end_dt - _start_dt) / pd.Timedelta(days=1)
            _title_suffix = f" (span: {_span:.1f} days; {_start_dt:%Y-%m-%d} to {_end_dt:%Y-%m-%d})"
        except Exception:
            _title_suffix = ""
        fig.suptitle(f"{self.station} – Variable Histograms{_title_suffix}", fontsize=13)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        if save_path:
            sp = Path(save_path)
            if sp.suffix == "":
                sp = sp.with_suffix(".png")
            sp = sp.with_name(sp.stem + "_hist" + sp.suffix)
            fig.savefig(sp, dpi=300, bbox_inches='tight')
            print(f"    Saved histogram figure to {sp}")

        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig
    
    def plot_correlation_matrix(self, vars=None, figsize=(8, 6),
                                cmap='coolwarm', save_path=None, show=True, make=True):
        """
        Plot correlation matrix heatmap for specified variables.
        """
        if not make:
            return None
        if self.results is None:
            print("No results available.")
            return None
        
        # Flatten results to DataFrame
        df = self.results.to_dataframe().reset_index()
        # If vars not specified, use all numeric columns
        if vars is None:
            vars = df.select_dtypes(include=[np.number]).columns.tolist()
        elif isinstance(vars, str):
            vars = [vars]
            
        # Resolve columns (support _anom fallback)
        resolved = []
        for v in vars:
            if v in df.columns:
                resolved.append(v)
            elif f"{v}_anom" in df.columns:
                resolved.append(f"{v}_anom")
            else:
                print(f"Variable {v} (or {v}_anom) not found, skipping.")
        if not resolved:
            print("No valid variables to plot.")
            return None
        
        # Compute correlation matrix
        corr_matrix = df[resolved].corr()
        
        # get label via self._var_label
        corr_matrix.index = [self._var_label(v) for v in corr_matrix.index]
        corr_matrix.columns = [self._var_label(v) for v in corr_matrix.columns]
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=figsize)
        cax = ax.matshow(corr_matrix, cmap=cmap, vmin=-1, vmax=1)
        # add R metric in each cell
        for (i, j), val in np.ndenumerate(corr_matrix.values):
            ax.text(j, i, f"{val:.2f}", ha='center', va='center', color='black' if abs(val) > 0.5 else 'black', fontsize=10)
        fig.colorbar(cax)
        ax.set_xticks(range(len(resolved)))
        ax.set_yticks(range(len(resolved)))
        ax.set_xticklabels(corr_matrix.columns, rotation=60, ha='center', fontsize=12)
        ax.set_yticklabels(corr_matrix.index, fontsize=12)
        ax.grid(False)
        plt.tight_layout()
        if save_path:
            sp = Path(save_path)
            if sp.suffix == "":
                sp = sp.with_suffix(".png")
            sp = sp.with_name(sp.stem + "_corr_matrix" + sp.suffix)
            fig.savefig(sp, dpi=300, bbox_inches='tight')
            print(f"    Saved correlation matrix figure to {sp}")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig
    
    def plot_diel(self, vars, algo=None, time_zone='UTC', y_range=None,
                  figsize=(12, 4), gradient_cmap=None, save_path=None,
                  show=True, make=True):
        """
        Plot full time series (left, 2/3 width) and diurnal curve (right, 1/3 width with mean ± std).
        """
        
        if not make:
            return None
        
        if self.results is None:
            print("No results available.")
            return None
        
        # Normalize vars input
        if isinstance(vars, str):
            vars = [vars]
        
        if gradient_cmap is None:
            gradient_cmap = 'viridis'
        
        # Work dataframe
        df = self.results.to_dataframe().reset_index()
        
        # Ensure datetime column
        if 'Epoch' not in df.columns:
            print("Epoch column missing in results.")
            return None
        
        df['datetime'] = pd.to_datetime(df['Epoch'])
        df['datetime_tz'] = df['datetime'].dt.tz_localize('UTC').dt.tz_convert(time_zone)
        
        # Resolve potential anomaly columns
        resolved_vars = []
        for v in vars:
            if v in df.columns:
                resolved_vars.append(v)
            elif f"{v}_anom" in df.columns:
                resolved_vars.append(f"{v}_anom")
            else:
                print(f"Variable {v} (or {v}_anom) not found, skipping.")
        if not resolved_vars:
            print("No valid variables to plot.")
            return None
        
        # Sort variables for consistent coloring
        resolved_vars = sorted(resolved_vars)
        
        # Build colors
        n = len(resolved_vars)
        cmap = plt.get_cmap(gradient_cmap)
        colors = cmap(np.linspace(0, 1, n))
        
        # Fractional hour (subhour precision)
        dt_local = df['datetime_tz']
        frac_hour = dt_local.dt.hour + dt_local.dt.minute / 60.0 + dt_local.dt.second / 3600.0
        df['fractional_hour'] = frac_hour
        
        # Diurnal cycle resolution: 5-min bins
        df['fh_rounded'] = (df['fractional_hour'] * 12).round() / 12.0  # 5 min bins
        
        # Compute diurnal mean and std
        diurnal_stats = {}
        for v in resolved_vars:
            grouped = df.groupby('fh_rounded')[v].agg(['mean', 'std']).dropna()
            diurnal_stats[v] = grouped
        
        # Figure layout: gridspec with unequal widths
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(1, 2, width_ratios=[2, 1], wspace=0.25)
        
        ax_ts = fig.add_subplot(gs[0, 0])
        ax_diel = fig.add_subplot(gs[0, 1])
        
        # Plot time series (left)
        for v, c in zip(resolved_vars, colors):
            ax_ts.plot(df['datetime'], df[v], color=c, label=self._var_label(v), linewidth=1)
        
        ax_ts.set_xlabel("Time (UTC)")
        ax_ts.set_ylabel("Value")
        if y_range is not None:
            ax_ts.set_ylim(*y_range)
        ax_ts.legend(loc='upper right', ncol=1, fontsize=8, frameon=False)
        ax_ts.set_title("Full Time Series")
        
        # Diurnal curve (right) – mean ± std, full panel
        for v, c in zip(resolved_vars, colors):
            stats = diurnal_stats[v]
            x = stats.index.values  # hours [0..24), 5-min bins
            y = stats['mean'].values
            s = stats['std'].values if 'std' in stats.columns else np.zeros_like(y)
            ax_diel.plot(x, y, color=c, linewidth=1.5, label=self._var_label(v))
            ax_diel.fill_between(x, y - s, y + s, color=c, alpha=0.2, linewidth=0)
        
        ax_diel.set_xlim(0, 24)
        ax_diel.set_xticks([0, 6, 12, 18, 24])
        ax_diel.set_xlabel("Hour of Day")
        ax_diel.set_ylabel("Mean ± 1σ")
        if y_range is not None:
            ax_diel.set_ylim(*y_range)
        ax_diel.grid(alpha=0.5, linestyle='--')
        ax_diel.legend(loc='upper right', ncol=1, fontsize=8, frameon=False)
        ax_diel.set_title("Mean Diurnal Curve")
        
        # FLEXIBLE DOY TICKS (moved here to apply after all plotting)
        start_dt = df['datetime'].min()
        end_dt = df['datetime'].max()
        span_days = (end_dt - start_dt).total_seconds() / 86400.0
        if span_days < 5:
            ax_ts.xaxis.set_major_locator(mdates.DayLocator())
            ax_ts.xaxis.set_minor_locator(mdates.HourLocator(byhour=range(0, 24, 6)))
            # set minor tick labels to hours
            ax_ts.xaxis.set_minor_formatter(mdates.DateFormatter('%H'))
            ax_ts.tick_params(axis='x', which='minor', labelsize=8)
        elif span_days < 15:
            ax_ts.xaxis.set_major_locator(mdates.DayLocator(interval=2))
            ax_ts.xaxis.set_minor_locator(mdates.DayLocator())
        elif span_days < 60:
            ax_ts.xaxis.set_major_locator(mdates.DayLocator(interval=5))
            ax_ts.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
        else:
            ax_ts.xaxis.set_major_locator(mdates.DayLocator(interval=30))
            ax_ts.xaxis.set_minor_locator(mdates.DayLocator(interval=5))
        ax_ts.xaxis.set_major_formatter(mdates.DateFormatter('%j'))  # DOY
        # set minor grid
        ax_ts.grid(which='minor', axis='x', linestyle=':', alpha=0.8)
        # set major grid
        ax_ts.grid(which='major', axis='x', linestyle='-', alpha=0.7)
        ax_ts.tick_params(axis='x', which='major', length=3, labelsize=10, color='grey')
        ax_ts.tick_params(axis='x', which='minor', length=0, color='gray')
        ax_ts.set_xlabel("Day of Year (DOY)", labelpad=20)
        # Ensure the x-limits are set to the data range for robust locator/formatter behavior
        ax_ts.set_xlim(start_dt, end_dt)
        
        # MONTH AXIS BELOW (uses full width of time-series panel)
        if span_days > 30:
            try:
                # todo: Some weird behavior still appears here
                fig.subplots_adjust(bottom=0.20)
                pos = ax_ts.get_position()
                ax_month = fig.add_axes([pos.x0, pos.y0 - 0.05, pos.width, 0.07], sharex=ax_ts)
                month_mid = pd.date_range(start=start_dt, end=end_dt, freq='MS') + pd.Timedelta('15D')
                # keep only those within the data range
                month_mid = month_mid[(month_mid >= start_dt) & (month_mid <= end_dt)]
                ax_month.set_xlim(start_dt, end_dt)
                ax_month.set_xticks(month_mid)
                ax_month.set_xticklabels([d.strftime('%b') for d in month_mid], fontsize=8)
                ax_month.tick_params(axis='x', length=0)
                ax_month.yaxis.set_visible(False)
                for spine in ax_month.spines.values():
                    spine.set_visible(False)
                # Make month axis transparent and behind main axis
                # ax_month.patch.set_alpha(0.0)
                # ax_month.set_zorder(0)
                ax_ts.set_zorder(2)
                # IMPORTANT: Re-enable DOY tick labels on the main axis (shared x would hide them)
                ax_ts.tick_params(axis='x', which='both', labelbottom=True)
                for lbl in ax_ts.get_xticklabels():
                    lbl.set_visible(True)
                    # set xorder
                    lbl.set_zorder(3)
            except Exception:
                pass
        
        plt.tight_layout()
        # Add date range (from actual plotted data)
        _span = (end_dt - start_dt) / pd.Timedelta(days=1)
        fig.suptitle(
            f"{self.station} – Time Series & Diurnal Curve (span: {_span:.1f} days; {start_dt:%Y-%m-%d} to {end_dt:%Y-%m-%d})",
            fontsize=13
        )
        
        if save_path:
            sp = Path(save_path)
            if sp.suffix == "":
                sp = sp.with_suffix(".png")
            fig.savefig(sp, dpi=300, bbox_inches='tight')
            print(f"    Saved diel figure to {sp}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def plot_time_series(self, vars, time_zone='UTC', y_range=None,
                         figsize=(12, 6), gradient_cmap=None, save_path=None,
                         show=True, make=True, temp_agg=None, temp_agg_fun='median',
                         clip_percentiles=(0.5, 99.5)):
        """
        Stacked time series with per-variable KDE histograms.
        Added percentile clipping (default 0.5 / 99.5) to condense dynamic range.

        Parameters
        ----------
        vars : str | list[str]
            Variable name(s). If missing, fall back to '{var}_anom' if present.
        time_zone : str
            Currently not used for plotting (time axis in UTC), kept for symmetry.
        y_range : tuple | None
            Fixed y-limits applied to all time-series axes.
        figsize : tuple
            Total figure size in inches.
        gradient_cmap : unused
            Kept for API symmetry; all lines are dark blue as requested.
        save_path : Path | str | None
            If provided, save figure to this path.
        show : bool
            Show the figure if True.
        make : bool
            Skip plotting when False.
        temp_agg : str | None
            Optional pandas offset string (e.g., '1h', '30min') to temporally aggregate the series.
        temp_agg_fun : str | callable, default 'median'
            Aggregation function for temporal aggregation.
        clip_percentiles : tuple(float,float) | None
            If not None, (low, high) percentiles used to auto-set y-limits when y_range not provided.
        """
        if not make:
            return None
        if self.results is None:
            print("No results available.")
            return None

        # Normalize vars input
        if isinstance(vars, str):
            vars = [vars]

        # Flatten results and ensure datetime
        df = self.results.to_dataframe().reset_index()
        if 'Epoch' not in df.columns:
            print("Epoch column missing in results.")
            return None
        df['datetime'] = pd.to_datetime(df['Epoch'])

        # Resolve requested variables (support _anom fallback)
        resolved = []
        for v in vars:
            if v in df.columns:
                resolved.append(v)
            elif f"{v}_anom" in df.columns:
                resolved.append(f"{v}_anom")
            else:
                print(f"Variable {v} (or {v}_anom) not found, skipping.")
        if not resolved:
            print("No valid variables to plot.")
            return None

        # Optional temporal aggregation on UTC time
        work = df[['datetime'] + resolved].copy()
        work = work.set_index('datetime').sort_index()
        if temp_agg:
            try:
                work = work.groupby(pd.Grouper(freq=temp_agg)).agg(temp_agg_fun)
            except Exception as e:
                print(f"Temporal aggregation failed ({temp_agg}, {temp_agg_fun}): {e}")
                return None
            work = work.dropna(how='all')
        # Prepare for plotting
        ts_df = work.reset_index().rename(columns={'datetime': 'Time'})
        start_dt = ts_df['Time'].min()
        end_dt = ts_df['Time'].max()
        span_days = (end_dt - start_dt) / pd.Timedelta(days=1)

        # Percentile clipping (compute global y-range if not provided)
        clip_range = None
        if y_range is None and clip_percentiles:
            lows = []
            highs = []
            for v in resolved:
                series = pd.to_numeric(work[v], errors='coerce').dropna()
                if series.empty:
                    continue
                p_low = np.percentile(series, clip_percentiles[0])
                p_high = np.percentile(series, clip_percentiles[1])
                lows.append(p_low)
                highs.append(p_high)
            if lows and highs:
                clip_range = (min(lows), max(highs))

        n = len(resolved)
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(nrows=n, ncols=2, width_ratios=[4, 1], wspace=0.25, hspace=0.25)

        # Left column: time series (stacked), share x
        axes_ts = []
        for i, v in enumerate(resolved):
            ax = fig.add_subplot(gs[i, 0], sharex=axes_ts[0] if axes_ts else None)
            ax.plot(ts_df['Time'], ts_df[v], color='#0b3c5d', linewidth=0.9,
                    label=self._var_label(v))
            ax.set_ylabel(self._var_label(v))
            if y_range is not None:
                ax.set_ylim(*y_range)
            elif clip_range is not None:
                ax.set_ylim(*clip_range)
            # Only label the bottom-most axis with x label
            if i < n - 1:
                ax.label_outer()
            axes_ts.append(ax)
        # Title for TS column
        axes_ts[0].set_title("Time series", loc='left')

        # Right column: KDE histograms (stacked), share x among hist panels
        # Compute shared x-limits across all variables from the plotted series
        xmin, xmax = np.inf, -np.inf
        for v in resolved:
            s = pd.to_numeric(ts_df[v], errors='coerce').dropna()
            if not s.empty:
                xmin = min(xmin, s.min())
                xmax = max(xmax, s.max())
        if not np.isfinite(xmin) or not np.isfinite(xmax):
            xmin, xmax = 0.0, 1.0
        pad = 0.02 * (xmax - xmin if xmax > xmin else 1.0)
        xlim_hist = (xmin - pad, xmax + pad)

        # Try seaborn for KDE; fall back to scipy.stats.gaussian_kde
        try:
            import seaborn as sns
            use_sns = True
        except Exception:
            use_sns = False
            from scipy.stats import gaussian_kde

        axes_kde = []
        for i, v in enumerate(resolved):
            axh = fig.add_subplot(gs[i, 1], sharex=axes_kde[0] if axes_kde else None)
            data = pd.to_numeric(ts_df[v], errors='coerce').dropna()
            if not data.empty:
                if use_sns:
                    sns.kdeplot(data, ax=axh, fill=True, alpha=0.25, color='#0b3c5d', linewidth=1.0)
                else:
                    try:
                        kde = gaussian_kde(data)
                        xgrid = np.linspace(xlim_hist[0], xlim_hist[1], 400)
                        y = kde(xgrid)
                        axh.fill_between(xgrid, y, color='#0b3c5d', alpha=0.25, linewidth=0)
                        axh.plot(xgrid, y, color='#0b3c5d', linewidth=1.0)
                    except Exception:
                        axh.hist(data, bins=40, density=True, color='#0b3c5d', alpha=0.25, edgecolor='none')
            axh.set_xlim(*xlim_hist)
            axh.grid(alpha=0.3, linestyle='--')
            # clip x-range to clip_range
            if clip_range is not None:
                axh.set_xlim(max(xlim_hist[0], clip_range[0]), min(xlim_hist[1], clip_range[1]))
            axh.set_ylabel("Density")
            # Hide y label; x-label only on bottom-most
            axh.set_yticklabels([])
            if i < n - 1:
                axh.label_outer()
            axes_kde.append(axh)
        # Title for histogram column
        axes_kde[0].set_title("KDE", loc='left')

        # Flexible DOY ticks: set after plotting, on the bottom-most time-series axis
        ax_bottom = axes_ts[-1]
        if span_days < 5:
            ax_bottom.xaxis.set_major_locator(mdates.DayLocator())
            ax_bottom.xaxis.set_minor_locator(mdates.HourLocator(byhour=range(0, 24, 6)))
            ax_bottom.xaxis.set_minor_formatter(mdates.DateFormatter('%H'))
            ax_bottom.tick_params(axis='x', which='minor', labelsize=8)
        elif span_days < 15:
            ax_bottom.xaxis.set_major_locator(mdates.DayLocator(interval=2))
            ax_bottom.xaxis.set_minor_locator(mdates.DayLocator())
        elif span_days < 60:
            ax_bottom.xaxis.set_major_locator(mdates.DayLocator(interval=5))
            ax_bottom.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
        else:
            ax_bottom.xaxis.set_major_locator(mdates.DayLocator(interval=30))
            ax_bottom.xaxis.set_minor_locator(mdates.DayLocator(interval=5))
        ax_bottom.xaxis.set_major_formatter(mdates.DateFormatter('%j'))  # DOY
        for ax in axes_ts:
            ax.grid(which='minor', axis='x', linestyle=':', alpha=0.8)
            ax.grid(which='major', axis='x', linestyle='-', alpha=0.7)
            ax.tick_params(axis='x', which='major', length=3, labelsize=10, color='grey')
            ax.tick_params(axis='x', which='minor', length=0, color='gray')
            ax.set_xlim(start_dt, end_dt)
        ax_bottom.set_xlabel("Day of Year (DOY)", labelpad=10)

        # Add month labels axis below if long span
        if span_days > 30:
            try:
                fig.subplots_adjust(bottom=0.15)
                pos = ax_bottom.get_position()
                ax_month = fig.add_axes([pos.x0, pos.y0 - 0.05, pos.width, 0.06], sharex=ax_bottom)
                month_mid = pd.date_range(start=start_dt, end=end_dt, freq='MS') + pd.Timedelta('15D')
                month_mid = month_mid[(month_mid >= start_dt) & (month_mid <= end_dt)]
                ax_month.set_xlim(start_dt, end_dt)
                ax_month.set_xticks(month_mid)
                ax_month.set_xticklabels([d.strftime('%b') for d in month_mid], fontsize=8)
                ax_month.tick_params(axis='x', length=0)
                ax_month.yaxis.set_visible(False)
                for spine in ax_month.spines.values():
                    spine.set_visible(False)
                # Ensure main DOY ticks are visible
                ax_bottom.tick_params(axis='x', which='both', labelbottom=True)
                for lbl in ax_bottom.get_xticklabels():
                    lbl.set_visible(True)
            except Exception:
                pass

        # Figure title with date span
        try:
            _span = span_days
            fig.suptitle(
                f"{self.station} – Stacked time series and KDE "
                f"(span: {_span:.1f} days; {start_dt:%Y-%m-%d} to {end_dt:%Y-%m-%d})",
                fontsize=13
            )
        except Exception:
            pass

        plt.tight_layout()

        if save_path:
            sp = Path(save_path)
            if sp.suffix == "":
                sp = sp.with_suffix(".png")
            fig.savefig(sp, dpi=300, bbox_inches='tight')
            print(f"    Saved time series figure to {sp}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig
    
    def plot_overpass_tod(self, constellations=None, elevation_min=None, elevation_max=None,
                          time_zone='UTC', min_points=50, bw_method=None, show_sv=False,
                          figsize=(10, 8), save_path=None, show=True, interactive=False, make=True,
                          viz: str = "kde"):
        """
        Plot time-of-day overpass distribution for satellites grouped by constellation (Figure A)
        and mean diurnal elevation angle per SV (Figure B).

        Parameters
        ----------
        constellations : list[str] | None
            Names like ['gps','glonass','galileo','beidou'] (case-insensitive).
        elevation_min, elevation_max : float | None
            Optional elevation angle filter (deg).
        time_zone : str
            Time zone for local time-of-day conversion.
        min_points : int
            Minimum points per SV to compute KDE (viz='kde').
        bw_method : float | str | None
            Bandwidth for gaussian_kde (viz='kde').
        figsize : tuple
            Figure size for both A and B.
        save_path : Path | None
            Base path; A and B saved as _A/_B variants if provided.
        show : bool
            Show figures.
        make : bool
            If False, skip plotting.
        viz : {'kde','hist'}
            Visualization for Figure A:
              'kde'  -> circular KDE density (default)
              'hist' -> 48 bins (0.5 h), step-wise density histogram (no fill)
        Returns
        -------
        (Figure|None, Figure|None)
            Tuple of figures (A: overpass distribution, B: mean diurnal elevation).
        """
        if not make:
            return None, None
        # Ensure parquet exists
        if self.vod_file_temp is None or not self.vod_file_temp.exists():
            print("VOD parquet file not found. Run process_vod first.")
            return None, None
        
        # Columns of interest
        usecols = ['Epoch', 'SV', 'Elevation']
        try:
            vod = pd.read_parquet(self.vod_file_temp, columns=usecols)
        except Exception:
            vod = pd.read_parquet(self.vod_file_temp)
            vod = vod[[c for c in usecols if c in vod.columns]]
        
        # make epoch a col from index
        if vod.index.name == 'Epoch':
            vod = vod.reset_index()
        
        if vod.empty or 'SV' not in vod.columns or 'Epoch' not in vod.columns:
            print("Required columns missing in parquet.")
            return None
        
        vod = vod.dropna(subset=['SV', 'Epoch']).copy()
        vod['SV'] = vod['SV'].astype(str)
        
        # Constellation filtering (first letter of SV, e.g. G21)
        # Provided mapping in processing.aux: constellation_map = {'G':'GPS','R':'GLONASS','E':'Galileo','C':'BeiDou','S':'SBAS'}
        # Normalize requested set
        if constellations is None:
            requested = {'gps', 'glonass', 'galileo', 'beidou'}
        else:
            requested = {c.lower() for c in constellations}
        
        letter_to_name = constellation_map  # already imported
        name_to_letter = {v.lower(): k for k, v in letter_to_name.items()}
        requested_letters = {name_to_letter[c] for c in requested if c in name_to_letter}
        
        vod['Constellation'] = vod['SV'].str[0].map(letter_to_name)
        vod = vod[vod['SV'].str[0].isin(requested_letters)]
        # Drop SBAS if accidentally included
        vod = vod[vod['SV'].str[0] != 'S']
        
        if vod.empty:
            print("No data after constellation filtering.")
            return None
        
        # Elevation filtering
        if elevation_min is not None:
            vod = vod[vod['Elevation'] >= elevation_min]
        if elevation_max is not None:
            vod = vod[vod['Elevation'] <= elevation_max]
        if vod.empty:
            print("No data after elevation filtering.")
            return None
        
        # Time-of-day (fractional hour) in given timezone
        vod['datetime'] = pd.to_datetime(vod['Epoch'])
        vod['datetime_local'] = vod['datetime'].dt.tz_localize('UTC').dt.tz_convert(time_zone)
        vod['hour_frac'] = (vod['datetime_local'].dt.hour +
                            vod['datetime_local'].dt.minute / 60.0 +
                            vod['datetime_local'].dt.second / 3600.0)
        
        # Group SVs per constellation -> dict: {Constellation: {SV: hour_array}}
        constellation_groups = {}
        for const in sorted(vod['Constellation'].dropna().unique()):
            sub = vod[vod['Constellation'] == const]
            constellation_groups[const] = {
                sv: sub[sub['SV'] == sv]['hour_frac'].values
                for sv in sorted(sub['SV'].unique())
            }
        
        if not constellation_groups:
            print("No constellation groups formed.")
            return None, None
        
        # Helper: build a gradient of n colors from a constellation base color
        from matplotlib.colors import to_rgb
        def gradient_colors_for_constellation(base_hex, n, start=0.35, end=1.0):
            base = np.array(to_rgb(base_hex))
            white = np.array([1.0, 1.0, 1.0])
            tvals = np.linspace(start, end, max(n, 1))
            # Blend white -> base to create perceptually clearer gradient
            cols = [(white * (1 - t) + base * t) for t in tvals]
            return np.clip(cols, 0, 1)
        
        # --------------------
        # Figure A: Distirbution of overpass times (density)
        # --------------------
        # Layout (max 2 columns)
        n_const = len(constellation_groups)
        cols = 2 if n_const > 1 else 1
        rows = int(np.ceil(n_const / cols))
        plasma = plt.get_cmap('plasma')
        
        from scipy.stats import gaussian_kde

        def circular_kde(hours_array):
            """Circular KDE over 0–24 using data wrapping (avoids taper at edges)."""
            if len(hours_array) < min_points:
                return None, None
            extended = np.concatenate([hours_array, hours_array + 24, hours_array - 24])
            try:
                kde = gaussian_kde(extended, bw_method=bw_method)
            except Exception:
                return None, None
            x_grid = np.linspace(0, 24, 288)  # 5-min resolution
            y = kde(x_grid)
            area = np.trapz(y, x_grid)
            if area > 0:
                y /= area
            return x_grid, y
        
        if viz not in ("kde", "hist"):
            raise ValueError("viz must be 'kde' or 'hist'")
        if viz == "hist":
            bin_edges = np.linspace(0.0, 24.0, 49)

        figA_series = {}
        for const, sv_dict in constellation_groups.items():
            entries = []
            sv_list = list(sv_dict.keys())
            n_svs = len(sv_list)
            sv_colors = plasma(np.linspace(0.1, 0.9, max(n_svs, 1)))
            for sv, col in zip(sv_list, sv_colors):
                hours = sv_dict[sv]
                if viz == "kde":
                    xg, yg = circular_kde(hours)
                    if xg is None:
                        continue
                    entries.append({
                        "sv": sv,
                        "kind": "kde",
                        "x": xg,
                        "y": yg,
                        "color": mpl.colors.to_hex(col),
                        "n_obs": len(hours),
                    })
                else:
                    if len(hours) == 0:
                        continue
                    counts, edges = np.histogram(hours, bins=bin_edges, density=False)
                    entries.append({
                        "sv": sv,
                        "kind": "hist",
                        "edges": edges,
                        "counts": counts,
                        "color": mpl.colors.to_hex(col),
                        "n_obs": len(hours),
                    })
            figA_series[const] = {"entries": entries, "n_svs": n_svs}

        _start_dt = vod['datetime'].min()
        _end_dt = vod['datetime'].max()
        _span = (_end_dt - _start_dt) / pd.Timedelta(days=1)
        figA_title = (
            f"Distribution of overpass times (span: {_span:.1f} days; {_start_dt:%Y-%m-%d} to {_end_dt:%Y-%m-%d})"
        )

        if not interactive:
            figA, axesA = plt.subplots(rows, cols, figsize=figsize, sharex=True, sharey=True)
            axesA = np.atleast_1d(axesA).ravel()
            for ax, const in zip(axesA, figA_series.keys()):
                data = figA_series[const]
                for entry in data["entries"]:
                    if entry["kind"] == "kde":
                        ax.plot(entry["x"], entry["y"], color=entry["color"], linewidth=0.9, alpha=0.95, label=entry["sv"])
                    else:
                        edges = entry["edges"]
                        counts = entry["counts"]
                        if len(counts) == 0:
                            continue
                        ax.step(edges[:-1], counts, where='post', color=entry["color"], linewidth=0.9, alpha=0.95,
                                label=entry["sv"])
                        ax.plot([24.0], [counts[0]], marker='o', markersize=0, color=entry["color"])
                # existing axis formatting/legend logic
                if data["n_svs"] > 0:
                    norm = mpl.colors.Normalize(vmin=1, vmax=max(data["n_svs"], 1))
                    sm = mpl.cm.ScalarMappable(cmap=plasma, norm=norm)
                    sm.set_array([])
                    cb = figA.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
                    cb.set_label('SV index (order)', fontsize=8)
                    cb.set_ticks([1] if data["n_svs"] <= 1 else [1, data["n_svs"]])
                    cb.ax.tick_params(labelsize=7)
            for j in range(len(figA_series), len(axesA)):
                axesA[j].set_visible(False)
            figA.suptitle(figA_title, fontsize=14)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
        else:
            from plotly.subplots import make_subplots
            import plotly.graph_objects as go

            figA = make_subplots(
                rows=rows,
                cols=cols,
                subplot_titles=[f"{c}  (#sats: {figA_series[c]['n_svs']})" for c in figA_series.keys()],
                shared_xaxes=True,
                shared_yaxes=True,
            )
            value_label = "Density" if viz == "kde" else "Count"
            value_format = ".3f" if viz == "kde" else ".0f"
            ordered_consts = list(figA_series.keys())
            for idx, const in enumerate(ordered_consts):
                row = idx // cols + 1
                col_idx = idx % cols + 1
                for entry in figA_series[const]["entries"]:
                    if entry["kind"] == "kde":
                        x_vals = entry["x"]
                        y_vals = entry["y"]
                        line_shape = "linear"
                    else:
                        counts = entry["counts"]
                        if len(counts) == 0:
                            continue
                        edges = entry["edges"]
                        x_vals = edges
                        y_vals = np.append(counts, counts[-1])
                        line_shape = "hv"
                    customdata = np.array([[const, entry["sv"]]] * len(x_vals), dtype=object)
                    hovertemplate = (
                        "Constellation: %{customdata[0]}<br>"
                        "SV: %{customdata[1]}<br>"
                        "Hour: %{x:.2f}<br>"
                        f"{value_label}: %{{y:{value_format}}}<extra></extra>"
                    )
                    figA.add_trace(
                        go.Scatter(
                            x=x_vals,
                            y=y_vals,
                            mode="lines",
                            line_shape=line_shape,
                            line=dict(color=entry["color"]),
                            name=entry["sv"],
                            hovertemplate=hovertemplate,
                            customdata=customdata,
                            showlegend=False,
                        ),
                        row=row,
                        col=col_idx,
                    )
                figA.update_xaxes(range=[0, 24], dtick=6, title_text="Hour of Day", row=row, col=col_idx)
                figA.update_yaxes(title_text=value_label, row=row, col=col_idx)
            figA.update_layout(
                title=figA_title,
                height=max(400, rows * 350),
                width=max(500, cols * 400),
                hovermode="closest",
                template="simple_white",
            )
            if save_path:
                sp = Path(save_path)
                if sp.suffix == "":
                    sp = sp.with_suffix(".html")
                spA = sp.with_name(sp.stem + "_A_interactive" + sp.suffix)
                figA.write_html(spA, include_plotlyjs='cdn')
                print(f"    Saved overpass figure (A, interactive) to {spA}")

        if not interactive and save_path:
            sp = Path(save_path)
            if sp.suffix == "":
                sp = sp.with_suffix(".png")
            spA = sp.with_name(sp.stem + "_A" + sp.suffix)
            figA.savefig(spA, dpi=300, bbox_inches='tight')
            print(f"    Saved overpass figure (A) to {spA}")

        # --------------------
        # Figure B: Mean diurnal elevation angle per SV
        # --------------------
        # Precompute 5-min bins for mean elevation
        vod['fh_rounded'] = (vod['hour_frac'] * 12).round() / 12.0  # 5-min bins
        
        figB, axesB = plt.subplots(rows, cols, figsize=figsize, sharex=True, sharey=True)
        axesB = np.atleast_1d(axesB).ravel()
        
        for ax, (const, _) in zip(axesB, constellation_groups.items()):
            # Subset for this constellation
            sub = vod[vod['Constellation'] == const]
            sv_list = sorted(sub['SV'].unique())
            n_svs = len(sv_list)
            # Use plasma colormap for satellites in this subplot
            plasma = plt.get_cmap('plasma')
            sv_colors = plasma(np.linspace(0.1, 0.9, max(n_svs, 1)))
            # Plot mean elevation per SV across time-of-day bins
            for sv, col in zip(sv_list, sv_colors):
                ssub = sub[sub['SV'] == sv]
                if ssub.empty:
                    continue
                stats = ssub.groupby('fh_rounded')['Elevation'].mean().dropna()
                if stats.empty:
                    continue
                x = stats.index.values
                y = stats.values
                if x[0] == 0 and x[-1] < 24:
                    x = np.append(x, 24.0)
                    y = np.append(y, y[0])
                ax.plot(x, y, color=col, linewidth=0.9, alpha=0.95, label=sv)
            ax.set_xlim(0, 24)
            ax.set_xticks([0, 6, 12, 18, 24])
            ax.set_xlabel("Hour of Day")
            ax.set_ylabel("Mean elevation (deg)")
            # Include number of satellites in title
            ax.set_title(f"{const}  (#sats: {n_svs})")
            ax.grid(alpha=0.3, linestyle='--')
            if 0 < n_svs <= 14:
                ax.legend(fontsize=7, ncol=2, frameon=False)
            # Add a plasma colorbar keyed to SV index order
            try:
                norm = mpl.colors.Normalize(vmin=1, vmax=max(n_svs, 1))
                sm = mpl.cm.ScalarMappable(cmap=plasma, norm=norm)
                sm.set_array([])
                cb = figB.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
                cb.set_label('SV index (order)', fontsize=8)
                if n_svs > 1:
                    cb.set_ticks([1, n_svs])
                else:
                    cb.set_ticks([1])
                cb.ax.tick_params(labelsize=7)
            except Exception:
                pass
        
        # Hide unused axes in B
        for j in range(len(constellation_groups), len(axesB)):
            axesB[j].set_visible(False)
        
        # Compute date span from filtered data
        _start_dt = vod['datetime'].min()
        _end_dt = vod['datetime'].max()
        _span = (_end_dt - _start_dt) / pd.Timedelta(days=1)
        
        figB.suptitle(
            f"Mean diurnal elevation angle (span: {_span:.1f} days; {_start_dt:%Y-%m-%d} to {_end_dt:%Y-%m-%d})",
            fontsize=14
        )
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        if save_path:
            sp = Path(save_path)
            if sp.suffix == "":
                sp = sp.with_suffix(".png")
            spB = sp.with_name(sp.stem + "_B" + sp.suffix)
            figB.savefig(spB, dpi=300, bbox_inches='tight')
            print(f"    Saved overpass figure (B) to {spB}")
        
        if show:
            if interactive and figA is not None:
                figA.show()
            plt.show()
        else:
            if isinstance(figA, plt.Figure):
                plt.close(figA)
            plt.close(figB)
        return figA, figB
