#!/usr/bin/env python
# -*- coding: utf-8 -*-
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from definitions import DATA
import gnssvod as gv
from processing.filepattern_finder import create_time_filter_patterns
from processing.settings import bands, station, time_interval


class VODProcessor:
    def __init__(self, station=station, bands=bands, time_interval=time_interval,
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
        self.station = station
        self.bands = bands
        self.band_ids = list(bands.keys())
        self.time_interval = time_interval
        self.plot = plot
        self.overwrite = overwrite
        self.recover_snr = recover_snr
        
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
    
    def process_vod(self, angular_resolutions=[0.5], angular_cutoffs=[10],
                    temporal_resolutions=[60], max_workers=15):
        """
        Process VOD data with multiple parameter combinations in parallel.

        Parameters
        ----------
        angular_resolutions : list
            List of angular resolution values to test
        angular_cutoffs : list
            List of angular cutoff values to test
        temporal_resolutions : list
            List of temporal resolution values to test
        max_workers : int, optional
            Maximum number of parallel workers

        Returns
        -------
        xarray.Dataset
            Combined results from all parameter combinations
        """
        print(
            f"Processing VOD data for {self.station} with {len(angular_resolutions) * len(angular_cutoffs) * len(temporal_resolutions)} parameter combinations")
        
        # Calculate VOD
        self.vod = gv.calc_vod(
            self.pattern,
            self.pairings,
            self.bands,
            self.time_interval,
            recover_snr=self.recover_snr
        )[self.station]
        
        print("NaN values in VOD:")
        print(self.vod.isna().mean() * 100)
        
        # Generate all parameter combinations
        param_combinations = []
        for ar in angular_resolutions:
            for ac in angular_cutoffs:
                for tr in temporal_resolutions:
                    param_combinations.append({
                        'angular_resolution': ar,
                        'angular_cutoff': ac,
                        'temporal_resolution': tr
                    })
        
        # Process parameter combinations in parallel
        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._run_parameter_combination, params)
                       for params in param_combinations]
            
            for future in futures:
                result = future.result()
                if result is not None:
                    results.append(result)
        
        # Combine results into a single xarray dataset
        self.results = self._combine_results(results)
        
        return self.results
    
    def build_hemi(self, angular_resolution, angular_cutoff):
        """
        Build hemispheric grid for VOD data.

        Parameters
        ----------
        angular_resolution : float
            Resolution of angular grid
        angular_cutoff : float
            Cutoff angle for the grid

        Returns
        -------
        tuple
            (hemi, patches, vod_with_cells)
        """
        # Initialize hemispheric grid
        hemi = gv.hemibuild(angular_resolution, angular_cutoff)
        
        # Get patches for plotting
        patches = hemi.patches()
        
        # Classify VOD into grid cells
        vod_with_cells = hemi.add_CellID(self.vod.copy()).drop(columns=['Azimuth', 'Elevation'])
        
        return hemi, patches, vod_with_cells
    
    def calculate_anomaly(self, vod_with_cells, temporal_resolution):
        """
        Calculate VOD anomalies using both methods.

        Parameters
        ----------
        vod_with_cells : pandas.DataFrame
            VOD data with cell IDs
        temporal_resolution : int
            Temporal resolution in minutes

        Returns
        -------
        tuple
            (vod_ts_1, vod_ts_2, vod_ts_combined)
        """
        # Method 1: Vincent's method (phi_theta)
        # Get average value per grid cell
        vod_avg = vod_with_cells.groupby(['CellID']).agg(['mean', 'std', 'count'])
        vod_avg.columns = ["_".join(x) for x in vod_avg.columns.to_flat_index()]
        
        # Calculate anomaly
        vod_anom = vod_with_cells.join(vod_avg, on='CellID')
        for band in self.band_ids:
            vod_anom[f"{band}_anom"] = vod_anom[band] - vod_anom[f"{band}_mean"]
        
        # Temporal aggregation
        vod_ts_1 = vod_anom.groupby(pd.Grouper(freq=f"{temporal_resolution}min", level='Epoch')).agg('mean')
        for band in self.band_ids:
            vod_ts_1[f"{band}_anom"] = vod_ts_1[f"{band}_anom"] + vod_ts_1[f"{band}"].mean()
        
        # Method 2: Konstantin's extension (phi_theta_sv)
        vod_ts_all = []
        
        # Process each satellite vehicle separately
        for sv in vod_with_cells.index.get_level_values('SV').unique():
            # Extract data for this SV only
            vod_sv = vod_with_cells.xs(sv, level='SV')
            
            # Skip if there's not enough data for this SV
            if len(vod_sv) < 100:
                continue
            
            # Calculate average values per grid cell for this specific SV
            vod_avg_sv = vod_sv.groupby(['CellID']).agg(['mean', 'std', 'count'])
            vod_avg_sv.columns = ["_".join(x) for x in vod_avg_sv.columns.to_flat_index()]
            
            # Join the cell averages back to the original data
            vod_anom_sv = vod_sv.join(vod_avg_sv, on='CellID')
            
            # Calculate anomalies for each band
            for band in self.band_ids:
                vod_anom_sv[f"{band}_anom"] = vod_anom_sv[band] - vod_anom_sv[f"{band}_mean"]
            
            # Add SV as a column before appending
            vod_anom_sv['SV'] = sv
            vod_anom_sv = vod_anom_sv.reset_index()
            vod_ts_all.append(vod_anom_sv)
        
        # Combine all SV results
        if vod_ts_all:
            vod_ts_svs = pd.concat(vod_ts_all).set_index(['Epoch', 'SV'])
            
            # Temporal aggregation
            vod_ts_2 = vod_ts_svs.groupby(pd.Grouper(freq=f"{temporal_resolution}min", level='Epoch')).agg('mean')
            
            # Add back the mean
            for band in self.band_ids:
                vod_ts_2[f"{band}_anom"] = vod_ts_2[f"{band}_anom"] + vod_ts_2[f"{band}"].mean()
        else:
            vod_ts_2 = pd.DataFrame()
        
        # Combine both methods
        if not vod_ts_1.empty:
            vod_ts_1['algo'] = 'tp'  # Vincent's method
        if not vod_ts_2.empty:
            vod_ts_2['algo'] = 'tps'  # Konstantin's extension
        
        # Concatenate results
        vod_ts_combined = pd.concat([vod_ts_1, vod_ts_2], axis=0)
        
        # Drop CellID if present
        vod_ts_combined = vod_ts_combined.drop(columns=['CellID'], errors='ignore')
        
        return vod_ts_1, vod_ts_2, vod_ts_combined
    
    def _run_parameter_combination(self, params):
        """
        Process a single parameter combination.

        Parameters
        ----------
        params : dict
            Dictionary of parameters

        Returns
        -------
        dict
            Results for this parameter combination
        """
        try:
            print(f"Processing combination: {params}")
            
            # Build hemispheric grid
            _, _, vod_with_cells = self.build_hemi(
                params['angular_resolution'],
                params['angular_cutoff']
            )
            
            # Calculate anomalies
            _, _, vod_ts_combined = self.calculate_anomaly(
                vod_with_cells,
                params['temporal_resolution']
            )
            
            # Add parameter columns
            vod_ts_combined['angular_resolution'] = params['angular_resolution']
            vod_ts_combined['angular_cutoff'] = params['angular_cutoff']
            vod_ts_combined['temporal_resolution'] = params['temporal_resolution']
            
            return {
                'params': params,
                'data': vod_ts_combined
            }
        except Exception as e:
            print(f"Error processing combination {params}: {str(e)}")
            return None
    
    def _combine_results(self, results):
        """
        Combine results from different parameter combinations into an xarray dataset.

        Parameters
        ----------
        results : list
            List of result dictionaries

        Returns
        -------
        xarray.Dataset
            Combined results
        """
        if not results:
            return None
        
        # Combine all DataFrames
        combined_df = pd.concat([r['data'] for r in results])
        
        # Convert to xarray
        ds = combined_df.to_xarray()
        
        # Save the combined results
        filename = f"vod_multi_param_{self.station}_{pd.Timestamp.now().strftime('%Y%m%d')}"
        
        # Save to NetCDF
        output_path = DATA / 'ard' / f"{filename}.nc"
        ds.to_netcdf(output_path)
        print(f"Saved combined results to {output_path}")
        
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
        
        # Filter to only include algo='tps'
        df_tps = df[df['algo'] == algo]
        
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
    
    def plot_by_parameter(self, gnssband="VOD1", algo="tps", figsize=(12, 6), save_dir=None, show=True,
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
            cmap = plt.cm.viridis
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
            param_info = ", ".join([f"{p}={v}" for p, v in fixed_params.items()])
            
            # Set the title with parameter information
            fig.suptitle(f"VOD Analysis - {self.station}\n"
                         f"Effect of {param} on VOD anomalies (Algorithm: tps)\n"
                         f"Fixed parameters: {param_info}",
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
                                 label=f"{param}={value}", color=colors[i], linewidth=0.8)
                
                # Diurnal cycle plot using pandas
                for band in band_cols:
                    # Group by hour of day to get mean values
                    diurnal_data = param_data.groupby('hod')[band].mean().reset_index()
                    diurnal_data.plot(x='hod', y=band, ax=axes[1],
                                      label=f"{param}={value}", color=colors[i], linewidth=0.8)
            
            # Configure time series plot
            axes[0].set_xlabel('Date')
            axes[0].set_ylabel(f'{gnssband} Anomaly')
            axes[0].set_title('Time Series')
            axes[0].grid(True, linestyle='--', alpha=0.7)
            
            # Configure diurnal cycle plot
            axes[1].set_xlabel('Hour of Day (Local Time)')
            axes[1].set_ylabel(f'{gnssband} Anomaly')
            axes[1].set_title('Diurnal Cycle')
            axes[1].set_xlim(0, 24)
            axes[1].grid(True, linestyle='--', alpha=0.7)
            
            # Apply y-limits if provided
            if y_limits is not None and gnssband in y_limits:
                for ax in axes:
                    ax.set_ylim(y_limits[gnssband])
            
            # Add legend (only to the second plot to save space)
            axes[1].legend(title=param)
            
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