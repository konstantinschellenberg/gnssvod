#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from definitions import FIG
from processing.settings import time_interval, tz


def filter_by_time_intervals(df, time_intervals, time_intervals_tz, datetime_col=None):
    """
    Filter DataFrame to only include rows within any of the given time intervals.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with a DatetimeIndex or a datetime column
    time_intervals : list of tuple
        List of (start, end) tuples (inclusive)
    datetime_col : str, optional
        Name of the datetime column if not using the index
    time_intervals_tz : str
        Time zone to assign the time_intervals to

    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame
    """
    if datetime_col:
        times = pd.to_datetime(df[datetime_col])
    else:
        times = df.index

    mask = pd.Series(False, index=df.index)
    for start, end in time_intervals:
        start = pd.to_datetime(start).tz_localize(time_intervals_tz)
        end = pd.to_datetime(end).tz_localize(time_intervals_tz)
        mask |= (times >= start) & (times <= end)
    return df[mask]


def read_vod_timeseries(vod_file):
    """
    Read VOD time series from a NetCDF file and convert to pandas DataFrame.

    Parameters
    ----------
    vod_file : str or Path
        Path to the VOD timeseries NetCDF file

    Returns
    -------
    pandas.DataFrame
        VOD timeseries data with datetime as index
    """
    import xarray as xr
    
    # Read the NetCDF file
    ds = xr.open_dataset(vod_file)
    
    # Convert to pandas DataFrame
    df = ds.to_dataframe()
    
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
    df.index = df.index.tz_convert('etc/GMT+6')
    
    # add hour-of-day as a column (subminute)
    df['hod'] = df.index.get_level_values('datetime').hour + df.index.get_level_values('datetime').minute / 60.0
    # add day-of-year as a column
    df['doy'] = df.index.get_level_values('datetime').dayofyear
    df['year'] = df.index.get_level_values('datetime').year
    
    # select only falling into any number of temporal intervals in "time_intervals" list-of-tuples
    df = filter_by_time_intervals(df, time_interval, tz)
    
    return df


def plot_vod_fingerprint(df, variable, title=None, figsize=(4, 7), cmap="viridis", scaling=99, hue_limit=None):
    """
    Create a fingerprint plot showing VOD data as a heatmap with hour of day (x-axis)
    and day of year (y-axis).

    Parameters
    ----------
    df : pandas.DataFrame
        VOD time series data from read_vod_timeseries
    variable : str
        Variable name to plot as the heatmap color
    interactive : bool or str, default=False
        If "interactive", use Plotly for interactive plotting
        Otherwise use Matplotlib
    title : str, optional
        Plot title
    figsize : tuple, default=(4, 7)
        Figure size in inches (for non-interactive plots)
    cmap : str, default="viridis"
        Colormap to use for the heatmap
    scaling : int, default=99
        Percentile (1-100) to cap color scale. Values below (100-scaling)/2 percentile
        and above (100+scaling)/2 percentile will be capped.

    Returns
    -------
    fig : matplotlib.figure.Figure or plotly.graph_objects.Figure
        The created figure object
    """
    if 'hod' not in df.columns or 'doy' not in df.columns:
        raise ValueError("DataFrame must contain 'hod' and 'doy' columns")

    if variable not in df.columns:
        raise ValueError(f"Variable '{variable}' not found in DataFrame")

    # Create a pivot table for the heatmap
    pivot_data = df.pivot_table(
        values=variable,
        index=['year', 'doy'],
        columns='hod',
        aggfunc='mean'
    )
    
    # print perc nan per day
    plot_nan_bars = False
    if plot_nan_bars:
        doynan = df.groupby(['doy']).apply(lambda x: x[variable].isna().sum() / len(x) * 100, include_groups=False)
        from matplotlib import pyplot as plt
        doynan.plot(kind="bar", title="nan per doy", figsize=(8, 4)); plt.show()
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize)

    # Calculate percentile bounds for color scaling
    if hue_limit:
        vmin = hue_limit[0]
        vmax = hue_limit[1]
    else:
        if scaling < 100:
            lower_percentile = (100 - scaling)
            upper_percentile = 100 - lower_percentile
            vmin = np.nanpercentile(pivot_data.values, lower_percentile)
            vmax = np.nanpercentile(pivot_data.values, upper_percentile)
        else:
            vmin = None
            vmax = None
            
    # nan –> 0
    pivot_data = pivot_data.fillna(0)
    
    # make date the index, pd.DatetimeIndex
    pivot_data.index = pd.to_datetime(pivot_data.index.get_level_values('year').astype(str) + '-' +
                                        pivot_data.index.get_level_values('doy').astype(str),
                                        format='%Y-%j')

    # Create the heatmap with vmin and vmax for color scaling
    img = ax.pcolormesh(
        pivot_data.columns,  # hours
        pivot_data.index,    # days
        pivot_data.values,
        cmap=cmap,
        shading='auto',  # 'nearest' or 'auto'
        vmin=vmin,
        vmax=vmax
    )

    # Flip the y-axis so doy increases downward
    ax.invert_yaxis()

    # Add colorbar
    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label(variable)

    # Set labels and title
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Day of Year")
    ax.set_title(title if title else f"Fingerprint Plot of {variable}")

    # Set xticks to show hours
    ax.set_xticks(np.arange(0, 24, 3))
    ax.set_xticklabels(np.arange(0, 24, 3))

    # Set yticks to show days
    # ax.set_yticks(np.arange(0, 366, 30))
    # ax.set_yticklabels(np.arange(0, 366, 30))

    filename = f"vod_fingerprint_{variable}.png"
    plt.tight_layout()
    plt.savefig(FIG / filename, dpi=300)
    plt.show()


def plot_vod_timeseries(df, variables, interactive=False, title=None, figsize=(8, 5)):
    """
    Plot VOD time series data for specified variables.

    Parameters
    ----------
    df : pandas.DataFrame
        VOD time series data from read_vod_timeseries
    variables : list
        List of variable names to plot
    interactive : bool or str, default=False
        If "interactive", use Plotly for interactive plotting
        Otherwise use Matplotlib
    title : str, optional
        Plot title
    figsize : tuple, default=(10, 6)
        Figure size in inches (for non-interactive plots)
    filename : str, optional
        If provided, save the plot to this file

    Returns
    -------
    fig : matplotlib.figure.Figure or plotly.graph_objects.Figure
        The created figure object
    """
    # Extract datetime values
    if isinstance(df.index, pd.MultiIndex):
        x = df.index.get_level_values('datetime')
    else:
        x = df.index
    
    if interactive == "interactive":
        # Plotly version
        import plotly.graph_objects as go
        
        fig = go.Figure()
        
        for var in variables:
            if var in df.columns:
                fig.add_trace(go.Scatter(
                    x=x,
                    y=df[var],
                    mode='lines',
                    name=var
                ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="GNSS-VOD",
            legend_title="Variables",
            height=600,
            width=900
        )
        
        fig.show()
    else:
        # Matplotlib version
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        
        fig, ax = plt.subplots(figsize=figsize)
        
        for var in variables:
            if var in df.columns:
                ax.plot(x, df[var], label=var)
        
        # Format date ticks
        fig.autofmt_xdate()
        
        ax.set_xlabel('Date')
        ax.set_ylabel('GNSS-VOD')
        ax.legend()
        
        if title:
            ax.set_title(title)
        
        plt.tight_layout()
        
        filename = f"vod_timeseries_{variables[0]}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()


def plot_vod_diurnal(df, show_std=False, figsize=(8, 6), title=None, filename="vod_diurnal_plot.png",
                     algos=None, compare_anomalies=True, diff=False):
    """
    Create a 2x2 matrix of diurnal plots for VOD variables with algorithm suffixes.

    Parameters
    ----------
    df : pandas.DataFrame
        VOD time series data with algorithm suffixes (e.g., VOD1_tp, VOD1_tps)
    show_std : bool, default=True
        Whether to show ±1 standard deviation ribbon
    figsize : tuple, default=(12, 8)
        Figure size in inches
    title : str, optional
        Overall plot title
    filename : str, default="vod_diurnal_plot.png"
        Name to use when saving the plot
    algos : list, optional
        List of algorithm suffixes to include (e.g., ['tp', 'tps'])
        If None, will try to detect from columns
    compare_anomalies : bool, default=True
        Whether to add orange lines comparing anomalies between algorithms

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # If no algorithms specified, try to detect them from the column names
    if algos is None:
        # Extract algorithm suffixes from column names (e.g., VOD1_tp, VOD1_tps)
        all_cols = df.columns
        algos = set()
        for col in all_cols:
            if col.startswith('VOD') and '_' in col:
                parts = col.split('_')
                if len(parts) >= 2 and parts[0] in ['VOD1', 'VOD2']:
                    # For columns like VOD1_tp, VOD1_anom_tp
                    if parts[-1] not in ['anom', 'std']:
                        algos.add(parts[-1])
                    elif len(parts) >= 3:
                        algos.add(parts[-1])
        
        algos = sorted(list(algos))
        if not algos:
            raise ValueError("Could not detect algorithm suffixes in column names")
        print(f"Detected algorithms: {', '.join(algos)}")
    
    # Create figure and subplots with shared x-axis
    fig, axs = plt.subplots(2, 2, figsize=figsize, sharex=True)
    
    # Define the base variables for each subplot
    subplot_vars = [
        ('VOD1', axs[0, 0]),
        ('VOD1_anom', axs[0, 1]),
        ('VOD2', axs[1, 0]),
        ('VOD2_anom', axs[1, 1])
    ]
    
    # Calculate hour-of-day aggregated values
    grouped = df.groupby('hod')
    
    # Set of colors for different algorithms (excluding orange which is reserved for comparison)
    algo_colors = plt.cm.Set2.colors
    
    # get common y-axis limits for each row
    y_limits = {}
    
    # Plot each variable for each algorithm
    for i, (base_var, ax) in enumerate(subplot_vars):
        for j, algo in enumerate(algos):
            # Construct column name with algorithm suffix
            var_name = f"{base_var}_{algo}"
            
            # Skip if column doesn't exist
            if var_name not in df.columns:
                continue
            
            # Calculate mean by hour of day
            var_mean = grouped[var_name].mean()
            
            if not show_std:
                # Initialize y_limits for this variable if not already done
                if base_var not in y_limits:
                    y_limits[base_var] = (var_mean.min(), var_mean.max())
                else:
                    # Update y_limits with current variable's min/max
                    y_limits[base_var] = (
                        min(y_limits[base_var][0], var_mean.min()),
                        max(y_limits[base_var][1], var_mean.max())
                    )
            
            # Plot the mean line
            color = algo_colors[j % len(algo_colors)]
            ax.plot(var_mean.index, var_mean.values, '-',
                    linewidth=2, label=f"{var_name}",
                    color=color)
            
            # Add std ribbon if requested and available
            std_col = f"{var_name}_std"
            if show_std and std_col in df.columns:
                var_std = grouped[std_col].mean()
                ax.fill_between(
                    var_mean.index,
                    var_mean.values - var_std.values,
                    var_mean.values + var_std.values,
                    alpha=0.2,
                    color=color
                )
                y_limits.append((var_mean.values - var_std.values).min(),
                                (var_mean.values + var_std.values).max())
        
        # Add comparison lines for anomalies if requested
        if compare_anomalies and '_anom' in base_var and len(algos) >= 2:
            # Default comparison is between first two algorithms
            algo1, algo2 = algos[0], algos[1]
            var1 = f"{base_var}_{algo1}"
            var2 = f"{base_var}_{algo2}"
            
            if diff:
                if var1 in df.columns and var2 in df.columns:
                    diff = grouped[var1].mean() - grouped[var2].mean()
                    ax.plot(diff.index, diff.values, '-',
                            linewidth=2,
                            color='orange',
                            label=f"{algo1} - {algo2}")
            
        # Set labels and grid
        ax.set_title(base_var)
        ax.set_xlabel("Hour of Day")
        ax.set_xticks(np.arange(0, 24, 3))
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize='small')
        
    # Set y-axis limits for each row
    if not show_std:
        # Set y-axis limits for each row
        for i, (base_var, ax) in enumerate(subplot_vars):
            # Determine the row (0 for VOD1 and VOD1_anom, 1 for VOD2 and VOD2_anom)
            row = i // 2
            if row == 0:
                # Row 0: VOD1 and VOD1_anom
                y_min = min(y_limits.get('VOD1', (0, 0))[0], y_limits.get('VOD1_anom', (0, 0))[0])
                y_max = max(y_limits.get('VOD1', (0, 0))[1], y_limits.get('VOD1_anom', (0, 0))[1])
            else:
                # Row 1: VOD2 and VOD2_anom
                y_min = min(y_limits.get('VOD2', (0, 0))[0], y_limits.get('VOD2_anom', (0, 0))[0])
                y_max = max(y_limits.get('VOD2', (0, 0))[1], y_limits.get('VOD2_anom', (0, 0))[1])
            
            # Apply the y-axis limits
            ax.set_ylim(y_min, y_max)
        
    # Set common y-axis labels
    axs[0, 0].set_ylabel("GNSS-VOD")
    axs[1, 0].set_ylabel("GNSS-VOD")
    
    # Set overall title if provided
    if title:
        fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    
    # Save figure to FIG directory
    plt.savefig(FIG / filename, dpi=300)
    plt.show()
    
    return fig

def plot_vod_scatter(df, **kwargs):
    """
    Create a scatter plot of VOD1 (1575.42 MHz) vs VOD2 (1227.60 MHz).

    Parameters
    ----------
    df : pandas.DataFrame
        VOD time series data from read_vod_timeseries
    **kwargs : dict
        Keyword arguments for customizing the plot:
        - hue : str, default="hod"
            Column name for color coding points
        - point_size : float, default=1.0
            Size of scatter points
        - add_linear_fit : bool, default=False
            Whether to add a linear regression line
        - figsize : tuple, default=(8, 8)
            Figure size in inches
        - cmap : str, default="viridis"
            Colormap for the scatter points
        - alpha : float, default=0.5
            Transparency of points
        - title : str, optional
            Plot title
        - filename : str, optional
            If provided, save the plot to this file

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats
    
    # Default parameters
    hue = kwargs.get('hue', 'hod')
    point_size = kwargs.get('point_size', 1.0)
    add_linear_fit = kwargs.get('add_linear_fit', False)
    figsize = kwargs.get('figsize', (8, 8))
    cmap = kwargs.get('cmap', 'viridis')
    alpha = kwargs.get('alpha', 0.5)
    title = kwargs.get('title', 'VOD1 vs VOD2 Scatter Plot')
    filename = kwargs.get('filename', 'vod_scatter.png')
    cutoff_percentile = kwargs.get('cutoff_percentile', 99)
    
    # Check if required columns exist
    required_vars = ['VOD1', 'VOD2']
    for var in required_vars:
        if var not in df.columns:
            raise ValueError(f"Required variable '{var}' not found in DataFrame")
    
    if hue not in df.columns:
        raise ValueError(f"Hue variable '{hue}' not found in DataFrame")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate lower and upper bounds for both VOD1 and VOD2
    lower = (100 - cutoff_percentile)
    upper = 100 - lower
    v1_min, v1_max = np.nanpercentile(df['VOD1'], [lower, upper])
    v2_min, v2_max = np.nanpercentile(df['VOD2'], [lower, upper])
    
    # Filter data within percentile bounds
    mask = (
            # (df['VOD1'] >= v1_min) & (df['VOD1'] <= v1_max) &
            (df['VOD2'] >= v2_min) & (df['VOD2'] <= v2_max)
    )
    df = df[mask]
    
    # Create scatter plot
    sc = ax.scatter(
        df['VOD1'],
        df['VOD2'],
        c=df[hue],
        s=point_size,
        cmap=cmap,
        alpha=alpha,
        edgecolor='none'
    )
    
    # Add a colorbar
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(hue)
    
    # Add linear regression line if requested
    if add_linear_fit:
        mask = ~np.isnan(df['VOD1']) & ~np.isnan(df['VOD2'])
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            df['VOD1'][mask], df['VOD2'][mask]
        )
        
        x_min, x_max = ax.get_xlim()
        x_line = np.linspace(x_min, x_max, 100)
        y_line = slope * x_line + intercept
        
        ax.plot(
            x_line, y_line, '-', color='black', linewidth=1,
            label=f'y = {slope:.3f}x + {intercept:.3f}\n$R^2 = {r_value ** 2:.3f}$'
        )
        ax.legend()
    
    # Labels
    ax.set_xlabel('VOD1 (1575.42 MHz)')
    ax.set_ylabel('VOD2 (1227.60 MHz)')
    ax.set_title(f"{title}\ncutoff percentile: {cutoff_percentile}%")
    
    # Set equal aspect ratio
    ax.set_aspect('equal', adjustable="datalim")
    
    # Grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Save figure if filename is provided
    plt.tight_layout()
    plt.savefig(FIG / filename, dpi=300)
    plt.show()



def plot_daily_diurnal_range(df, vars_to_plot=['VOD1', 'VOD2'], figsize=(8, 5), title=None, filename=None, qq99=False):
    """
    Plot the daily diurnal range (max-min) for specified variables.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the VOD data with 'hod' (hour of day) and 'datetime' columns.
    vars_to_plot : list, default=['VOD1', 'VOD2']
        List of variable names to calculate and plot the diurnal range for.
    figsize : tuple, default=(10, 6)
        Size of the plot.
    title : str, optional
        Title of the plot.
    filename : str, optional
        If provided, saves the plot to the specified file.
    """
    
    import matplotlib.pyplot as plt
    
    # datetime can be index
    if 'hod' not in df.columns:
        raise ValueError("DataFrame must contain 'hod' and 'datetime' columns.")
    
    df = df.copy()
    df.reset_index(inplace=True)  # Ensure 'datetime' is a column if it was an index

    # Group by date and calculate nighttime and midday means
    df['date'] = df['datetime'].dt.date
    nighttime = df[(df['hod'] >= 0) & (df['hod'] < 4)].groupby('date')[vars_to_plot].mean()
    midday = df[(df['hod'] >= 12) & (df['hod'] < 16)].groupby('date')[vars_to_plot].mean()

    # Calculate diurnal range (midday - nighttime)
    diurnal_range = midday - nighttime
    
    # calc qq99 bothsided for diurnal range
    lower = (100 - qq99)
    upper = 100 - lower
    vmin = np.nanpercentile(diurnal_range.values, lower)
    vmax = np.nanpercentile(diurnal_range.values, upper)
    ylims = (vmin, vmax)

    # Plot the diurnal range
    plt.figure(figsize=figsize)
    for var in vars_to_plot:
        if var in diurnal_range.columns:
            plt.plot(diurnal_range.index, diurnal_range[var], label=f"{var} Diurnal Range")

    # Add labels, title, and legend
    plt.xlabel("Date")
    plt.ylabel("Diurnal Range (Midday - Nighttime)")
    if qq99:
        plt.ylim(ylims)
    plt.title(title if title else "Daily Diurnal Range")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Save the plot if a filename is provided
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')

    plt.show()