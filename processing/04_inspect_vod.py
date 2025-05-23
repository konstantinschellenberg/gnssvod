#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import cmocean

from definitions import DATA, FIG



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
    
    return df

def plot_vod_fingerprint(df, variable, interactive=False, title=None, figsize=(4, 7), cmap="viridis", scaling=99):
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
        index='doy',
        columns='hod',
        aggfunc='mean'
    )

    # Matplotlib version
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=figsize)

    # Calculate percentile bounds for color scaling
    if scaling < 100:
        lower_percentile = (100 - scaling) / 2
        upper_percentile = 100 - lower_percentile
        vmin = np.nanpercentile(pivot_data.values, lower_percentile)
        vmax = np.nanpercentile(pivot_data.values, upper_percentile)
    else:
        vmin = None
        vmax = None

    # Create the heatmap with vmin and vmax for color scaling
    img = ax.pcolormesh(
        pivot_data.columns,  # hours
        pivot_data.index,    # days
        pivot_data.values,
        cmap=cmap,
        shading='auto',
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
    ax.set_yticks(np.arange(0, 366, 30))
    ax.set_yticklabels(np.arange(0, 366, 30))

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

def plot_vod_diurnal(df, show_std=True, figsize=(12, 8), title=None, filename="vod_diurnal_plot.png"):
    """
    Create a 2x2 matrix of diurnal plots for VOD1, VOD1_anom, VOD2, and VOD2_anom.
    Each plot shows the average value by hour of day with optional ±1 std ribbon.

    Parameters
    ----------
    df : pandas.DataFrame
        VOD time series data from read_vod_timeseries
    show_std : bool, default=True
        Whether to show ±1 standard deviation ribbon
    figsize : tuple, default=(12, 8)
        Figure size in inches
    title : str, optional
        Overall plot title
    filename : str, default="vod_diurnal_plot.png"
        Name to use when saving the plot

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Check if required columns exist
    required_vars = ['VOD1', 'VOD1_anom', 'VOD2', 'VOD2_anom']
    std_vars = ['VOD1_std', 'VOD1_anom_std', 'VOD2_std', 'VOD2_anom_std']
    
    for var in required_vars:
        if var not in df.columns:
            raise ValueError(f"Required variable '{var}' not found in DataFrame")
    
    if show_std:
        for var in std_vars:
            if var not in df.columns:
                print(f"Warning: Standard deviation column '{var}' not found")
    
    # Create figure and subplots with shared x-axis
    fig, axs = plt.subplots(2, 2, figsize=figsize, sharex=True)
    
    # Define the variables and their corresponding subplot
    subplot_vars = [
        ('VOD1', axs[0, 0]),
        ('VOD1_anom', axs[0, 1]),
        ('VOD2', axs[1, 0]),
        ('VOD2_anom', axs[1, 1])
    ]
    
    # Calculate hour-of-day aggregated values
    grouped = df.groupby('hod')
    
    # Plot each variable
    for var_name, ax in subplot_vars:
        # Calculate mean by hour of day
        var_mean = grouped[var_name].mean()
        
        # Plot the mean line
        ax.plot(var_mean.index, var_mean.values, '-', linewidth=2, label=var_name)
        
        # Add std ribbon if requested and available
        std_col = f"{var_name}_std"
        if show_std and std_col in df.columns:
            var_std = grouped[std_col].mean()
            ax.fill_between(
                var_mean.index,
                var_mean.values - var_std.values,
                var_mean.values + var_std.values,
                alpha=0.3,
                label=f"±1σ"
            )
        
        # Set labels and grid
        ax.set_title(var_name)
        ax.set_xlabel("Hour of Day")
        ax.set_xticks(np.arange(0, 24, 3))
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
    
    # Match y-axes within rows
    # First row: VOD1 and VOD1_anom
    y_min_row1 = min(axs[0, 0].get_ylim()[0], axs[0, 1].get_ylim()[0])
    y_max_row1 = max(axs[0, 0].get_ylim()[1], axs[0, 1].get_ylim()[1])
    axs[0, 0].set_ylim(y_min_row1, y_max_row1)
    axs[0, 1].set_ylim(y_min_row1, y_max_row1)
    
    # Second row: VOD2 and VOD2_anom
    y_min_row2 = min(axs[1, 0].get_ylim()[0], axs[1, 1].get_ylim()[0])
    y_max_row2 = max(axs[1, 0].get_ylim()[1], axs[1, 1].get_ylim()[1])
    axs[1, 0].set_ylim(y_min_row2, y_max_row2)
    axs[1, 1].set_ylim(y_min_row2, y_max_row2)
    
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
    

if __name__ == '__main__':
    
    
    """
    What happened with the missing data in the second half of 2024?
    """
    
    # -----------------------------------
    vod_file = DATA / "timeseries" / "vod_timeseries_MOz_20240101_20241230.nc"
    vod_ts = read_vod_timeseries(vod_file)
    
    # -----------------------------------
    # show all pandas columns
    
    pd.set_option('display.max_columns', None)
    vod_ts.describe()
    
    # -----------------------------------
    # time series plot
    # Static matplotlib plot
    
    plot = False
    if plot:
        plot_vod_timeseries(vod_ts, ['VOD1', 'VOD1_anom'], title="VOD Time Series")
        plot_vod_timeseries(vod_ts, ['VOD2', 'VOD2_anom'], title="VOD Time Series")
    
        # Interactive plotly plot
        plot_vod_timeseries(vod_ts, ['VOD1_anom', 'VOD1_std'], interactive="interactive")
        
        # Save plot to file
        plot_vod_timeseries(vod_ts, ['VOD1_anom', 'VOD2_anom'], filename="vod_timeseries.png")
    
    # -----------------------------------
    # diurnal plot
    plot = False
    if plot:
        # Create the 2x2 diurnal plot
        plot_vod_diurnal(vod_ts, show_std=False,
                         figsize=(8, 6),
                         title="VOD Diurnal Patterns",
                         filename="vod_diurnal_patterns.png")
    
    # -----------------------------------
    # fingerprint plot
    plot = False
    if plot:
        # plot_vod_fingerprint(vod_ts, 'VOD1', title="Fingerprint Plot of VOD1")
        plot_vod_fingerprint(vod_ts, 'VOD1_anom', title="Fingerprint Plot of VOD1 (anomaly)")
        # plot_vod_fingerprint(vod_ts, 'VOD2', title="Fingerprint Plot of VOD2")
        # plot_vod_fingerprint(vod_ts, 'VOD2_anom', title="Fingerprint Plot of VOD2 (anomaly)")

    # -----------------------------------
    # polarimetry
    # Basic scatter plot colored by hour of day
    plot = True
    if plot:
        
        # With linear fit and custom settings
        plot_vod_scatter(
            vod_ts,
            hue='doy',
            point_size=1,
            add_linear_fit=True,
            cmap='plasma',
            figsize=(5,4),
            title='VOD Frequency Relationship',
            filename='vod_frequency_scatter.png'
        )
        
        plot_vod_scatter(
            vod_ts,
            hue='hod',
            point_size=1,
            add_linear_fit=True,
            cmap=cmocean.cm.balance,
            figsize=(5,4),
            title='VOD Frequency Relationship',
            filename='vod_frequency_scatter.png'
        )