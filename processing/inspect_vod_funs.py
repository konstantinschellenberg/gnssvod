#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from definitions import FIG
from processing.settings import single_file_interval, visualization_timezone
import matplotlib.pyplot as plt


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


def plot_vod_timeseries(df, variables, interactive=False, title=None, figsize=(8, 5), **kwargs):
    """
    Plot VOD time series data for specified variables with customizable styling.

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
    figsize : tuple, default=(8, 5)
        Figure size in inches (for non-interactive plots)
    **kwargs : dict
        Additional customization options:
        - linewidth, lw : float, default=1.0
            Line width for plots
        - colors : list or str, default=None
            Colors for each variable (uses default color cycle if None)
        - linestyle, ls : str or list, default='-'
            Line style(s) for plots
        - alpha : float, default=1.0
            Transparency of lines
        - marker : str, default=''
            Marker style for data points
        - ylim : tuple, default=None
            Y-axis limits (min, max)
        - legend_loc : str, default='best'
            Legend location
        - grid : bool, default=True
            Whether to show grid lines
        - filename : str, optional
            Custom filename for saving the plot
        - daily_average : bool, default=False
            If True, also plot daily averages at midday with darker colors

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
    
    # Extract styling parameters from kwargs
    linewidth = kwargs.get('linewidth', kwargs.get('lw', 1.0))
    colors = kwargs.get('colors', None)
    linestyle = kwargs.get('linestyle', kwargs.get('ls', '-'))
    alpha = kwargs.get('alpha', 1.0)
    marker = kwargs.get('marker', '')
    ylim = kwargs.get('ylim', None)
    legend_loc = kwargs.get('legend_loc', 'upper left')
    grid = kwargs.get('grid', True)
    filename = kwargs.get('filename', None)
    daily_average = kwargs.get('daily_average', False)
    
    if interactive == "interactive":
        # Plotly version
        import plotly.graph_objects as go
        
        fig = go.Figure()
        
        for i, var in enumerate(variables):
            if var in df.columns:
                # Get color for this variable if specified
                color = colors[i] if isinstance(colors, list) and i < len(colors) else colors
                
                fig.add_trace(go.Scatter(
                    x=x,
                    y=df[var],
                    mode='lines' if not marker else 'lines+markers',
                    name=var,
                    line=dict(
                        width=linewidth if isinstance(linewidth, (int, float)) else linewidth[i],
                        dash='solid' if linestyle == '-' else 'dash',
                        color=color
                    ),
                    opacity=alpha,
                    marker=dict(symbol=marker) if marker else dict()
                ))
                
                # Add daily averages if requested
                if daily_average:
                    # Calculate daily averages
                    df_copy = df.copy()
                    df_copy['date'] = df_copy.index.date
                    daily_avg = df_copy.groupby('date')[var].mean()
                    
                    # Create midday timestamps for each day
                    midday_times = [pd.Timestamp(d).replace(hour=12, minute=0, second=0)
                                    for d in daily_avg.index]
                    
                    # Darken the color for daily averages
                    darker_color = color  # We'll use plotly's colorscale functions if color is specified
                    
                    fig.add_trace(go.Scatter(
                        x=midday_times,
                        y=daily_avg.values,
                        mode='lines+markers',
                        name=f"{var} (daily avg)",
                        line=dict(
                            width=linewidth * 1.5 if isinstance(linewidth, (int, float)) else linewidth[i] * 1.5,
                            color=darker_color
                        ),
                        alpha=1,
                        # marker=dict(size=6),
                        opacity=1.0
                    ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="GNSS-VOD",
            legend_title="Variables",
            height=600,
            width=900
        )
        
        if ylim:
            fig.update_layout(yaxis_range=ylim)
        
        fig.show()
    else:
        # Matplotlib version
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import matplotlib.colors as mcolors
        
        fig, ax = plt.subplots(figsize=figsize)
        
        for i, var in enumerate(variables):
            if var in df.columns:
                # Handle list parameters
                lw = linewidth[i] if isinstance(linewidth, list) and i < len(linewidth) else linewidth
                ls = linestyle[i] if isinstance(linestyle, list) and i < len(linestyle) else linestyle
                
                # Get color from default cycle if not specified
                if colors is None:
                    c = plt.rcParams['axes.prop_cycle'].by_key()['color'][
                        i % len(plt.rcParams['axes.prop_cycle'].by_key()['color'])]
                else:
                    c = colors[i] if isinstance(colors, list) and i < len(colors) else colors
                
                # Plot with specified styling
                line = ax.plot(x, df[var],
                               label=var,
                               linewidth=lw,
                               linestyle=ls,
                               marker=marker,
                               color=c,
                               alpha=alpha)
                
                # Add daily averages if requested
                if daily_average:
                    # Calculate daily averages
                    df_copy = df.copy()
                    df_copy['date'] = df_copy.index.date
                    daily_avg = df_copy.groupby('date')[var].mean()
                    
                    # Create midday timestamps for each day
                    midday_times = [pd.Timestamp(d).replace(hour=12, minute=0, second=0)
                                    for d in daily_avg.index]
                    
                    # Darken the color for daily averages
                    if c is not None:
                        darker_c = mcolors.to_rgb(c)
                        # Make color ~30% darker
                        darker_c = tuple([max(0, x * 0.7) for x in darker_c])
                    else:
                        # Get the current color and darken it
                        darker_c = mcolors.to_rgb(line[0].get_color())
                        darker_c = tuple([max(0, x * 0.7) for x in darker_c])
                    
                    # Plot daily averages with thicker line and darker color
                    ax.plot(midday_times, daily_avg.values,
                            label=f"{var} (daily avg)",
                            linewidth=lw * 1.5,
                            alpha=1.0,
                            color=darker_c)
        
        # Format date ticks
        fig.autofmt_xdate()
        
        # Apply other customization
        ax.set_xlabel('Date')
        ax.set_ylabel('GNSS-VOD')
        ax.legend(loc=legend_loc, fontsize='x-small', handletextpad=0.5, labelspacing=0.3)
        
        if ylim:
            ax.set_ylim(ylim)
        
        if grid:
            ax.grid(True, linestyle='--', alpha=0.7)
        
        if title:
            ax.set_title(title)
        
        plt.tight_layout()
        
        # Generate filename if not provided
        if not filename:
            filename = f"vod_timeseries_{'_'.join(variables)}.png"
        
        plt.savefig(FIG / filename, dpi=300, bbox_inches='tight')
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


def plot_vod_scatter(df, x_var=None, y_var=None, polarization='compare', algo='tps',
                     hue='hour', point_size=2, add_linear_fit=False,
                     cmap='viridis', figsize=(6, 6), title=None, filename=None, **kwargs):
    """
    Create scatter plots of VOD data with flexible axis variable selection.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing VOD data
    x_var : str, optional
        Explicit variable name for x-axis. If provided, overrides polarization and algo settings.
    y_var : str, optional
        Explicit variable name for y-axis. If provided, overrides polarization and algo settings.
    polarization : str, default='compare'
        One of:
        - 'compare': Compare VOD1 vs VOD2 (x=VOD1, y=VOD2)
        - 'VOD1': Use VOD1 for both axes, but comparing algorithms
        - 'VOD2': Use VOD2 for both axes, but comparing algorithms
    algo : str, default='tps'
        One of:
        - 'tps': Use tps algorithm for both axes (when comparing polarizations)
        - 'tp': Use tp algorithm for both axes (when comparing polarizations)
        - 'compare': Compare tp vs tps algorithms (when using the same polarization)
    hue : str, default='hour'
        Column to use for point colors (usually 'hour' or 'doy')
    point_size : float, default=2
        Size of scatter points
    add_linear_fit : bool, default=False
        Whether to add a linear regression line
    cmap : str, default='viridis'
        Colormap for the scatter points
    figsize : tuple, default=(6, 6)
        Figure size (width, height) in inches
    title : str, optional
        Plot title (auto-generated if None)
    filename : str, optional
        Filename to save the plot (derived from variables if None)

    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats
    
    only_outliers = kwargs.get('only_outliers', False)
    add_1_to_1_line = kwargs.get('add_1_to_1_line', True)
    
    if only_outliers:
        # Filter out non-outliers based on outside the 90% quantile range
        # assure that only_outliers is a float or int
        if not isinstance(only_outliers, (float, int)):
            raise ValueError("only_outliers must be a float or int representing the max/min quantile. Given in percentage.")
        # Calculate bounds for both variables first
        bounds = {}
        for var in [x_var, y_var]:
            if var is not None and var in df.columns:
                bounds[var] = {
                    'lower': df[var].quantile((100 - only_outliers) / 100),
                    'upper': df[var].quantile(only_outliers / 100)
                }
        
        # Apply filtering once using combined condition
        filter_condition = True
        for var, var_bounds in bounds.items():
            filter_condition |= (df[var] < var_bounds['lower']) | (df[var] > var_bounds['upper'])
        
        extraquantile_range = (100 - only_outliers) * 2
        if len(bounds) > 0:
            df = df[filter_condition]
    
    # Determine x and y variables based on parameters if not explicitly provided
    if x_var is None or y_var is None:
        if polarization == 'compare':
            # Compare VOD1 vs VOD2 using the same algorithm
            suffix = f"_anom_{algo}" if algo in ['tp', 'tps'] else ""
            x_var = f"VOD1{suffix}"
            y_var = f"VOD2{suffix}"
            comparison_type = "polarization"
        elif polarization in ['VOD1', 'VOD2']:
            # Compare algorithms using the same polarization
            if algo == 'compare':
                x_var = f"{polarization}_anom_tp"
                y_var = f"{polarization}_anom_tps"
                comparison_type = "algorithm"
            else:
                # If a specific algorithm is given but we're not comparing,
                # default to comparing polarizations
                x_var = f"VOD1_anom_{algo}"
                y_var = f"VOD2_anom_{algo}"
                comparison_type = "polarization"
        else:
            raise ValueError("polarization must be 'compare', 'VOD1', or 'VOD2'")
    else:
        # Variables were explicitly provided
        comparison_type = "custom"
    
    # Check if variables exist in the dataframe
    if x_var not in df.columns:
        raise ValueError(f"Variable {x_var} not found in the dataframe")
    if y_var not in df.columns:
        raise ValueError(f"Variable {y_var} not found in the dataframe")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get the hue values
    if hue in df.columns:
        hue_values = df[hue]
        scatter = ax.scatter(df[x_var], df[y_var], c=hue_values, s=point_size, cmap=cmap, alpha=0.7)
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        if hue == 'hour':
            cbar.set_label('Hour of Day')
            # Set colorbar ticks to whole hours
            cbar.set_ticks(np.arange(0, 24, 3))
        elif hue == 'doy':
            cbar.set_label('Day of Year')
        else:
            cbar.set_label(hue.capitalize())
    else:
        ax.scatter(df[x_var], df[y_var], s=point_size, alpha=0.7)
    
    # Add linear fit
    if add_linear_fit:
        # Remove NaN values for regression
        mask = ~(np.isnan(df[x_var]) | np.isnan(df[y_var]))
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            df[x_var][mask], df[y_var][mask]
        )
        
        # Calculate regression line
        x_range = np.linspace(df[x_var].min(), df[x_var].max(), 100)
        y_fit = slope * x_range + intercept
        
        # Plot regression line
        ax.plot(x_range, y_fit, 'r-', linewidth=1.5)
        
        # Add equation and R² to plot
        equation = f"y = {slope:.3f}x + {intercept:.3f}"
        r_squared = f"R² = {r_value ** 2:.3f}"
        ax.annotate(f"{equation}\n{r_squared}",
                    xy=(0.05, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                    verticalalignment='top')
    
    # 1:1 line
    if add_1_to_1_line:
        min_val = min(df[x_var].min(), df[y_var].min())
        max_val = max(df[x_var].max(), df[y_var].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
    # Set equal aspect ratio
    ax.set_aspect('equal')
    
    # Labels and title
    ax.set_xlabel(x_var)
    ax.set_ylabel(y_var)
    
    if title is None:
        if comparison_type == "polarization":
            title = f"VOD1 vs VOD2 ({algo.upper()} algorithm)"
        elif comparison_type == "algorithm":
            title = f"{polarization} TP vs TPS algorithm comparison"
        else:
            title = f"{x_var} vs {y_var}"
            
    # add \n extraquantile_range to the tile if only_outliers is set
    if only_outliers:
        title += f"\nOnly Outliers ({extraquantile_range}%)"
    
    ax.set_title(title)
    
    # Grid
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure if filename provided
    if filename is None:
        filename = f"scatter_{x_var}_vs_{y_var}.png"
    
    plt.savefig(FIG / filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

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


def prepare_wavelet_analysis(df, variable, datetime_col='datetime'):
    """
    Prepare time series data for wavelet analysis by resampling to regular intervals
    and preprocessing the signal.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the time series data
    variable : str
        Name of the variable to analyze
    datetime_col : str, optional
        Name of the datetime column

    Returns:
    --------
    tuple: (time, signal, scales)
        Arrays needed for wavelet analysis
    """
    import numpy as np
    import pandas as pd
    import scipy.signal as sig
    
    # just keep var
    if variable not in df.columns:
        raise ValueError(f"Variable '{variable}' not found in DataFrame")
    
    # reset index if datetime_col is not index
    if datetime_col not in df.columns:
        # reset index to make datetime_col a column
        df = df.reset_index()
    
    # Ensure datetime is properly formatted
    if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        
    df = df.copy()
    # keep only datetime and var
    df = df[[datetime_col, variable]].dropna(subset=[datetime_col, variable])
    
    # Sort by datetime and create regular time series
    df_sorted = df.sort_values(by=datetime_col).copy()
    df_indexed = df_sorted.set_index(datetime_col)
    
    # Determine appropriate resampling frequency
    time_deltas = np.diff(df_indexed.index.astype(np.int64))
    median_minutes = np.median(time_deltas) / (1000000000 * 60)
    
    # Select resampling frequency based on data granularity
    if median_minutes < 15:
        freq = '5min'
    elif median_minutes < 60:
        freq = '15min'
    elif median_minutes < 24 * 60:
        freq = 'H'
    else:
        freq = 'D'
    
    # Resample and interpolate missing values
    df_resampled = df_indexed.resample(freq).mean().interpolate(method='linear')
    
    # Extract and preprocess signal
    signal_raw = df_resampled[variable].values
    detrended = sig.detrend(signal_raw)
    normalized = (detrended - np.mean(detrended)) / np.std(detrended)
    
    # Calculate time in decimal years for better visualization
    time_raw = df_resampled.index.to_numpy().astype('datetime64[s]')
    years = time_raw.astype('datetime64[Y]').astype(int) + 1970
    days_in_year = np.array([366 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 365 for y in years])
    day_of_year = (time_raw - time_raw.astype('datetime64[Y]')).astype('timedelta64[D]').astype(int) + 1
    time_of_day = (time_raw - time_raw.astype('datetime64[D]')).astype('timedelta64[s]').astype(int) / 86400
    time_decimal = years + (day_of_year + time_of_day) / days_in_year
    
    # Create scales for wavelet transform
    dt = (time_decimal[1] - time_decimal[0])
    min_period = 2 * dt  # Nyquist frequency
    max_period = 0.5 * (time_decimal[-1] - time_decimal[0])
    
    min_scale = min_period / dt
    max_scale = max_period / dt
    
    # Create logarithmic scale points
    num_scales = 50
    scales = np.logspace(np.log10(min_scale), np.log10(max_scale), num_scales)
    
    return time_decimal, normalized, scales


def analyze_wavelets(df, var, save_dir=None, **kwargs):
    """
    Perform wavelet analysis on tau_r and tau_g time series and their ratio.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing tau_r, tau_g, and datetime columns
    save_dir : Path or str, optional
        Directory to save the plots, defaults to FIGDIR
    """

    from pathlib import Path
    
    if save_dir is None:
        save_dir = FIG
    else:
        save_dir = Path(save_dir)
        
    smoothing = kwargs.get('smoothing', True)
    window_length = kwargs.get('window_length', 15)
    polyorder = kwargs.get('polyorder', 3)

    # Process each variable and store their wavelet data
    print(f"Processing {var} for wavelet analysis...")
    
    # check if var in df
    if var not in df.columns:
        raise ValueError(f"Variable '{var}' not found in DataFrame")
    
    # -----------------------------------
    # 1) smoothing
    
    if smoothing:
        from scipy.signal import savgol_filter
        # Apply Savitzky-Golay filter for smoothing
        print(f"Smoothing {var} with Savitzky-Golay filter (window_length={window_length}, polyorder={polyorder})...")
        df[var] = savgol_filter(df[var], window_length=window_length, polyorder=polyorder, mode='nearest')
    else:
        print(f"Skipping smoothing for {var}. Using raw data for wavelet analysis.")
    
    # -----------------------------------
    # 2) prepare time series for wavelet analysis
    
    time, signal, scales = prepare_wavelet_analysis(df, var)

    # Define output filename
    figname = save_dir / f"wavelet_{var}.png"
    
    # -----------------------------------
    # 3) plot wavelet analysis
    
    print(f"Creating wavelet plot for {var}...")
    # Create wavelet plot
    plot_wavelet(
        time=time,
        signal=signal,
        scales=scales,
        title=f"Wavelet Transform (Power Spectrum) of {var}",
        ylabel='Period (days)',
        figname=figname
    )
    print(f"Wavelet analysis for {var} saved to {figname}")


def plot_wavelet(time, signal, scales, waveletname='cmor1.5-1.0', cmap=plt.cm.seismic,
                 title='Wavelet Transform (Power Spectrum) of signal',
                 ylabel='Period (hours)', xlabel='Time', figname=None, plt=None):
    import matplotlib.dates as mdates
    import datetime
    import numpy as np
    import matplotlib.pyplot as plt
    import pywt
    
    # Convert decimal years to datetime objects
    datetimes = []
    for t in time:
        year = int(t)
        fraction = t - year
        # Calculate days in the year
        days_in_year = 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365
        # Calculate day of year from fraction
        day_of_year = int(fraction * days_in_year)
        
        # Create datetime for the beginning of that day
        dt = datetime.datetime(year, 1, 1) + datetime.timedelta(days=day_of_year)
        
        # Add time of day (fractional part of the day)
        day_fraction = fraction * days_in_year - day_of_year
        hours = int(day_fraction * 24)
        minutes = int((day_fraction * 24 - hours) * 60)
        seconds = int(((day_fraction * 24 - hours) * 60 - minutes) * 60)
        dt = dt.replace(hour=hours, minute=minutes, second=seconds)
        
        datetimes.append(dt)
    
    dt = time[1] - time[0]
    [coefficients, frequencies] = pywt.cwt(signal, scales, waveletname, dt)
    power = (abs(coefficients)) ** 2
    period = (1. / frequencies) * 365.5  # Convert to days
    
    # Set up levels for contour plot
    scale0 = 1 / 100
    numlevels = 20
    levels = [scale0]
    for ll in range(1, numlevels):
        scale0 *= 2
        levels.append(scale0)
    contourlevels = np.log2(levels)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot contours with datetime x-axis
    im = ax.contourf(mdates.date2num(datetimes), np.log2(period), np.log2(power),
                     contourlevels, extend='both', cmap=cmap)
    
    # Format x-axis to show months
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    
    # Get years in data
    years = sorted(list(set([d.year for d in datetimes])))
    
    # If data spans multiple years, add year labels at bottom
    if len(years) > 1:
        # Create a second x-axis for year labels
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        
        # Set year ticks at the middle of each year
        year_ticks = []
        year_labels = []
        
        for year in years:
            # Find all dates in this year
            year_dates = [d for d in datetimes if d.year == year]
            if year_dates:
                # Use middle of the available data for this year
                mid_idx = len(year_dates) // 2
                year_ticks.append(year_dates[mid_idx])
                year_labels.append(str(year))
        
        # Set up the year axis
        ax2.set_xticks(mdates.date2num(year_ticks))
        ax2.set_xticklabels(year_labels)
        ax2.xaxis.set_ticks_position('bottom')
        ax2.xaxis.set_label_position('bottom')
        ax2.spines['bottom'].set_visible(False)
        ax2.tick_params(axis='x', which='both', bottom=False, top=False)
        ax2.xaxis.set_tick_params(pad=25)  # Position year labels below month labels
    
    # Set up the rest of the plot
    ax.set_title(title, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=18)
    
    # Format y-axis (log scale periods)
    yticks = 2 ** np.arange(np.ceil(np.log2(period.min())),
                            np.ceil(np.log2(period.max())))
    ax.set_yticks(np.log2(yticks))
    ax.set_yticklabels(yticks)
    ax.invert_yaxis()
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.95, 0.15, 0.03, 0.7])
    fig.colorbar(im, cax=cbar_ax, orientation="vertical")
    
    # Save figure
    if not figname:
        plt.savefig('wavelet_{}.png'.format(waveletname),
                    dpi=300, bbox_inches='tight')
    else:
        plt.savefig(figname, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_vod_by_author(df, author, save_dir=None):
    """
    Create VOD plots according to specific author specifications.
    
    1. Yitong:
        - interval: 05-2022 to 11-2023
        - VOD1_anom
        - figsize=(5, 3)
        - daily average (red)
        - 95% percentile shaded area (light red)
        - linewidth 1.2
        - ylims=(0.4, 0.9)
    2. Humphrey:
        - interval: 05-2023 to 12-2023
        - hourly data
        - figsize=(10, 5)
        - two curves: "raw" VOD1 (grey) and "processed" VOD1_anom (black), line width 0.8
        - ylims=(0.6,1)
    3. Burns
        - interval: 2023, doy 210-310
        - hourly data
        - VOD1_anom
        - figsize=(5, 3)
        - ylims=(0, 1)
        
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing VOD data
    author : str
        Author name ('yitong', 'humphrey', or 'burns')
    save_dir : Path or str, optional
        Directory to save the plot, defaults to FIG

    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from pathlib import Path
    
    if save_dir is None:
        save_dir = FIG
    else:
        save_dir = Path(save_dir)
    
    # Convert author to lowercase for case-insensitive matching
    author = author.lower()
    
    # Make a copy of the data to avoid modifying the original
    df = df.copy()
    
    if author == 'yitong':
        # Yitong's specifications
        # Interval: 05-2022 to 11-2023
        start_date = '2022-01-01'
        end_date = '2023-11-30'
        
        # Filter by date range
        mask = (df.index >= start_date) & (df.index <= end_date)
        data = df[mask].copy()
        
        # Calculate daily average and 95% percentile
        qq = 95/100
        data['date'] = data.index.date
        daily_avg = data.groupby('date')['VOD1_anom'].mean()
        daily_p025 = data.groupby('date')['VOD1_anom'].quantile(1-qq)
        daily_p975 = data.groupby('date')['VOD1_anom'].quantile(qq)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(4.3, 2))
        
        # Convert date to datetime for proper plotting
        dates = [pd.Timestamp(d) for d in daily_avg.index]
        
        # Plot 95% confidence interval
        ax.fill_between(dates, daily_p025, daily_p975, color='red', alpha=0.2)
        
        # Plot daily average
        ax.plot(dates, daily_avg, color='red', linewidth=0.5)
        
        # Set y-limits
        ax.set_ylim(0.4, 0.9)
        
        # Labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('VOD1_anom')
        ax.set_title("VOD1 Anomaly (Yitong style)")
        
        # Format x-axis
        from matplotlib.dates import MonthLocator, YearLocator, DateFormatter
        ax.xaxis.set_major_locator(MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
        # rotate x-tick labels
        ax.tick_params(axis='x', rotation=30)
        
        # make minor grid
        ax.grid(which='minor', linestyle=':', linewidth=0.5, color='grey')
        fig.autofmt_xdate()
    
    elif author == 'humphrey':
        # Humphrey's specifications
        # Interval: 05-2023 to 12-2023
        start_date = '2023-05-01'
        end_date = '2023-12-31'
        
        # Filter by date range
        mask = (df.index >= start_date) & (df.index <= end_date)
        data = df[mask].copy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(7, 2))
        
        # Plot "raw" VOD1 in grey and "processed" VOD1_anom in black
        ax.plot(data.index, data['VOD1'], color='grey', linewidth=0.4, label='Raw VOD1')
        ax.plot(data.index, data['VOD1_anom'], color='black', linewidth=0.4, label='Processed VOD1_anom')
        
        # Set y-limits
        ax.set_ylim(0.3, 1.0)
        
        # Labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('VOD')
        ax.set_title("VOD1 Raw and Processed (Humphrey style)")
        
        # format x-axis to show dates
        from matplotlib.dates import DayLocator, DateFormatter, MonthLocator
        ax.xaxis.set_major_locator(MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(DateFormatter('%b'))
        
        # ax.legend(loc='upper left')
    
    elif author == 'burns':
        # Burns' specifications
        # 2023, doy 210-310
        year = 2023
        start_doy = 210
        end_doy = 310
        
        # Create a new column for day of year if it doesn't exist
        if 'doy' not in df.columns:
            df['doy'] = df.index.dayofyear
        
        # Filter by year and day of year
        mask = (df.index.year == year) & (df['doy'] >= start_doy) & (df['doy'] <= end_doy)
        data = df[mask].copy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(6, 2))
        
        # Plot hourly VOD1_anom
        ax.plot(data.index, data['VOD1_anom'], linewidth=0.5, color='black')
        
        # Set y-limits
        ax.set_ylim(0, 1)
        
        # Labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('VOD1_anom')
        ax.set_title(f"VOD1 Anomaly, DOY {start_doy}-{end_doy} (Burns style)")
        
        # Format x-axis to show dates
        from matplotlib.dates import DayLocator, DateFormatter
        ax.xaxis.set_major_locator(DayLocator(interval=14))
        ax.xaxis.set_major_formatter(DateFormatter('%j'))
    
    else:
        raise ValueError(f"Unknown author: {author}. Choose from 'yitong', 'humphrey', or 'burns'.")
    
    # Save figure
    plt.tight_layout()
    plt.savefig(save_dir / f"vod_{author}_style.png", dpi=300)
    plt.show()
    
    return fig