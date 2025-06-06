#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd

from gnssvod.io.vodreader import VODReader
from processing.inspect_vod_funs import plot_vod_scatter

# | Metric name                                         | Symbol            | Mathematical representation          | Rationale                                           |
# |-----------------------------------------------------|-------------------|:-------------------------------------|-----------------------------------------------------|
# | Number of sats in view                              | $N_s(t)$          |                                      | Gaps in the overall coverage                        |
# | Standard deviation of the number of sats in view    | $\sigma_{N_s(t)}$ |                                      | Variability of observations with in a time interval |
# | Fraction of sky currently observed (cutoff applied) | $C(t)$            | $C(t) = \frac{C_t * 100}{C_{total}}$ | Probably correlated to  $N_s(t)$                    |
# | Binned fraction of sky observed (cutoff applied)    | $C_b(t)$          | $C_i(t) = \frac{C_{t,i}*100}{C_t}$   | Variability in biomass areas observed               |


# Dictionary mapping variable names in dataframe to LaTeX symbols
metrics_rename_dict = {
    'Ns_t': 'N_s(t)',
    'SD_Ns_t': '\\sigma_{N_s(t)}',
    'C_t_perc': 'C(t)'
}

# For bin-specific variables (dynamic pattern)
# Variables like 'Ci_t_VOD1_bin0_pct', 'Ci_t_VOD1_bin1_pct', etc.
# will be renamed to 'C_0(t)', 'C_1(t)', etc.

def plot_vod_vs_metrics(df, vod_var, metric_vars=None, hue='doy', figsize=(12, 10),
                        cmap='viridis', point_size=10, alpha=0.7,
                        add_linear_fit=True, add_correlation=True, add_loess=False,
                        title=None, filename=None, **kwargs):
    """
    Create a 2x2 grid of scatter plots showing VOD as function of other variables.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing VOD and metric variables
    vod_var : str
        Name of the VOD variable to plot on y-axis
    metric_vars : list, optional
        List of metric variable names to plot on x-axis. If None, will use the first
        4 suitable numeric variables found in the dataframe.
    hue : str, default='doy'
        Column to use for point colors
    figsize : tuple, default=(12, 10)
        Figure size (width, height) in inches
    cmap : str, default='viridis'
        Colormap for the scatter points
    point_size : float, default=10
        Size of scatter points
    alpha : float, default=0.7
        Transparency of scatter points
    add_linear_fit : bool, default=True
        Whether to add a linear regression line
    add_correlation : bool, default=True
        Whether to add correlation metrics (r, r², p-value)
    add_loess : bool, default=False
        Whether to add a LOESS smoothing curve
    title : str, optional
        Overall plot title
    filename : str, optional
        If provided, saves the plot to the specified file
    **kwargs : dict
        Additional customization options:
        - ylim : tuple, default=None
            Y-axis limits (min, max) for all subplots
        - loess_frac : float, default=0.6
            Fraction of points to use for LOESS smoothing
        - subplot_titles : list, default=None
            Custom titles for each subplot
        - annotate_pos : tuple, default=(0.05, 0.95)
            Position of the correlation annotation (x, y in axis fraction)
        - grid : bool, default=True
            Whether to show grid lines
        - legend : bool, default=True
            Whether to show the colorbar legend
        - shared_ylim : bool, default=True
            Whether to use the same y-axis limits for all subplots
        - filter_outliers : float, default=None
            Percentile (0-100) to filter outliers

    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats
    import pandas as pd
    from definitions import FIG
    from pathlib import Path
    
    # Extract kwargs
    ylim = kwargs.get('ylim', None)
    loess_frac = kwargs.get('loess_frac', 0.6)
    subplot_titles = kwargs.get('subplot_titles', None)
    annotate_pos = kwargs.get('annotate_pos', (0.05, 0.95))
    grid = kwargs.get('grid', True)
    show_legend = kwargs.get('legend', True)
    shared_ylim = kwargs.get('shared_ylim', True)
    filter_outliers = kwargs.get('filter_outliers', None)
    
    # Check if VOD variable exists
    if vod_var not in df.columns:
        raise ValueError(f"VOD variable '{vod_var}' not found in DataFrame")
    
    # If no metric variables specified, find numeric columns
    if metric_vars is None:
        numeric_cols = df.select_dtypes(include=np.number).columns
        # Exclude the VOD variable and hue variable
        exclude_cols = [vod_var]
        if hue in numeric_cols:
            exclude_cols.append(hue)
        
        metric_vars = [col for col in numeric_cols if col not in exclude_cols]
        # Take up to 4 variables
        metric_vars = metric_vars[:4]
        
        if not metric_vars:
            raise ValueError("No suitable numeric variables found for x-axis")
    
    # Make sure we have at most 4 variables
    if len(metric_vars) > 4:
        print(f"Warning: Only the first 4 variables will be used: {metric_vars[:4]}")
        metric_vars = metric_vars[:4]
    
    # Create figure and subplots
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    axs = axs.flatten()
    
    # If hue is not in the dataframe, use a default color
    if hue not in df.columns:
        print(f"Warning: Hue variable '{hue}' not found. Using default coloring.")
        hue_values = None
        scatter_kwargs = {'c': 'blue'}
    else:
        hue_values = df[hue]
        scatter_kwargs = {'c': hue_values, 'cmap': cmap}
    
    # Determine global y-limits if shared_ylim is True
    if shared_ylim:
        ymin, ymax = np.inf, -np.inf
        for var in metric_vars:
            if var in df.columns:
                mask = ~(np.isnan(df[vod_var]) | np.isnan(df[var]))
                if mask.any():
                    ymin = min(ymin, df.loc[mask, vod_var].min())
                    ymax = max(ymax, df.loc[mask, vod_var].max())
        
        # Add some padding
        yrange = ymax - ymin
        ymin -= yrange * 0.05
        ymax += yrange * 0.05
        
        # Override with user-provided ylim if available
        if ylim is not None:
            ymin, ymax = ylim
    
    # Function to add LOESS curve
    def add_loess_curve(ax, x, y, frac=loess_frac):
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            # Sort data for smooth curve
            sorted_data = pd.DataFrame({'x': x, 'y': y}).sort_values('x')
            # Apply LOWESS smoothing
            z = lowess(sorted_data['y'], sorted_data['x'], frac=frac)
            # Plot the smoothed curve
            ax.plot(z[:, 0], z[:, 1], 'r-', linewidth=2, alpha=0.8)
        except ImportError:
            print("Warning: statsmodels not available. LOESS curve not added.")
            pass
    
    # Plot each variable
    for i, var in enumerate(metric_vars):
        if i >= len(axs):
            print(f"Warning: Not enough subplots for variable '{var}'. Skipping.")
            continue
        
        ax = axs[i]
        
        if var not in df.columns:
            ax.text(0.5, 0.5, f"Variable '{var}'\nnot found",
                    ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Get data, removing NaN values
        mask = ~(np.isnan(df[vod_var]) | np.isnan(df[var]))
        if not mask.any():
            ax.text(0.5, 0.5, f"No valid data\nfor '{var}'",
                    ha='center', va='center', transform=ax.transAxes)
            continue
        
        x = df.loc[mask, var]
        y = df.loc[mask, vod_var]
        
        # Filter outliers if requested
        if filter_outliers is not None:
            lower = (100 - filter_outliers) / 2
            upper = 100 - lower
            x_min, x_max = np.percentile(x, [lower, upper])
            y_min, y_max = np.percentile(y, [lower, upper])
            outlier_mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
            x = x[outlier_mask]
            y = y[outlier_mask]
            if hue_values is not None:
                scatter_kwargs['c'] = hue_values[mask][outlier_mask]
        
        # Create scatter plot
        if hue_values is not None:
            scatter = ax.scatter(x, y, s=point_size, alpha=alpha, **scatter_kwargs)
        else:
            ax.scatter(x, y, s=point_size, alpha=alpha, **scatter_kwargs)
        
        # Add linear fit
        if add_linear_fit:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, 'k--', linewidth=1.5, alpha=0.7)
            
            # Add correlation metrics
            if add_correlation:
                r_squared = r_value ** 2
                annotation = f"r = {r_value:.3f}\nr² = {r_squared:.3f}\np = {p_value:.3e}"
                ax.annotate(annotation, xy=annotate_pos, xycoords='axes fraction',
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                            verticalalignment='top')
        
        # Add LOESS curve
        if add_loess:
            add_loess_curve(ax, x, y, frac=loess_frac)
        
        # Set labels and title
        ax.set_xlabel(var)
        ax.set_ylabel(vod_var)
        
        if subplot_titles and i < len(subplot_titles):
            ax.set_title(subplot_titles[i])
        else:
            ax.set_title(f"{vod_var} vs {var}")
        
        # Set y-limits if shared across all plots
        if shared_ylim:
            ax.set_ylim(ymin, ymax)
        
        # Add grid
        if grid:
            ax.grid(True, linestyle='--', alpha=0.7)
    
    # Hide empty subplots
    for i in range(len(metric_vars), len(axs)):
        axs[i].set_visible(False)
    
    # Add colorbar if hue is used
    if hue_values is not None and show_legend:
        cbar = fig.colorbar(scatter, ax=axs, pad=0.5)
        cbar.set_label(hue)
        
        # Format colorbar for doy
        if hue == 'doy':
            # Set ticks to represent months
            month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            # Only include months that are within the range of the data
            min_doy, max_doy = hue_values.min(), hue_values.max()
            tick_indices = [i for i, doy in enumerate(month_starts) if min_doy <= doy <= max_doy]
            
            if tick_indices:
                ticks = [month_starts[i] for i in tick_indices]
                labels = [month_names[i] for i in tick_indices]
                cbar.set_ticks(ticks)
                cbar.set_ticklabels(labels)
    
    # Set overall title
    if title:
        fig.suptitle(title, fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    if title:
        plt.subplots_adjust(top=0.9)  # Make room for the suptitle
    
    # Save figure if filename provided
    if filename:
        if not isinstance(filename, Path):
            filename = FIG / filename
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return fig


def plot_sky_coverage_distribution(df, gnssband='VOD1', figsize=(12, 6), title=None,
                                   cmap='viridis', filename=None, **kwargs):
    """
    Create a stacked bar plot showing the daily distribution of sky coverage across VOD percentile bins.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing Ci_t_VOD*_bin*_pct columns
    gnssband : str, default='VOD1'
        GNSS band to use for analysis (VOD1 or VOD2)
    figsize : tuple, default=(12, 6)
        Figure size in inches
    title : str, optional
        Plot title
    cmap : str, default='viridis'
        Colormap for the stacked bars
    filename : str, optional
        Filename to save the plot
    **kwargs : dict
        Additional plotting parameters:
        - month_labels : bool, default=True
            Whether to show month labels on x-axis
        - legend_position : str, default='right'
            Position of legend ('right', 'top', 'bottom', 'inside')

    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from definitions import FIG
    
    # Extract kwargs
    month_labels = kwargs.get('month_labels', True)
    legend_position = kwargs.get('legend_position', 'right')
    
    # Create 'doy' column if it doesn't exist
    df = df.copy()
    if 'doy' not in df.columns:
        df['doy'] = df.index.dayofyear
    
    # Get all Ci columns for the specified band
    bin_columns = [col for col in df.columns if col.startswith(f'Ci_t_{gnssband}_bin') and col.endswith('_pct')]
    
    if not bin_columns:
        raise ValueError(f"No Ci_t_{gnssband}_bin*_pct columns found in DataFrame")
    
    # Sort bin columns by bin number
    bin_columns.sort(key=lambda x: int(x.split('bin')[1].split('_')[0]))
    
    # Group by day of year and calculate mean for each bin
    daily_bins = df.groupby('doy')[bin_columns].mean()
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get colormap
    colormap = plt.cm.get_cmap(cmap, len(bin_columns))
    colors = [colormap(i) for i in range(len(bin_columns))]
    
    # Create stacked bar chart
    bottom = np.zeros(len(daily_bins))
    
    for i, col in enumerate(bin_columns):
        # Extract bin number for label
        bin_num = int(col.split('bin')[1].split('_')[0])
        # Create label for legend
        label = f"Bin {bin_num}"
        
        # Plot this bin's data as a stacked component
        ax.bar(daily_bins.index, daily_bins[col], bottom=bottom,
               width=1.0, color=colors[i], label=label, alpha=0.8)
        
        # Update bottom for next layer
        bottom += daily_bins[col].values
    
    # Set labels and title
    ax.set_xlabel('Day of Year')
    ax.set_ylabel('Percentage of Sky Coverage (%)')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'{gnssband} Sky Coverage Distribution by Percentile Bins')
    
    # Format x-axis to show month boundaries
    if month_labels:
        # Generate month boundary DOYs
        month_boundaries = []
        month_labels = []
        
        # Get an arbitrary non-leap year for month boundaries
        year = 2023
        for month in range(1, 13):
            # Get the first day of each month
            date = pd.Timestamp(f"{year}-{month:02d}-01")
            month_boundaries.append(date.dayofyear)
            # Use abbreviated month names
            month_labels.append(date.strftime("%b"))
        
        # Set the ticks and labels
        ax.set_xticks(month_boundaries)
        ax.set_xticklabels(month_labels)
    
    # Add minor ticks every 10 days
    minor_ticks = np.arange(1, 366, 10)
    ax.set_xticks(minor_ticks, minor=True)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.grid(True, which='minor', linestyle=':', alpha=0.4)
    
    # Set y-axis to go from 0 to 100%
    ax.set_ylim(0, 100)
    
    # Add legend based on position
    if legend_position == 'right':
        ax.legend(title="VOD Percentile Bins", bbox_to_anchor=(1.05, 1), loc='upper left')
    elif legend_position == 'top':
        ax.legend(title="VOD Percentile Bins", bbox_to_anchor=(0.5, 1.05), loc='upper center', ncol=len(bin_columns))
    elif legend_position == 'bottom':
        ax.legend(title="VOD Percentile Bins", bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=len(bin_columns))
    else:  # 'inside'
        ax.legend(title="VOD Percentile Bins", loc='upper right')
    
    plt.tight_layout()
    
    # Save figure if filename provided
    if filename:
        plt.savefig(FIG / filename, dpi=300, bbox_inches='tight')
    else:
        plt.savefig(FIG / f"sky_coverage_distribution_{gnssband}.png", dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return fig


def plot_diurnal_sky_coverage_distribution(df, gnssband='VOD1', figsize=(12, 6), title=None,
                                           cmap='viridis', filename=None, **kwargs):
    """
    Create a stacked bar chart showing the diurnal distribution of sky coverage percentiles.

    Each bar represents an hour of day, with colored segments showing how the distribution
    of sky coverage across different VOD percentile bins changes throughout the day.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the VOD data with bin columns
    gnssband : str, default='VOD1'
        The GNSS band to use for bin columns
    figsize : tuple, default=(12, 6)
        Figure size as (width, height) in inches
    title : str, optional
        Plot title. If None, a default title is generated
    cmap : str, default='viridis'
        Matplotlib colormap to use for bin colors
    filename : str, optional
        If provided, save the figure to this filename
    **kwargs : dict
        Additional keyword arguments:
        - legend_position : str, default='right'
          Position of the legend ('right', 'top', 'bottom', or 'inside')

    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from definitions import FIG
    
    # Extract kwargs
    legend_position = kwargs.get('legend_position', 'right')
    
    # Create 'hod' column if it doesn't exist
    df = df.copy()
    if 'hod' not in df.columns:
        df['hod'] = df.index.hour
    
    # Get all Ci columns for the specified band
    bin_columns = [col for col in df.columns if col.startswith(f'Ci_t_{gnssband}_bin') and col.endswith('_pct')]
    
    if not bin_columns:
        raise ValueError(f"No bin columns found for band {gnssband}. Available columns: {df.columns}")
    
    # Sort bin columns by bin number
    bin_columns.sort(key=lambda x: int(x.split('bin')[1].split('_')[0]))
    
    # Group by hour of day and calculate mean for each bin
    hourly_bins = df.groupby('hod')[bin_columns].mean()
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get colormap
    colormap = plt.cm.get_cmap(cmap, len(bin_columns))
    colors = [colormap(i) for i in range(len(bin_columns))]
    
    # Create stacked bar chart
    bottom = np.zeros(len(hourly_bins))
    
    for i, col in enumerate(bin_columns):
        values = hourly_bins[col].values
        bin_number = int(col.split('bin')[1].split('_')[0])
        bin_label = f"Bin {bin_number}"
        ax.bar(hourly_bins.index, values, bottom=bottom, color=colors[i], label=bin_label)
        bottom += values
    
    # Set labels and title
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Percentage of Sky Coverage (%)')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Diurnal Distribution of {gnssband} Sky Coverage Percentiles')
    
    # Set x-axis ticks for every hour
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(24)])
    plt.xticks(rotation=45)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Set y-axis to go from 0 to 100%
    ax.set_ylim(0, 100)
    
    # Add legend based on position
    if legend_position == 'right':
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    elif legend_position == 'top':
        ax.legend(bbox_to_anchor=(0.5, 1.1), loc='upper center', ncol=len(bin_columns))
    elif legend_position == 'bottom':
        ax.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=len(bin_columns))
    else:  # 'inside'
        ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    # Save figure if filename provided
    if filename:
        plt.savefig(FIG / filename, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    single_file_settings = {
        'station': 'MOz',
        'time_interval': ('2024-01-01', '2024-11-30'),
    }
    
    reader = VODReader(single_file_settings)
    vod = reader.get_data(format='long')
    
    # -----------------------------------
    # THIS SHOULD BE DEPRECATED SOON
    # Fill NaN values across algorithms for the same timestamp
    vod_filled = vod.groupby('datetime').apply(lambda x: x.ffill().bfill()).reset_index(drop=True)
    
    vod_tp = vod_filled[vod_filled['algo'] == 'tp']
    vod_tps = vod_filled[vod_filled['algo'] == 'tps']
    
    # -----------------------------------
    # Now apply the renaming dictionary
    print(vod_tp.columns.tolist())
    
    scatter = False
    # Example: Plot VOD1 vs satellite metrics
    if scatter:
        plot_vod_vs_metrics(
            vod_tps,
            vod_var='VOD1_anom',
            metric_vars=['Ns_t', 'SD_Ns_t', 'C_t_perc', 'Ci_t_VOD1_bin0_pct'],
            hue='doy',
            add_linear_fit=True,
            title='VOD1 vs Satellite Metrics',
            filename='vod_metrics_correlation.png',
            filter_outliers=95,  # Remove extreme 5% of points
            figsize=(7,7),
        )
        
        
    # -----------------------------------
    # fingerprint plots
    
    finger = True
    if finger:
        from processing.inspect_vod_funs import plot_vod_fingerprint
        # Plot fingerprint for VOD1
        plot_vod_fingerprint(vod_tps, 'VOD1_anom', title="VOD1 Fingerprint")
        plot_vod_fingerprint(vod_tps, 'Ns_t', title="VOD1 Fingerprint")
        plot_vod_fingerprint(vod_tps, 'SD_Ns_t', title="VOD1 Fingerprint")
        plot_vod_fingerprint(vod_tps, 'C_t_perc', title="VOD1 Fingerprint")


    # -----------------------------------
    # sky coverage
    # Basic usage
    
    sky_coverage = True
    
    if sky_coverage:
        vod_data = vod_tps
        
        # With custom options
        plot_sky_coverage_distribution(
            vod_data,
            gnssband='VOD1',
            title='VOD1 Sky Coverage Distribution (2024)',
            cmap='plasma',
            filename='vod1_sky_coverage_2023.png',
            figsize=(7, 4),
            legend_position='top'
        )
        
        # With custom options
        plot_diurnal_sky_coverage_distribution(
            vod_tps,
            gnssband='VOD1',
            cmap='plasma',
            filename='vod1_diurnal_sky_coverage.png',
            figsize=(7, 4),
            legend_position='top'
        )