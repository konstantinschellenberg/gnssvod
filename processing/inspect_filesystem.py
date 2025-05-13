#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import glob
from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt

from datetime import datetime
from definitions import DATA, ROOT, GROUND, TOWER, FIG



def analyze_file_sizes(indir, format="*.bnx$"):
    """
    Crawls directory for files matching regex pattern and analyzes file sizes by time periods.

    Parameters
    ----------
    indir : str or Path
        Directory to search for files
    format : str
        Regex pattern to match files, default is "*.bnx$"

    Returns
    -------
    pandas.DataFrame
        Table with mean file sizes by year, month, and day-of-year
    """
    # Find all files matching the pattern
    all_files = []
    for root, _, _ in os.walk(indir):
        files = glob.glob(os.path.join(root, format))
        all_files.extend(files)
    
    # Extract file information and date components
    data = []
    pattern = re.compile(r'SEPT(\d{3})([a-z])\.(\d{2})\..*')
    
    for file_path in all_files:
        filename = os.path.basename(file_path)
        size = os.path.getsize(file_path)
        
        match = pattern.match(filename)
        if match:
            doy, _, yy = match.groups()
            doy = int(doy)
            year = 2000 + int(yy)  # Assuming 20xx
            
            # Convert DOY to month and day
            date = datetime.strptime(f"{year}-{doy}", "%Y-%j")
            month = date.month
            
            data.append({
                'filename': filename,
                'size': size,
                'year': year,
                'month': month,
                'doy': doy,
                'path': file_path
            })
    
    # Create dataframe
    df = pd.DataFrame(data)
    
    if len(df) == 0:
        print(f"No files found matching the pattern {format} in {indir}")
        return pd.DataFrame()
    
    # Calculate statistics
    stats = df.groupby(['year', 'month', 'doy'])['size'].agg(
        count='count',
        mean_size='mean',
        total_size='sum'
    ).reset_index()
    
    # Convert sizes to more readable format (MB)
    stats['mean_size_MB'] = stats['mean_size'] / (1024 * 1024)
    stats['total_size_MB'] = stats['total_size'] / (1024 * 1024)
    
    return stats


def plot_file_size_statistics(stats_df, title="File Size Statistics", **kwargs):
    """
    Creates a figure with three plots:
    1. Mean size per year (barplot)
    2. Mean size per month (lineplot)
    3. Mean size per day-of-year (lineplot)

    Parameters
    ----------
    stats_df : pandas.DataFrame
        DataFrame with file size statistics as returned by analyze_file_sizes()
    title : str
        Title for the overall figure

    Returns
    -------
    matplotlib.figure.Figure
        Figure with the three plots
    """
    import matplotlib.pyplot as plt
    
    if stats_df.empty:
        print("No data to plot")
        return None
    
    figsize = kwargs.get('figsize', (8,10))
    
    # Create figure with 3 subplots
    fig, axs = plt.subplots(3, 1, figsize=figsize)
    fig.suptitle(title, fontsize=16)
    
    # 1. Mean size per year (barplot)
    stats_df["year"] = stats_df["year"].astype(str)
    yearly_stats = stats_df.groupby('year')['mean_size_MB'].mean().reset_index()
    axs[0].bar(yearly_stats['year'], yearly_stats['mean_size_MB'], color='steelblue')
    axs[0].set_title('Mean File Size per Year')
    axs[0].set_xlabel('Year')
    axs[0].set_ylabel('Mean File Size (MB)')
    axs[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # 2. Mean size per month (lineplot)
    monthly_stats = stats_df.groupby(['year', 'month'])['mean_size_MB'].mean().reset_index()
    
    for year, group in monthly_stats.groupby('year'):
        axs[1].plot(group['month'], group['mean_size_MB'], marker='o',
                    label=str(year), linewidth=2)
    
    axs[1].set_title('Mean File Size per Month')
    axs[1].set_xlabel('Month')
    axs[1].set_ylabel('Mean File Size (MB)')
    axs[1].grid(linestyle='--', alpha=0.7)
    axs[1].legend(title='Year')
    axs[1].set_xticks(range(1, 13))
    
    # 3. Mean size per day-of-year (lineplot)
    daily_stats = stats_df.groupby(['year', 'doy'])['mean_size_MB'].mean().reset_index()
    
    for year, group in daily_stats.groupby('year'):
        axs[2].plot(group['doy'], group['mean_size_MB'], label=str(year), alpha=0.7)
    
    axs[2].set_title('Mean File Size per Day of Year')
    axs[2].set_xlabel('Day of Year')
    axs[2].set_ylabel('Mean File Size (MB)')
    axs[2].grid(linestyle='--', alpha=0.7)
    axs[2].legend(title='Year')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    return fig


def main():
    # Example usage
    file_format = "*.obs"
    tower_stats = analyze_file_sizes(DATA / TOWER, format=file_format)
    print("Tower file statistics:")
    print(tower_stats)
    
    if not tower_stats.empty:
        tower_fig = plot_file_size_statistics(tower_stats, title="Tower Files Size Statistics", figsize=(6, 8))
        tower_fig.savefig(str(Path(FIG, "tower_file_statistics.png")))
        plt.show()
    
    ground_stats = analyze_file_sizes(DATA / GROUND, format=file_format)
    print("\nGround file statistics:")
    print(ground_stats)
    
    if not ground_stats.empty:
        ground_fig = plot_file_size_statistics(ground_stats, title="Ground Files Size Statistics", figsize=(6, 8))
        ground_fig.savefig(str(Path(FIG, "ground_file_statistics.png")))
        plt.show()

if __name__ == '__main__':
    main()