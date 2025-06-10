#!/usr/bin/env python
# -*- coding: utf-8 -*-

def plot_sv_observation_counts(vod_ts_svs, min_threshold=10, figsize=(10, 8), save_path=None):
    """
    Plot histograms of observation counts by constellation.

    Parameters
    ----------
    vod_ts_svs : pandas.DataFrame
        DataFrame with SV data that includes 'n' column with observation counts
    min_threshold : int, default=10
        Minimum observation threshold to mark with vertical line
    figsize : tuple, default=(10, 8)
        Figure size (width, height) in inches
    save_path : str, optional
        Path to save the figure. If None, figure is displayed but not saved

    Returns
    -------
    None
    """
    import seaborn as sns
    from matplotlib import pyplot as plt
    
    # Extract observation counts per SV and cell
    num_obs = vod_ts_svs.groupby(['SV', 'CellID'])['n'].median().to_frame(name='n')
    # Exclude SBAS satellites (those starting with 'S')
    num_obs = num_obs[~num_obs.index.get_level_values('SV').str.startswith('S')]
    
    # Set aesthetic style
    sns.set_style("whitegrid")
    
    # Create a 2x2 grid of histograms
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    axs = axs.flatten()  # Flatten to easily iterate
    
    # Reset index to get SV as a column for grouping
    sv_data = num_obs.reset_index()
    
    # please cut off SVs with less than 0 and more than 200 observations
    sv_data = sv_data[(sv_data['n'] >= 0) & (sv_data['n'] <= 200)]
    
    # Add constellation information based on first letter of SV
    constellation_map = {
        'G': 'GPS',
        'R': 'GLONASS',
        'E': 'Galileo',
        'C': 'BeiDou',
        'S': 'SBAS'
    }
    sv_data['Constellation'] = sv_data['SV'].str[0].map(constellation_map)
    
    # Create color palette for constellations
    constellation_colors = {
        'GPS': '#1f77b4',  # blue
        'GLONASS': '#ff7f0e',  # orange
        'Galileo': '#2ca02c',  # green
        'BeiDou': '#d62728',  # red
        'SBAS': '#9467bd'  # purple
    }
    
    # Main constellations to plot individually
    main_constellations = ['GPS', 'GLONASS', 'Galileo', 'BeiDou']
    
    # Plot each constellation in its own subplot
    for i, constellation in enumerate(main_constellations):
        # Filter data for this constellation
        constellation_data = sv_data[sv_data['Constellation'] == constellation]
        
        # Create histogram
        sns.histplot(
            data=constellation_data,
            x='n',
            bins=50,
            color=constellation_colors[constellation],
            ax=axs[i]
        )
        
        # Add vertical line at minimum threshold
        axs[i].axvline(
            x=min_threshold,
            color='black',
            linestyle='--',
            linewidth=2,
            label=f'Minimum threshold ({min_threshold} obs)'
        )
        
        # Set title and labels
        axs[i].set_title(f"{constellation}", fontsize=12)
        axs[i].set_xlabel('Number of Observations per SV', fontsize=10)
        axs[i].set_ylabel('Count', fontsize=10)
        axs[i].set_xlim(0, 200)
        
        # Add legend only once
        if i == 0:
            axs[i].legend()
    
    # Add overall title
    plt.suptitle("Histograms of Observations by Constellation", fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Make room for the suptitle
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()