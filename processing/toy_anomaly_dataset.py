#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from analysis.calculate_anomaly_functions import ke_fun
from analysis.calculate_anomalies import calculate_anomaly
import gnssvod as gv
from processing.export_vod_funs import plot_hemi
from processing.settings import *
from definitions import DATA, FIG

FIG = FIG / "anomaly"
FIG.mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':
    # make a toy data set of VOD data with SV and CellID and Epoch as indeces, CellID are associated with an azimuth and theta column. one vod variable
    # Create a sample DataFrame
    
    d = 20  # canopy height in meters
    angular_cutoff = 10  # degrees, for plotting purposes
    
    dates = pd.date_range(start='2022-01-01', periods=100, freq='10min')
    svs = [f'SV{i}' for i in range(1, 6)]
    
    vod_dict = {
        'Epoch': np.tile(dates, len(svs) * len(dates)),
        'SV': np.repeat(svs, len(dates) * len(dates)),
        'VOD1': np.random.normal(loc=1, scale=0.3, size=len(dates) * len(svs) * len(dates)),
        'Azimuth': np.random.rand(len(dates) * len(svs) * len(dates)) * 360,
        'Elevation': np.random.rand(len(dates) * len(svs) * len(dates)) * 90
    }
    
    vod = pd.DataFrame(vod_dict)

    # vod.plot(kind='hist', y='VOD1', bins=30, title='VOD1 Distribution'); plt.show()
    # vod.plot(kind='hist', y='Azimuth', bins=30, title='Azimuth Distribution'); plt.show()
    # vod.plot(kind='hist', y='Elevation', bins=30, title='Elevation Distribution'); plt.show()
    
    # -----------------------------------
    # Build hemispheric grid
    hemi = gv.hemibuild(1, 10)
    
    # Classify VOD into grid cells
    vod_cell = hemi.add_CellID(vod.copy())
    # vod_cell.set_index(['Epoch', 'SV'], inplace=True)
    
    # -----------------------------------
    # plot VOD1 time series for each SV
    
    # select a random SV
    sv = 1
    vod_cell_sv = vod_cell[vod_cell['SV'] == f'SV{sv}']
    vod_cell_sv.set_index('Epoch', inplace=True)
    # vod_cell_sv['VOD1'].plot(title=f'VOD1 Time Series for SV{sv}', figsize=(10, 5)); plt.show()

    # -----------------------------------
    # print the percentage of NaN values in the VOD1 column
    
    vod_cell['ke'], vod_cell['pathlength'] = ke_fun(
        vod_cell['VOD1'],
        d=20,  # Example canopy height
        elevation=vod_cell["Elevation"]  # Example elevation angle based on time
    )
    
    # viz
    vod_cell.set_index(['Epoch', 'SV'], inplace=True)
    _, avg = calculate_anomaly(vod_cell, ["ke"], 60)  # Example temporal resolution in minutes

    # -----------------------------------
    # Histograms
    avg.plot(kind="hist", y="ke_mean", bins=30, title="Histogram of ke values"); plt.show()
    avg.plot(kind="hist", y="pathlength_mean", bins=30, title="Histogram of pathlength values"); plt.show()
    avg.plot(kind="hist", y="VOD1_mean", bins=30, title="Histogram of VOD1 values"); plt.show()

    # -----------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    
    # VOD vs ke
    avg.plot(kind="scatter", x="VOD1_mean", y="ke_mean", s=3, ax=axes[0, 0])
    axes[0, 0].set_title("VOD vs Extinction Coefficient")
    axes[0, 0].grid(True, alpha=0.3)
    
    # Normalize pathlength by canopy height for better interpretation
    avg["path_per_d"] = avg["pathlength_mean"] / d
    
    # ke vs normalized pathlength
    avg.plot(kind="scatter", x="ke_mean", y="path_per_d", s=3, ax=axes[0, 1])
    axes[0, 1].set_title("Extinction Coefficient vs Normalized Pathlength")
    axes[0, 1].set_ylabel(f"Pathlength / Canopy Height")
    axes[0, 1].grid(True, alpha=0.3)
    
    # VOD vs normalized pathlength
    avg.plot(kind="scatter", x="VOD1_mean", y="path_per_d", s=3, ax=axes[1, 0])
    axes[1, 0].set_title("VOD vs Normalized Pathlength")
    axes[1, 0].set_ylabel(f"Pathlength / Canopy Height")
    axes[1, 0].grid(True, alpha=0.3)
    
    # Elevation vs VOD
    avg.plot(kind="scatter", x="Elevation_mean", y="VOD1_mean", s=3, ax=axes[1, 1])
    axes[1, 1].set_title("Elevation vs VOD")
    axes[1, 1].grid(True, alpha=0.3)
    plt.savefig(FIG / "scatter_vod_ke_pathlength.png", dpi=300)
    plt.tight_layout()
    plt.show()
    
    # -----------------------------------
    # anomaly calculation
    
    band_ids = ['VOD1']  # Example band IDs
    temporal_resolution = 60  # Example temporal resolution in minutes
    # Call the function
    vod_ts_combined, vod_avg = calculate_anomaly(vod_cell, band_ids, temporal_resolution, angular_cutoff=10)
    
    # -----------------------------------
    # Plot hemispheric distributions of VOD, extinction coefficient, and path length
    # Create a figure with 3 subplots for hemispheric distributions
    # Plot the hemispheric distribution of VOD
    plot_hemi(avg, hemi.patches(), "VOD1_mean",
              title="VOD Distribution", clim="auto", angular_cutoff=angular_cutoff)

    # Plot the hemispheric distribution of extinction coefficient
    plot_hemi(avg, hemi.patches(), "ke_mean",
              title="Extinction Coefficient", clim="auto", angular_cutoff=angular_cutoff)

    # Plot the hemispheric distribution of path length
    plot_hemi(avg, hemi.patches(), "pathlength_mean",
              title="Path Length (m)", clim="auto", angular_cutoff=angular_cutoff)