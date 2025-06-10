#!/usr/bin/env python
# -*- coding: utf-8 -*-

def create_vod_metadata():
    """Create a metadata dataframe from the VOD column descriptions."""
    # Parse the commented section to extract column information
    column_info = [
        ('VOD1_anom_tp', 'anom', 'tp', 'VOD anomaly (Humphrey algorithm)', 1, None, None, 'all'),
        ('VOD1', 'raw', None, 'Original VOD (L1)', 1, None, None, 'all'),
        ('VOD2', 'raw', None, 'Original VOD (L2)', 2, None, None, 'all'),
        ('VOD1_ke_anom', 'anom', 'ke', 'VOD anomaly', 1, None, None, 'all'),
        ('VOD2_ke_anom', 'anom', 'ke', 'VOD anomaly', 2, None, None, 'all'),
        ('VOD1_anom', 'anom', None, 'VOD anomaly', 1, None, None, 'all'),
        ('VOD2_anom', 'anom', None, 'VOD anomaly', 2, None, None, 'all'),
        ('Ns_t', 'metric', None, 'number of satellites', None, None, None, None),
        ('SD_Ns_t', 'metric', None, 'standard deviation of number of satellites', None, None, None, None),
        ('C_t_perc', 'metric', None, 'percentage of canopy coverage', None, None, None, None),
        ('VOD1_S31', 'raw', None, 'SBAS', 1, 'S31', None, None),
        ('VOD1_S33', 'raw', None, 'SBAS', 1, 'S33', None, None),
        ('VOD1_S35', 'raw', None, 'SBAS', 1, 'S35', None, None),
        ('S1_ref_S31', 'reference', None, 'reference', 1, 'S31', None, None),
        ('S1_ref_S33', 'reference', None, 'reference', 1, 'S33', None, None),
        ('S1_ref_S35', 'reference', None, 'reference', 1, 'S35', None, None),
        ('S1_grn_S31', 'ground', None, 'ground', 1, 'S31', None, None),
        ('S1_grn_S33', 'ground', None, 'ground', 1, 'S33', None, None),
        ('S1_grn_S35', 'ground', None, 'ground', 1, 'S35', None, None),
        ('VODe_S', 'other', None, '-', None, None, None, None),
        ('VOD1_S', 'mean', None, 'SBAS', 1, None, None, 'SBAS'),
        ('VOD2_S', 'mean', None, 'SBAS', 2, None, None, 'SBAS'),
        ('VOD1_ke_anom_gps', 'anom', 'ke', 'VOD anomaly', 1, None, None, 'GPS'),
        ('VOD2_ke_anom_gps', 'anom', 'ke', 'VOD anomaly', 2, None, None, 'GPS'),
        ('VOD1_anom_gps', 'anom', None, 'VOD anomaly', 1, None, None, 'GPS'),
        ('VOD2_anom_gps', 'anom', None, 'VOD anomaly', 2, None, None, 'GPS'),
        ('VOD1_ke_anom_gps+gal', 'anom', 'ke', 'VOD anomaly', 1, None, None, 'GPS+Galileo'),
        ('VOD2_ke_anom_gps+gal', 'anom', 'ke', 'VOD anomaly', 2, None, None, 'GPS+Galileo'),
        ('VOD1_anom_gps+gal', 'anom', None, 'VOD anomaly', 1, None, None, 'GPS+Galileo'),
        ('VOD2_anom_gps+gal', 'anom', None, 'VOD anomaly', 2, None, None, 'GPS+Galileo')
    ]
    
    # Add binned columns
    for band in [1, 2]:
        for algo in [None, 'ke']:
            for sat in ['gps', '']:
                for bin_num in range(5):
                    algo_str = f"_{algo}" if algo else ""
                    sat_str = f"_{sat}" if sat else ""
                    col_name = f"VOD{band}{algo_str}_anom_bin{bin_num}{sat_str}"
                    col_info = (col_name, 'binned anom', algo, f'VOD anomaly binned', band, None, bin_num,
                                'GPS' if sat else 'all')
                    column_info.append(col_info)
    
    # Add metric columns
    metric_cols = [
        ('angular_resolution', 'metric', None, 'angular resolution in degrees', None, None, None, None),
        ('angular_cutoff', 'metric', None, 'angular cutoff in degrees', None, None, None, None),
        ('temporal_resolution', 'metric', None, 'temporal resolution in seconds', None, None, None, None),
        ('hod', 'metric', None, 'hour of day (0-23)', None, None, None, None),
        ('doy', 'metric', None, 'day of year (1-365)', None, None, None, None),
        ('year', 'metric', None, 'year', None, None, None, None)
    ]
    column_info.extend(metric_cols)
    
    # Create DataFrame
    import pandas as pd
    metadata = pd.DataFrame(column_info, columns=[
        'column_name', 'data_type', 'algorithm', 'description',
        'band', 'satellite', 'bin', 'constellation'
    ])
    
    # Set index to column_name for easier lookup
    return metadata.set_index('column_name')
