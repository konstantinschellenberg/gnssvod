#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Understand and implement in gnssvod for my product
"""

def create_satellite_mask(df, min_satellites=13, satellite_col='Ns_t', **kwargs):
    """
    Create a mask based on minimum number of satellites.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with satellite count data
    min_satellites : int
        Minimum number of satellites required
    satellite_col : str
        Column name containing satellite count data

    Returns:
    --------
    pandas.Series
        Boolean mask where True means the row meets the satellite count requirement
    """
    if satellite_col not in df.columns:
        print(f"Warning: Satellite column '{satellite_col}' not found in the dataframe")
        return pd.Series(True, index=df.index)  # Default to no masking
    
    nan_in_nsat = pd.isna(df[satellite_col])
    min_sat = (df[satellite_col] >= min_satellites)
    # Combine conditions: True if NaN or meets minimum satellite count
    mask = nan_in_nsat | min_sat
    return mask


def create_vod_percentile_mask(df, vod_column='VOD1_anom', min_percentile=0.05, **kwargs):
    pass


def characterize_daily_vod(df, dataset_col='VOD1_anom_gps+gal', precip_quantile=0.9,
                           interpolate_when_day_too_many_nan_in_day=False,
                           min_hours_per_day=12):
    """
    Characterize precipitation patterns and calculate daily mean VOD values.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with VOD time series
    dataset_col : str
        Column name for the dataset to use for precipitation detection
    precip_quantile : float
        Quantile threshold for precipitation event detection (0-1)
    min_hours_per_day : int
        Minimum hours of data required per day for valid mean calculation

    Returns:
    --------
    pandas.DataFrame
        New columns: 'wetness_flag', 'VOD1_anom_masked', 'VOD1_daily'
    """
    result = pd.DataFrame(index=df.index)
    
    try:
        wetness = pd.read_csv(filepath_environmentaldata, index_col=0, parse_dates=True)["wet"]
        # make wetness mask (if wetness == "yes", then True)
        wetness_mask = (wetness == "yes").astype(int)
        # First convert to datetime, then ensure all dates are set to midnight
        wetness_mask.index = pd.to_datetime(wetness_mask.index, format='mixed', utc=True)
        wetness_mask.name = 'wetness_flag'
        # merge wetness mask on index (left on result)
        result = result.join(wetness_mask, how='left')
    
    except FileNotFoundError:
        print(f"Warning: '{filepath_environmentaldata}' not found. Using default precipitation detection.")
        raise ValueError("Wetness data not found. Please provide a valid path to wetness data.")
        # If wetness data is not available, initialize with zeros
        result['wetness_flag'] = 0
        
        # Create flag for upper quantile (precipitation events) based on monthly thresholds
        result['month'] = df.index.month  # Extract month information
        
        # Calculate and apply threshold for each month separately
        for month in result['month'].unique():
            # Get data for this month only
            month_data = df.loc[df.index.month == month, dataset_col]
            
            if not month_data.empty:
                # Calculate threshold for this month
                month_threshold = month_data.quantile(precip_quantile)
                
                # Apply threshold to flag precipitation events for this month
                month_mask = (df.index.month == month) & (df[dataset_col] > month_threshold)
                result.loc[month_mask, 'wetness_flag'] = 1
        # Convert to integer type and drop the temporary month column
        result['wetness_flag'] = result['wetness_flag'].astype(int)
        result.drop('month', axis=1, inplace=True)
    
    # Mask anomalies during precipitation events
    var_masked = f'{dataset_col}_masked'
    
    result[var_masked] = df[dataset_col].copy()
    result.loc[result['wetness_flag'] == 1, var_masked] = np.nan
    
    # Calculate required samples based on data frequency
    time_delta = df.index[1] - df.index[0]
    min_samples_per_day = min_hours_per_day * (3600 / pd.Timedelta(time_delta).total_seconds())
    
    # convert index to vizualization timezone
    result.index = result.index.tz_convert(visualization_timezone)
    # Calculate daily mean VOD
    daily_counts = result[var_masked].groupby(pd.Grouper(freq='D')).count()
    daily_means = result[var_masked].groupby(pd.Grouper(freq='D')).mean()
    
    # Identify days with insufficient data
    insufficient_days = daily_counts < min_samples_per_day
    daily_means[insufficient_days] = np.nan
    
    # Interpolate values for days with insufficient data
    if interpolate_when_day_too_many_nan_in_day:
        print("Interpolating daily means for days with too many NaNs...")
        interpolated_daily = daily_means.interpolate(method='linear', limit=5)
        # Reindex back to original timestamp frequency
        result['VOD1_daily'] = interpolated_daily.reindex(df.index, method='ffill')
    else:
        # First reindex without filling method to map days to their original timestamps
        result['VOD1_daily'] = daily_means.reindex(df.index)
        # Then forward fill only within each day, preserving NaN days
        result['VOD1_daily'] = result.groupby(result.index.date)['VOD1_daily'].transform('ffill')
    
    # tranform back to utc
    result.index = result.index.tz_convert('UTC')
    
    return result