# Process GNSS-T VOD data description
> Konstantin Schellenberg, 2025-08-09

## Preprocessing GNSS-T VOD data

### Specs of raw data
- **Data format**: BINEX
- **Data source**: Christian Frankenberg (Caltech) and Jeffrey Wood (University of Missouri), field technician: Sami Overby
- **Time period**: 2025-04-01 to 2025-10-31
- **Native sampling rate**: 15 sec

### Processing steps
- **Step 1**: Convert BINEX to RINEX using `convbin -r binex -os` (teqc)
- **Step 2**: Calculate VOD from RINEX using package [gnssvod](https://github.com/vincenthumphrey/gnssvod/)
- **Step 3**: Anomaly calculation using customized Humphrey algorithm:
  - Calculate the median VOD for each satellite in a cell
  - Calculate the median VOD for each time step
  - Calculate the anomaly as the difference between the VOD and the median VOD for each satellite in a cell
  - Apply a spatial aggregation function to the anomalies (e.g., median)
  - Aggregate anomalies over time using a temporal aggregation function (e.g., median)
- **Step 4**: Mask anomalies based on wetness data from MOFLUX
- **Step 5**: Mask anomalies that are below a certain quantile threshold in detrended data (e.g., 0.02) to filter out dip-artifacts

### Processing parameters:
- **angular_resolution**: 1° \- (angular resolution for GNSS satellites; determines the spatial resolution of the VOD data)
- **temporal_resolution**: 30 min \- (temporal resolution for GNSS satellites; determines the time interval for final VOD data aggregation)
- **angular_cutoff**: >30°  \- (angular cutoff for GNSS satellites; satellites with an elevation angle below this threshold are not considered)
- **temporal**: "median"  \- (aggregation function for VOD offset added to the anomaly; can be "mean" or "median")
- **agg_fun_ts**: "median"  \- (aggregation function for time series)
- **agg_fun_satincell**: "median"  \- (Konstantin's aggregation function for satellite in cell; can be "mean" or "median")
- **minimum_nsat**: 10 (Minimum number of satellites in view on average in a time interval to be considered valid)
- **min_vod_quantile**: 0.02  (Lower VOD cutoff to filter dip-artifacts, e.g. 0.05 for 5\% quantile)
- **mask_wetness_globally**: True (Using the wetness data from MOFLUX to mask all VOD data)