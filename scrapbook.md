# Scrapbook for task within `gnssvod`

Todos:
- Write metadata and filename specification in `04_merge…`
- Precipiation filter is horrible for long time series!!

## 6/9/25

- Calculation of bins now band-specific
- Change in standard time series aggregation from `mean` to `median` (more robust)
- Change in calculation of average VOD per tps cell from `mean` to `median` (more robust?)
- Change in calculation of average overall VOD from `mean` to `median` (more robust)
- Tradeoff between time interval, angular neigborhood, and number of satellites
  - time interval higher
  - angular neighborhood higher
      –> more satellites

New variable in anomaly processing: `n`:
  - Median number of observations per tps-unit per time interval

## 6/8/25

Rationale VOD optimized:

---------

    Goals:
    1. Best long-term vod estimate
    2. Best weekly vod (dry-down)
    3. Best diurnal vod (daily)

    Manufacture signal by
    VOD = 1 + 2 + 3

    -----
    1. Best long-term vod estimate
        - Fullest canopy view of high-credibility constellations:
        - Includes canopy gaps to

        Tests:
        * compare with literature VOD values

        Candidates:
        * mean (VOD(GPS, GLONASS)), θ > 30°

    -----
    2. Best weekly vod (dry-down)
        - Usually not diurnal trend
        - Precipitation events
        - Trend

        Tests:
        * compare with radiometer
        * compare with branch water potential

        Candidates:
        * SBAS (33 > 35 > [31]*excluded) –> all or mixture?

        Todo:
        - normalize magnitude to VOD1_anom magnitude

    -----
    3. Best diurnal vod (daily)
        - Strong diurnal vars don't show seasonal trend and precipitation (but dew)

        Tests:
        * compare with branch water potential
        * independent of temperature?

        Candidates:
        * VOD1_anom_highbiomass (VOD1_anom_bin2, VOD1_anom_bin3, VOD1_anom_bin4) (60% of high biomass)
            - does not show strong seasonal trend
            - does not show precipitation events

        Todo:
        * heavy smoothing (savitzky-golay filter)
        * detrending (lowess filter) –> the overall dynamics are reduce...



    Sbas:
    - Why are stripes in S31 ref? is the WAAS sat moving? However, not seen in VOD :)

    Claude 3.7 Sonnet Thinking:
    I want you to do the following analysis. I want to create an optimal VOD estimator variable that harnesses the different strength of signales (cols) as desribed in  the underneat text. Please 1)  Characterize Precipitation patterns using temperal anomalies in VOD1_anom (create a flag based on the quantile of upper 10% of the signal). Mask anom by the flag. Calculate mean daily VOD values (transform not summarize) and add to vod_ts. 2) use the SBAS data to characterize the weekly trends (dry-down events). First, subtract the mean of each VOD1_S?? band from the time series, then caluculate the mean VOD1_S. 3) The best diurnal VOD descriptor is VOD1_anom_highbiomass. Please apply a window-size (6 hours, polyorder 2) savitzky-golay filter to the data. Finally merge all three product to a new dataframe and add a new column where all of the them are added
------

This flowchart illustrates:
1. The three input data sources
2. Processing steps for each component:
   - Long-term trend (precipitation filtering)
   - Weekly trend (SBAS satellites)
   - Diurnal pattern (high biomass filtering)
3. Z-score transformation (the default method)
4. Alternative combination methods
5. Final VOD optimal estimator production

The z-score approach standardizes each component before combining them, ensuring no single component dominates due to scale differences, then transforms back to the original scale and adds the long-term trend.

```mermaidflowchart TD
  subgraph "Input Data"
    VOD1_anom["VOD anomaly"]
    SBAS["SBAS Satellites (S33, S35)"]
    VOD1_highbiomass["VOD anomaly (high-biomass, mean of bins 2-4 (of 0-4))"]
    VOD["Raw VOD data"]
  end

  subgraph "Long-term Trend"
    VOD1_anom -->|"Flag precipitation using >90th quantile VOD data (this should idealy be precipitation data)"| mask_precip["Mask precipitation"]
    mask_precip -->|"Daily mean"| daily_mean["Daily VOD"]
    daily_mean -->|"Interpolate gaps"| interp["Interpolated VOD"]
    interp --> VOD1_daily["Daily VOD anomaly"]
    VOD-->|"Add mean"| VOD1_daily["Daily VOD anomaly"]
  end

  subgraph "Weekly Trend"
    SBAS -->|"Normalize bands (come in VOD, not anomaly)"| sbas_norm["Normalized SBAS"]
    sbas_norm -->|"Calculate mean"| VOD-SBAS["VOD-SBAS"]
  end

  subgraph "Diurnal Pattern"
    VOD1_highbiomass -->|"Savitzky-Golay filter"| sg_filter["Filtered signal"]
    sg_filter -->|"(Optional LOESS), can the diurnal VOD contain a trend?)"| VOD-diurnal["VOD-diurnal"]
  end

  VOD1_daily --> Trend["Trend"]
  VOD-SBAS --> weekly["weekly"]
  VOD-diurnal --> diurnal["diurnal"]

  subgraph "Z-Score Transformation"
    weekly -->|"Standardize"| z_weekly["Z-weekly"]
    diurnal -->|"Standardize"| z_diurnal["Z-diurnal"]
    z_weekly & z_diurnal -->|"Sum"| z_sum["Z-sum"]
    z_sum -->|"Back-transform using inverse mean coefficients (μ, σ)"| scaled_z["Scaled Z"]
  end

  scaled_z --> VOD_optimal_zscore["VOD_optimal_zscore"]
  Trend -->|"add"|VOD_optimal_zscore["VOD_optimal_zscore"]

  VOD_optimal_zscore -->|"Output options"| final["Final VOD"]
```
## 6/6/25

- Change standard time series aggregation to "median" instead of "mean" (more robust)
- NO BeiDou satellites tracked!

From https://www.gpsrchive.com/Shared/Satellites/Satellites.html#Satellite%20Identification

| SBAS PRN | System  | Identification      | NORAD ID | Azimuth | Elevation | Launch Date       | Checked date |
|----------|---------|---------------------|----------|---------|-----------|-------------------|--------------|
| 131      | WAAS    | Eutelsat 117 West B | 41589    | 216.5   | 31.1      | June 15, 2016     | June 6, 2025 |
| 133      | WAAS    | SES-15              | 42709    | 230.3   | 38.3      | May 18, 2017      | June 6, 2025 |
| 135      | WAAS    | Galaxy 30           | 46114    | 225.8   | 33.7      | August 15, 2020   | June 6, 2025 |
| 148      | AL-SBAS | ALCOMSAT 1          | 43039    | 104.6   | 8.9       | December 10, 2017 | June 6, 2025 |
| 158      | -       | -                   | -        | -       | -         | -                 | -            |

## 6/5/25

- ke for each observation (sv, time, elevation, azimuth)

## 6/3/25

| Metric name                                         | Symbol            | Mathematical representation          | Rationale                                           |
|-----------------------------------------------------|-------------------|:-------------------------------------|-----------------------------------------------------|
| Number of sats in view                              | $N_s(t)$          |                                      | Gaps in the overall coverage                        |
| Standard deviation of the number of sats in view    | $\sigma_{N_s(t)}$ |                                      | Variability of observations with in a time interval |
| Fraction of sky currently observed (cutoff applied) | $C(t)$            | $C(t) = \frac{C_t * 100}{C_{total}}$ | Probably correlated to  $N_s(t)$                    |
| Binned fraction of sky observed (cutoff applied)    | $C_b(t)$          | $C_i(t) = \frac{C_{t,i}*100}{C_t}$   | Variability in biomass areas observed               |

## 6/2/25

- Save intermediate results under /temp with the hope that this reduces the memory weight in the anomaly processing
- Final layout of multiparameter plots
- Conception of new gnss features

## 6/1/25

- Implemented changing gnss processing parameters
- Iterate over parameters to compare results

## Notes

- RINEX data worthless, no CN0 data included
- Ground data (doy 150, 2022) onwards corrupt? no obs file expelled, only nav
- What happened DOY 150 (2022)?
  - binex file size increases
  - using `convbin`, only nav file is created, no obs. Even on explicit request
  - RINEX files from Sami don't contain SNR in both, before and after DOY 150 
- Clock file: `GFZ0MGXRAP_20243440000_01D_30S_CLK.CLK` not found: Copied clock file from day before artificially to this date. Workaround for now.
- Orbit file: `GFZ0MGXRAP_20241740000_01D_05M_ORB.SP3` not found: Copied orbit file from day before artificially to this date. Workaround for now.
- Clock file: `GFZ0MGXRAP_20241740000_01D_30S_CLK.CLK` not found: Copied clock file from day before artificially to this date. Workaround for now.
- DOY 24166 (2024) last before a 5-month gap. Why is that? Try to find RINEX. Issue might be the modify data: data been overwritten?
- Check [RINEX handbook](http://acc.igs.org/misc/rinex304.pdf) for PRN codes
- Check [Sbas codes](https://media.defense.gov/2016/Jul/26/2001583103/-1/-1/1/PRN%20CODE%20ASSIGNMENT%20PROCESS.PDF)

Anomalous dates
- 2024 344 (no clock file)
- 2024 174 (no clock and orbit file)


## Todos

- [x] Make Binex converter
- [x] Process towards RINEX
- [x] installed gfzrnx: Does not support BINEX conversion
- [x] installing teqc
- [x] Implement teqc pipeline
- [x] Check missing ground obs data –> using BINEX-RINEX 2.11 to recover
[//]: # (- [ ] Multiprocessing not yet working: Results differ when redone...)
- [x] Run all nc calculations (integrate new data)
- [ ] reduce file size in `GATHER` by merging S?? fields

## Comm
- Ask Benjamin L. if they were successful with RINEX proc

## Features
- BINEX –> RINEX converter

## Binex DRIVERS

### RTKLIB

Use RTKLIB to convert BINEX to RINEX

```bash
Installation:
wget https://github.com/tomojitakasu/RTKLIB/archive/refs/tags/2.4.3.b34L-pre0.tar.gz
tar -xzf 2.4.3.b34L-pre0.tar.gz

cd app/convbin
make
sudo make install

convbin is now an executable.
```

### teqc

Download from [UNAVCO](https://www.unavco.org/software/data-processing/teqc/teqc.html)

```bash
cp teqc /usr/local/bin
chmod a+x /usr/local/bin/teqc
teqc -version
```

