# Scrapbook for task within `gnssvod`

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

