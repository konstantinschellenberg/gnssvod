#!/usr/bin/env python
# -*- coding: utf-8 -*-
import warnings

import numpy as np
import pandas as pd

from analysis.aux_plotting import plot_sv_observation_counts


def vod_fun(grn, ref, ele):
    return -np.log(np.power(10, (grn - ref) / 10)) * np.cos(np.deg2rad(90 - ele))


def ke_fun(vod, d, elevation):
    """
    vod: vegetation optical depth
    d: canopy height
    elevation: elevation angle in degrees
    ke: effective extinction coefficient
    """
    theta = 90 - elevation  # convert elevation to zenith angle
    pathlength = d / np.cos(np.deg2rad(theta))
    ke = vod / pathlength
    return ke, pathlength


def calculate_extinction_coefficient(vod_with_cells, band_ids, canopy_height=None, z0=None):
    """
    Calculate extinction coefficient (ke) for each band in the VOD dataset.

    Parameters
    ----------
    vod_with_cells : pandas.DataFrame
        VOD data with Elevation information
    band_ids : list
        List of band identifiers (e.g., ['VOD1', 'VOD2'])
    canopy_height : float, optional
        Height of the canopy in meters
    z0 : float, optional
        Height of the ground receiver in meters

    Returns
    -------
    tuple
        vod_with_cells : pandas.DataFrame
            Input DataFrame with added extinction coefficient columns
        band_ids : list
            Updated list of band identifiers including ke bands
    """
    if canopy_height is None:
        raise ValueError("Canopy height must be provided if extinction coefficient is to be calculated.")
    if z0 is None:
        raise ValueError("Ground receiver height (z0) must be provided if extinction coefficient is to be calculated.")
    
    di = canopy_height - z0
    
    # Calculate ke for each band and add to vod_with_cells
    for band in band_ids:
        vod_with_cells[f"{band}_ke"], _ = ke_fun(
            vod_with_cells[band],
            d=di,
            elevation=vod_with_cells["Elevation"]
        )
        # todo: add a vod_rect as ke*d to with tau values
    
    # append 'VOD_ke' to band_ids
    updated_band_ids = [f"{band}_ke" for band in band_ids] + band_ids  # add ke bands to the list
    
    return vod_with_cells, updated_band_ids


def create_aggregation_dict(vod_with_cells, band_ids, agg_fun_ts='mean'):
    """
    Create an aggregation dictionary for groupby operations based on column types.

    Parameters
    ----------
    vod_with_cells : pandas.DataFrame
        DataFrame containing VOD data
    band_ids : list
        List of band identifiers

    Returns
    -------
    dict
        Dictionary with column names as keys and aggregation operations as values
    """
    agg_dict = {}
    # Add base band_ids (without suffixes) for mean, std, count
    for col in band_ids:
        if col in vod_with_cells.columns:
            if col.endswith('_ke'):
                agg_dict[col] = ['mean', 'std']
            else:
                agg_dict[col] = [agg_fun_ts, 'std', 'count']
    # Add _ref and _grn columns for mean, std only
    ref_grn_cols = [col for col in vod_with_cells.columns if "_ref" in col or "_grn" in col]
    for col in ref_grn_cols:
        agg_dict[col] = [agg_fun_ts, 'std']
    # Keep Azimuth and Elevation if they exist
    for col in ["Azimuth", "Elevation"]:
        if col in vod_with_cells.columns:
            agg_dict[col] = agg_fun_ts
    return agg_dict


def calculate_binned_sky_coverage(vod_with_cells, vod_avg, band_ids, temporal_resolution, num_bins=5,
                                  plot_bins=False, suffix=None):
    """
    Calculate Ci(t): Binned percentage of sky plots covered per time period, based on VOD percentiles.

    Parameters
    ----------
    df : pandas.DataFrame
        Time series DataFrame to be augmented with bin coverage information
    vod_with_cells : pandas.DataFrame
        VOD data with cell IDs
    vod_avg : pandas.DataFrame
        Average VOD values per grid cell
    band_ids : list
        List of band identifiers to calculate bin coverage for
    temporal_resolution : int
        Temporal resolution in minutes
    num_bins : int, default=5
        Number of percentile bins to create
    plot_bins : bool, default=False
        Whether to plot the bin histograms for visualization

    Returns
    -------
    pandas.DataFrame
        Time series DataFrame with added bin coverage columns
    """
    final = pd.DataFrame()
    for band in band_ids:
        # Get the mean VOD values for this band
        band_means = vod_avg[f"{band}_mean"]
        
        # Create bins based on VOD percentiles
        try:
            # Try to create equal-sized bins by percentile
            bins = pd.qcut(band_means, num_bins, labels=False, duplicates='drop')
        except ValueError:
            # Fall back to equal-width bins if not enough unique values
            bins = pd.cut(band_means, min(num_bins, len(band_means.unique())), labels=False)
        
        
        # Create a mapping from CellID to bin
        cell_to_bin = pd.Series(bins.values, index=band_means.index, name='bin')
        
        # Count total cells in each bin
        total_cells_per_bin = cell_to_bin.value_counts().sort_index()
        
        # Create a temporary DataFrame with just the necessary data
        temp_df = pd.DataFrame({
            'Epoch': vod_with_cells.index.get_level_values('Epoch'),
            'CellID': vod_with_cells['CellID']
        }).drop_duplicates()
        
        # Add bin information based on CellID
        temp_df = temp_df.join(cell_to_bin, on='CellID')
        
        # Group by time window and bin, count unique cells
        bin_counts = temp_df.groupby([
            pd.Grouper(key='Epoch', freq=f"{temporal_resolution}min"),
            'bin'
        ])['CellID'].nunique().unstack(fill_value=0)
        
        # Calculate percentage of each bin covered relative to total current coverage
        bin_coverage = pd.DataFrame(index=bin_counts.index)
        # Calculate total cells covered at each time step across all bins
        total_current_coverage = bin_counts.sum(axis=1)
        for bin_idx in range(num_bins):
            if bin_idx in bin_counts.columns and bin_idx in total_cells_per_bin.index:
                # Calculate relative coverage (percentage of current total coverage this bin represents)
                bin_coverage[f"Ci_t_{band}_bin{bin_idx}_pct"] = (
                        bin_counts[bin_idx] / total_current_coverage.replace(0, np.nan) * 100
                )
        final.loc[:, f"Ci_t_{band}_bin*_pct"] = bin_coverage
        
    return final


def calc_anomaly_vh(vod, band_ids, suffix="",
                    temporal_resolution=30, **kwargs):
    """
    Vincent (global per-cell) anomaly calculation.

    Compute one per-CellID mean over the full input span (no interval partition),
    join it back to the raw rows, form anomalies as:
        anomaly = VOD - VOD_mean_cell + global_offset
    and aggregate to the requested temporal resolution.

    Parameters
    ----------
    vod : pandas.DataFrame
        Must contain columns for each band in band_ids and 'CellID'.
        Index must have level 'Epoch' (or an 'Epoch' column convertible to index).
    band_ids : list[str]
        Bands to process (e.g. ['VOD1','VOD2']).
    suffix : str, default ""
        Suffix added to anomaly columns (prepended with '_' if non-empty).
    temporal_resolution : int | str, default 30
        If int/float, interpreted as minutes (e.g. 30 -> '30min').
        If str, passed directly to pd.Grouper(freq=...).
    **kwargs :
        Optional:
          agg_fun_vodoffset : str, default 'median'
              Offset operator re-added after removing per-cell means.
          agg_fun_ts : str, default 'median'
              Temporal aggregation operator for anomalies.

    Returns
    -------
    pandas.DataFrame
        Aggregated anomaly time series with columns: f"{band}_anom{_suffix}"
    """
    if vod is None or len(vod) == 0:
        return pd.DataFrame()

    # Resolve frequency (accept minutes int or pandas offset string)
    def _to_freq(val):
        if isinstance(val, (int, float)):
            return f"{int(val)}min"
        return val

    agg_fun_vodoffset = kwargs.get("agg_fun_vodoffset", "median")
    agg_fun_ts = kwargs.get("agg_fun_ts", "median")
    freq = _to_freq(temporal_resolution)

    # Normalize suffix formatting
    _suffix = f"_{suffix}" if suffix else ""

    # Ensure we have an Epoch index level
    work = vod.copy()
    if 'Epoch' not in work.index.names:
        if 'Epoch' in work.columns:
            work = work.set_index('Epoch', drop=True)
        else:
            raise ValueError("vod must have 'Epoch' as index level or column.")
    if 'CellID' not in work.columns:
        raise ValueError("vod must contain a 'CellID' column.")

    # Sort by time for deterministic grouping
    work = work.sort_index()

    # Per-CellID means over full span
    cell_means = (
        work
        .groupby('CellID')[band_ids]
        .mean()
        .add_suffix('_mean')
    )

    # Join back on CellID
    work = work.join(cell_means, on='CellID')

    # Global offsets per band (using chosen operator over entire dataset)
    global_offsets = {
        band: getattr(work[band], agg_fun_vodoffset)()
        for band in band_ids
        if band in work.columns
    }

    # Form anomalies per band
    for band in band_ids:
        mean_col = f"{band}_mean"
        if band not in work.columns or mean_col not in work.columns:
            continue
        work[f"{band}_anom{_suffix}"] = work[band] - work[mean_col] + global_offsets[band]

    # Temporal aggregation to requested resolution
    anom_cols = [f"{band}_anom{_suffix}" for band in band_ids if f"{band}_anom{_suffix}" in work.columns]
    if not anom_cols:
        return pd.DataFrame()

    ts = (
        work[anom_cols]
        .groupby(pd.Grouper(freq=freq, level='Epoch'))
        .agg(agg_fun_ts)
        .dropna(how='all')
    )

    return ts, cell_means



def calc_anomaly_ks(vod, band_ids, ks_strategy="sv", suffix="",
                    temporal_resolution=30, **kwargs):
    """
    Konstantin (SV-specific per-cell) anomaly calculation — concise code, verbose annotations.

    Concept (hybrid of AK & VH):
      • Like VH: subtract a per-cell mean, but
      • Do it per-SV: means are computed per (SV, CellID) across the full span (no interval partition as in AK).
      • Add back a per-SV band offset to preserve level (median/mean per-SV).

    Vectorized pipeline:
      1) Normalize suffix + resolve frequency (int minutes or pandas offset).
      2) Ensure ['Epoch','SV'] are index levels and 'CellID' exists.
      3) Compute per-(SV,CellID) means for all bands in one grouped operation.
      4) Join means back (broadcast per row).
      5) Compute per-SV offsets per band (groupby('SV').transform(...)) and form anomalies:
              anom = raw - mean_(SV,CellID) + offset_(SV)
      6) Aggregate anomalies to the requested temporal resolution on 'Epoch'.

    Parameters
    ----------
    vod : pandas.DataFrame
        Must contain:
          - index levels: 'Epoch' (datetime-like), 'SV' (e.g., G21, E05, ...)
          - column: 'CellID'
          - columns for each band in band_ids (e.g., 'VOD1','VOD2', ...)
    band_ids : list[str]
        Bands to process.
    suffix : str, default ""
        Optional suffix; if non-empty, leading underscore is added automatically.
    temporal_resolution : int | str, default 30
        - int/float -> treated as minutes, e.g. 30 -> '30min'
        - str       -> passed to pd.Grouper(freq=...)
    **kwargs :
        Optional:
          agg_fun_vodoffset : str, default 'median'
              Operator for offset re-addition (per SV per band).
          agg_fun_ts : str, default 'median'
              Operator for time aggregation of anomalies.
          agg_fun_satincell : str, default 'median'
              Unused here (kept for signature symmetry).
          eval_num_obs_tps : bool, default False
              If True and show=True, plot observation counts per (SV,CellID).
          show : bool, default False
              If True, show optional evaluation plots.

    Returns
    -------
    pandas.DataFrame
        Time-aggregated anomalies with columns: f"{band}_anom{suffix}"
        Indexed by 'Epoch'.

    Notes
    -----
    - No interval partitioning (AK does), this is a global (per-SV) per-cell mean removal.
    - Fully vectorized; no per-SV loops, making it fast and concise.
    """
    if vod is None or len(vod) == 0:
        return pd.DataFrame()

    # ---- config / helpers ----------------------------------------------------
    def _to_freq(val):
        """Translate temporal_resolution to a pandas offset string."""
        if isinstance(val, (int, float)):
            return f"{int(val)}min"
        return val

    agg_fun_vodoffset = kwargs.get("agg_fun_vodoffset", "median")
    agg_fun_ts = kwargs.get("agg_fun_ts", "median")
    # eval_num_obs_tps = kwargs.get("eval_num_obs_tps", False)
    # show = kwargs.get("show", False)
    freq = _to_freq(temporal_resolution)
    _suffix = f"_{suffix}" if suffix else ""
    assert ks_strategy in ("sv","con"), "ks_strategy must be 'sv' or 'con'"
    
    strategy = "SV" if ks_strategy == "sv" else "Constellation"

    # ---- input validation / index normalization ------------------------------
    work = vod.copy()
    # Ensure 'Epoch' is an index level
    if 'Epoch' not in work.index.names:
        if 'Epoch' in work.columns:
            work = work.set_index('Epoch', drop=True)
        else:
            raise ValueError("vod must have 'Epoch' as index level or column.")
    # Ensure 'SV' is an index level
    if 'SV' not in work.index.names:
        if 'SV' in work.columns:
            work = work.set_index('SV', append=True)
        else:
            raise ValueError("vod must contain 'SV' either as index level or column.")
    # Ensure CellID exists
    if 'CellID' not in work.columns:
        raise ValueError("vod must contain a 'CellID' column.")
    
    if ks_strategy == "con":
        # group sv to constellation mapping
        # GPS -> G, GALILEO -> E, GLONASS -> R, BEIDOU -> C
        def sv_to_con(sv):
            if sv.startswith('G'):
                return 'GPS'
            elif sv.startswith('E'):
                return 'GALILEO'
            elif sv.startswith('R'):
                return 'GLONASS'
            elif sv.startswith('C'):
                return 'BEIDOU'
            else:
                return 'OTHER'
        sv_levels = work.index.get_level_values('SV')
        con = sv_levels.map(sv_to_con)
        # add con as column
        work = work.reset_index()
        work['Constellation'] = con
        # make Epoch and con indices
        work = work.set_index(['Epoch', 'Constellation'], drop=True)

    work = work.sort_index()  # deterministic grouping

    # ---- 1) per-(SV, CellID) means ------------------------------------------
    # Compute means for all bands at once; columns become e.g. 'VOD1_mean'
    sv_cell_means = (
        work.groupby(["SV", 'CellID'])[band_ids]
            .mean()
            .add_suffix('_mean')
    )
    
    # Join means back to each row using the two keys
    work = work.join(sv_cell_means, on=['SV', 'CellID'])

    # ---- 2) per-SV offsets and anomalies ------------------------------------
    # Build anomalies for each band in a tight loop (vectorized operations).
    anom_cols = []
    for band in band_ids:
        mean_col = f"{band}_mean"
        if band not in work.columns or mean_col not in work.columns:
            continue
        # Per-SV offset: median/mean over raw values for that band and SV
        # transform aligns offsets back to each row (same index shape)
        offset_sv = work.groupby(strategy)[band].transform(agg_fun_vodoffset)
        anom_col = f"{band}_anom{_suffix}"
        work[anom_col] = work[band] - work[mean_col] + offset_sv
        anom_cols.append(anom_col)

    if not anom_cols:
        return pd.DataFrame()

    # ---- (Optional) evaluate observations per (SV, CellID) -------------------
    # Keep behavior from the previous implementation; gated by flags.
    # if eval_num_obs_tps and show:
    #     try:
    #         counts = work.groupby(['SV', 'CellID']).size().to_frame('n')
    #         # Reuse the existing plotting helper
    #         plot_sv_observation_counts(counts.reset_index(), min_threshold=10, figsize=(6, 4))
    #     except Exception:
    #         pass

    # ---- 3) temporal aggregation on 'Epoch' ---------------------------------
    ts = (
        work[anom_cols]
            .groupby(pd.Grouper(freq=freq, level='Epoch'))
            .agg(agg_fun_ts)
            .dropna(how='all')
    )

    return ts

def calc_anomaly_ksak(vod, band_ids, timedelta: pd.Timedelta, suffix="",
                      temporal_resolution=30, **kwargs):
    """
    Hybrid KS+AK anomaly calculation (interval hemi means + per-SV anomalies).

    Idea:
      1) Partition timeline into fixed-length intervals (timedelta).
      2) For each interval & CellID compute hemi mean per band using ALL SV together
         (no SV distinction for the interval mean) -> interval_cell_mean.
      3) For each row (Epoch, SV, CellID) form anomaly:
            anomaly = raw_band - interval_cell_mean + sv_band_offset
         where sv_band_offset = per-SV statistic (median/mean) of raw band over full span.
      4) Aggregate anomalies to requested temporal_resolution.

    Differences:
      - AK: interval mean per CellID (done) + global offset.
      - KS: per-(SV,CellID) mean + per-SV offset.
      - KS+AK hybrid: interval per-CellID mean (all SV pooled) + per-SV offset.

    Parameters
    ----------
    vod : DataFrame
        MultiIndex (Epoch[, SV]) or columns 'Epoch','SV'; must contain 'CellID' and band_ids.
    band_ids : list[str]
        Bands to process.
    timedelta : pd.Timedelta
        Interval block length (e.g. pd.Timedelta('7D')).
    suffix : str, default ""
        Optional suffix for anomaly columns (prepended with '_' if non-empty).
    temporal_resolution : int | str, default 30
        Minutes (int/float -> 'XXmin') or pandas offset string.
    **kwargs :
        agg_fun_vodoffset : str, default 'median'
            Function name applied per SV for offset (e.g. 'median' or 'mean').
        agg_fun_ts : str, default 'median'
            Aggregation over time windows.
        eval_num_obs_tps : bool, default False
            If True and show=True plot observation counts (SV,CellID).
        show : bool, default False
            Show optional evaluation plot.

    Returns
    -------
    (DataFrame, DataFrame)
        (aggregated anomalies, interval CellID means)
        Anomaly columns: f"{band}_anom{suffix}"

    Notes
    -----
    - Interval means ignore SV to represent hemispheric cell state.
    - Per-SV offset re-centers each satellite relative to its own long-term level.
    - Fully vectorized (no explicit Python loops except band loop).
    """
    if vod is None or len(vod) == 0:
        return pd.DataFrame(), pd.DataFrame()
    if not isinstance(timedelta, pd.Timedelta):
        raise TypeError("timedelta must be a pandas.Timedelta instance")

    # ---- helpers / config ----
    def _to_freq(val):
        if isinstance(val, (int, float)):
            return f"{int(val)}min"
        return val
    freq = _to_freq(temporal_resolution)
    agg_fun_vodoffset = kwargs.get("agg_fun_vodoffset", "median")
    agg_fun_ts = kwargs.get("agg_fun_ts", "median")
    eval_num_obs_tps = kwargs.get("eval_num_obs_tps", False)
    show = kwargs.get("show", False)
    _suffix = f"_{suffix}" if suffix else ""

    # ---- index normalization ----
    work = vod.copy()
    # Ensure Epoch index level
    if 'Epoch' not in work.index.names:
        if 'Epoch' in work.columns:
            work = work.set_index('Epoch', drop=True)
        else:
            raise ValueError("vod must have 'Epoch' as index level or column.")
    # Ensure SV index level
    if 'SV' not in work.index.names:
        if 'SV' in work.columns:
            work = work.set_index('SV', append=True)
        else:
            raise ValueError("vod must contain 'SV' as index level or column.")
    if 'CellID' not in work.columns:
        raise ValueError("vod must contain 'CellID' column.")

    work = work.sort_index()

    # ---- derive interval_start per row (AK style) ----
    epoch = work.index.get_level_values('Epoch')
    first = epoch.min()
    secs_from_first = (epoch - first).total_seconds()
    interval_idx = (secs_from_first // timedelta.total_seconds()).astype(int)
    work['interval_start'] = first + interval_idx * timedelta

    # ---- compute interval hemi means (ALL SV pooled) ----
    # Group only by interval_start + CellID, ignoring SV
    interval_means = (
        work
        .groupby(['interval_start', 'CellID'])[band_ids]
        .mean()
        .add_suffix('_mean')
    )

    # Join back
    work = work.join(interval_means, on=['interval_start', 'CellID'])

    # ---- compute per-SV offsets (KS style) ----
    # For each band do groupby('SV').transform(agg_fun_vodoffset)
    offsets = {}
    for band in band_ids:
        if band not in work.columns:
            continue
        offsets[band] = work.groupby('SV')[band].transform(agg_fun_vodoffset)

    # ---- form anomalies ----
    anom_cols = []
    for band in band_ids:
        mean_col = f"{band}_mean"
        if band not in work.columns or mean_col not in work.columns:
            continue
        anom_col = f"{band}_anom{_suffix}"
        # raw - interval_cell_mean + per-SV offset
        work[anom_col] = work[band] - work[mean_col] + offsets[band]
        anom_cols.append(anom_col)

    if not anom_cols:
        return pd.DataFrame(), interval_means

    # ---- optional evaluation ----
    if eval_num_obs_tps and show:
        try:
            counts = work.groupby(['SV', 'CellID']).size().to_frame('n')
            plot_sv_observation_counts(counts.reset_index(), min_threshold=10, figsize=(6, 4))
        except Exception:
            pass

    # ---- temporal aggregation ----
    ts = (
        work[anom_cols]
        .groupby(pd.Grouper(freq=freq, level='Epoch'))
        .agg(agg_fun_ts)
        .dropna(how='all')
    )

    return ts, interval_means

def calc_anomaly_ak(vod, band_ids, timedelta: pd.Timedelta, suffix="",
                    temporal_resolution="", **kwargs):
    """
    Interval-wise anomaly calculation (Alex concept).

    For each fixed-length interval (size = `timedelta`):
      1. Compute per-CellID mean VOD (vod_avg_interval).
      2. Join those means onto the raw rows of that interval.
      3. Form anomalies as (VOD - per-CellID interval mean + global offset).
      4. Aggregate anomalies to `temporal_resolution`.

    Parameters
    ----------
    vod : pandas.DataFrame
        Must contain columns for each band in band_ids and 'CellID'; index must have level 'Epoch'.
    band_ids : list[str]
        Bands to process (e.g. ['VOD1','VOD2']).
    timedelta : pd.Timedelta
        Length of each interval block (e.g. pd.Timedelta('1D')).
    suffix : str, default ""
        Suffix added to anomaly columns (prepended with '_' if non-empty).
    temporal_resolution : str, default "30min"
        Pandas offset alias for temporal aggregation of anomalies.
    **kwargs :
        Optional:
          agg_fun_vodoffset : str, default 'median'
              Offset operator re-added after removing interval means.
          agg_fun_ts : str, default 'median'
              Temporal aggregation operator for anomalies.

    Returns
    -------
    pandas.DataFrame
        Aggregated anomaly time series with columns: f"{band}_anom{_suffix}"
    """
    if vod.empty:
        return pd.DataFrame()

    # Validate timedelta
    if not isinstance(timedelta, pd.Timedelta):
        raise TypeError("timedelta must be a pandas.Timedelta instance")

    agg_fun_vodoffset = kwargs.get("agg_fun_vodoffset", "median")
    agg_fun_ts = kwargs.get("agg_fun_ts", "median")

    # Normalize suffix formatting
    _suffix = f"_{suffix}" if suffix else ""

    # Keep working copy
    work = vod.copy()

    # Sort by time
    work = work.sort_index()

    # Derive interval start timestamps for each row, access multiindex level 'Epoch'
    first = work.index.get_level_values('Epoch').min()
    # Compute offset in whole timedelta units
    delta_seconds = (work.index.get_level_values('Epoch') - first).total_seconds()
    block_len = timedelta.total_seconds()
    interval_indexer = (delta_seconds // block_len).astype(int)
    work['interval_start'] = first + (interval_indexer * timedelta)

    # Compute per-interval, per-CellID means
    # Result index: (interval_start, CellID)
    interval_means = (
        work
        .groupby(['interval_start', 'CellID'])[band_ids]
        .mean()
        .add_suffix('_mean')
    )

    # Join back on (interval_start, CellID)
    work = work.join(interval_means, on=['interval_start', 'CellID'])

    # Global offsets per band (using chosen operator over entire dataset)
    global_offsets = {
        band: getattr(work[band], agg_fun_vodoffset)()
        for band in band_ids
        if band in work.columns
    }

    # Form anomalies
    for band in band_ids:
        mean_col = f"{band}_mean"
        if band not in work.columns or mean_col not in work.columns:
            continue
        # (raw - interval_cell_mean + global_offset)
        work[f"{band}_anom{_suffix}"] = (
            work[band] - work[mean_col] + global_offsets[band]
        )

    # Temporal aggregation to requested resolution
    # Keep only anomaly columns
    anom_cols = [f"{band}_anom{_suffix}" for band in band_ids if f"{band}_anom{_suffix}" in work.columns]
    if not anom_cols:
        return pd.DataFrame()

    ts = (
        work[anom_cols]
        .groupby(pd.Grouper(freq=f"{temporal_resolution}min", level='Epoch'))
        .agg(agg_fun_ts)
    )

    # Drop empty rows
    ts = ts.dropna(how='all')

    return ts

def calculate_biomass_binned_anomalies(vod, vod_avg, band_ids, con=None, biomass_bins=5, **kwargs):
    """
    Calculate VOD anomalies for cells grouped by biomass (VOD) bins and filtered by constellation.
    Bins go from low to high biomass

    Parameters
    ----------
    vod : pandas.DataFrame
        VOD data with cell IDs
    vod_avg : pandas.DataFrame
        Average VOD values per grid cell
    band_ids : list
        List of band identifiers to calculate anomalies for
    temporal_resolution : int
        Temporal resolution in minutes
    con : list or None, default=None
        List of constellation names to include (e.g., ['GPS', 'Galileo'])
    biomass_bins : int, default=5
        Number of biomass bins to create
    **kwargs : dict
        Additional keyword arguments

    Returns
    -------
    pandas.DataFrame
        Time series of VOD anomalies for each biomass bin with Epoch as index
    """
    pd.options.mode.chained_assignment = None
    agg_fun_ts = kwargs.get('agg_fun_ts', 'mean')
    suffix = kwargs.get('suffix', '')
    plot_bins = kwargs.get('plot_bins', False)  # whether to plot the bin histograms for visualization
    temporal_resolution = kwargs.get('temporal_resolution', 30)
    
    if suffix and not suffix.startswith('_'):
        suffix = f"_{suffix}"
    
    # Constellation mapping
    constellation_ident = {
        'G': 'GPS',
        'R': 'GLONASS',
        'E': 'Galileo',
        'C': 'BeiDou',
        'S': 'SBAS'
    }
    
    # Filter by constellation if specified
    if con is not None:
        con_prefixes = []
        for c in con:
            for prefix, name in constellation_ident.items():
                if name.upper() == c.upper():
                    con_prefixes.append(prefix)
        
        if not con_prefixes:
            raise ValueError(f"No valid constellations found in {con}")
        
        # Filter VOD data
        sv_filter = vod.index.get_level_values('SV').str[0].isin(con_prefixes)
        vod_filtered = vod[sv_filter]
        
        # Print constellation filtering info
        selected_cons = [constellation_ident[prefix] for prefix in sorted(set(con_prefixes))]
        print(f"Filtering data for constellations: {', '.join(selected_cons)}")
        print(
            f"Selected {len(vod_filtered)} out of {len(vod)} observations ({len(vod_filtered) / len(vod) * 100:.1f}%)")
    else:
        vod_filtered = vod

    
    # Initialize results DataFrame
    combined_results = pd.DataFrame()
    
    # Process each band
    for band in band_ids:
        # Create biomass bins based on VOD averages for the reference band
        band_means = vod_avg[f"{band}_mean"]
        
        # Create bins based on VOD percentiles
        try:
            bins = pd.qcut(band_means, biomass_bins, labels=False, duplicates='drop')
        except ValueError:
            actual_bins = min(biomass_bins, len(band_means.unique()))
            bins = pd.cut(band_means, actual_bins, labels=False)
            print(f"Warning: Not enough unique values for {biomass_bins} bins. Using {actual_bins} bins instead.")
        
        # Create a mapping from CellID to bin
        cell_to_bin = pd.Series(bins.values, index=band_means.index, name='biomass_bin')
        
        # Add bin information to VOD data
        vod_with_bins = vod_filtered.join(cell_to_bin, on='CellID')
        
        # Adding the sky sector mean (does not need to be bin-wise sky sectors fall into bins automatically)
        vod_with_means = vod_with_bins.join(vod_avg[[f"{band}_mean"]], on='CellID')
        
        bins = {}
        # -----------------------------------
        # Calculate anomalies for each bin
        for bin_num in range(biomass_bins):
            # Filter data for this bin
            bin_data = vod_with_means[vod_with_means['biomass_bin'] == bin_num]
            bin_mean = bin_data[band].mean()
            
            # Subtracts long-term mean of sky-sector from the binned VOD values
            # mute SettingWithCopyWarning
            bin_data.loc[:, f"{band}_anom"] = bin_data[band] - bin_data[f"{band}_mean"]
                
            # populate the bins dict with the anomalies
            bins[bin_num] = pd.DataFrame({
                'Epoch': bin_data.index.get_level_values('Epoch'),
                f"{band}_anom": bin_data[f"{band}_anom"],
                f"{band}": bin_data[band],
            })
            
            # adding bin-internal mean to the anomaly data
            bin_data.loc[:, f"{band}_anom"] = bin_data[f"{band}_anom"] + bin_mean
                
            # Temporal aggregation
            bin_ts = bin_data.groupby(pd.Grouper(freq=f"{temporal_resolution}min", level='Epoch'))[f"{band}_anom"].agg(
                agg_fun_ts)
            
            # Add to results with appropriate column name
            combined_results[f"{band}_anom_bin{bin_num}{suffix}"] = bin_ts
            
        # After processing individual bins, add a combined high-biomass bin (3-5)
        # Collect dataframes from bins 3-5 (higher biomass)
        high_biomass_bins = pd.concat([bins[i] for i in range(3, min(5 + 1, biomass_bins))])
        high_biomass_mean = high_biomass_bins[band].mean()
        high_biomass_bins[f"{band}_anom_mean"] = high_biomass_bins[f"{band}_anom"] + high_biomass_mean
        
        # Temporal aggregation for the combined high biomass bins
        high_biomass_ts = high_biomass_bins[f"{band}_anom_mean"].groupby(
            pd.Grouper(freq=f"{temporal_resolution}min", level='Epoch')
        ).agg(agg_fun_ts)
        
        # Add to results with appropriate column name
        combined_results[f"{band}_anom_bin3-5{suffix}"] = high_biomass_ts
        
        if plot_bins:
            # Calculate actual bin edges for visualization
            if isinstance(bins, pd.Series):
                # Get the unique bin values and their corresponding data
                bin_groups = {}
                for bin_num in range(biomass_bins):
                    bin_groups[bin_num] = band_means[bins == bin_num]
                
                # Calculate min and max for each bin to use as edges
                bin_edges = [bin_groups[i].min() for i in sorted(bin_groups.keys())]
                bin_edges.append(bin_groups[max(bin_groups.keys())].max())
                
                from matplotlib import pyplot as plt
                plt.figure(figsize=(5, 3))
                band_means.hist(bins=30, alpha=0.5, label='VOD Means')
                for edge in bin_edges:
                    plt.axvline(edge, color='red', linestyle='--')
                plt.axvline(bin_edges[0], color='red', linestyle='--', label='Bin Edges')  # Add label only once
                plt.title(f"VOD Means Histogram with Bins for {band}")
                plt.xlabel('VOD Mean Values')
                plt.ylabel('Frequency')
                plt.legend()
                plt.show()

    return combined_results


def _drop_clearsky_cells(band_ids, vod_df, threshold):
    vod_clearsky = vod_df.copy()
    _vod_avg = (
        vod_df
        .groupby('CellID')[band_ids]
        .mean()
        .add_suffix('_mean')
    )
    
    # flag all cellid with vod_avg < threshold as clear-sky cells
    clear_sky_cells = _vod_avg[(_vod_avg < threshold).any(axis=1)].index.tolist()
    # remove clear-sky cells from vod
    vod_clearsky = vod_clearsky[~vod_clearsky["CellID"].isin(clear_sky_cells)]
    del _vod_avg
    return vod_clearsky


def drop_outlier_sats(vod_df, rmv_svs):
    # GPS: None
    # GLONASS: R13
    # GALILEO: E06, E29
    vod_no_outliers = vod_df.copy()
    if rmv_svs:
        vod_no_outliers = vod_no_outliers[~vod_no_outliers.index.get_level_values('SV').isin(rmv_svs)]
    return vod_no_outliers