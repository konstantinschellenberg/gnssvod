#!/usr/bin/env python
# -*- coding: utf-8 -*-
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class VodConfig:
    """Explicit configuration for process_vod (replaces kwargs)."""
    local_file: bool = False
    overwrite: bool = False
    n_workers: int = 15  # dask workers used while concatenating files

@dataclass(frozen=True)
class AnomalyConfig:
    # spatio-temporal
    angular_resolution: int
    angular_cutoff: int
    temporal_resolution: int
    
    # options/flow
    make_ke: bool = False
    overwrite: bool = False
    show: bool = False
    
    # ANOMALY DETECTION
    # all
    drop_clearsky: bool = False
    drop_clearsky_threshold: float = 0.1
    drop_outliersats: bool = False
    drop_dips: bool = False
    drop_dips_threshold: float = 0.95
    drop_dips_loessfrac: float = 0.1  # loess window for dip detection smoothing
    
    # 1) Humphrey approach
    agg_fun_ts: str = "median"
    
    # 2) Konstantin approach
    agg_fun_vodoffset: str = "median"
    agg_fun_satincell: str = "median"
    ks_strategy: str = "sv"  # or "sv", "con" for satellite vs constellation-based
    
    # 3) Alex approach
    anom_ak_timedelta: pd.Timedelta = pd.Timedelta(days=1)
    
    # Misc calculations
    calculate_biomass_bins: bool = False
    constellations: list = None  # e.g., ['GPS', 'GALILEO']
    

# Variable label lookup (used in plotting)
# Keys are method codes (parsed from pattern: VODx_anom_<code>)
VAR_LABELS = {
    "vh": "VOD (vh)",
    "ak": "VOD (ak)",
    "ks": "VOD (ks, all filt.)",
    "ksak": "VOD (ksak)",
    "anom": "VOD (anom)",  # fallback for generic anomalies
    "ks_con": "VOD (ks, con)",
    "ks_clearsky": "VOD (ks, clear)",
    "ks_nooutliers": "VOD (ks, noout)",
    "ks_backup": "VOD (ks, before)"
}
