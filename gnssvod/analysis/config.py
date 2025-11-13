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
    # 1) Humphrey approach
    agg_fun_ts: str = "median"
    
    # 2) Konstantin approach
    agg_fun_vodoffset: str = "median"
    agg_fun_satincell: str = "median"
    eval_num_obs_tps: bool = True
    
    # 3) Alex approach
    anom_ak_timedelta: pd.Timedelta = pd.Timedelta(days=1)
    
    # Misc calculations
    calculate_biomass_bins: bool = False
    constellations: list = None  # e.g., ['GPS', 'GALILEO']
