#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from matplotlib import pyplot as plt, rcParams
import pandas as pd

# add gnssvod to path
import sys
sys.path.append(str(Path(__file__).resolve().parent / "gnssvod"))

rcParams['figure.figsize'] = 7, 5
rcParams['figure.autolayout'] = True
rcParams['lines.markersize'] = .5
rcParams['figure.dpi'] = 300

plt.style.use('seaborn-v0_8-whitegrid')
pd.set_option('display.max_columns', None)

def get_repo_root() -> Path:
    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        if (parent / '.git').exists():
            return parent
    raise FileNotFoundError("Repository root with .git directory not found")

def get_relative_path_to_root(current_file):
    current_path = Path(current_file).resolve()
    root_path = get_repo_root()
    relative_path = current_path.relative_to(root_path)
    return relative_path

# -----------------------------------
# directories
ROOT = get_repo_root()  # get the root path of the project
FIG = ROOT / "figures"
DATA = ROOT / "data"
ZIP = ROOT / 'zip_archive'
AUX = DATA / 'orbit'
TEST = DATA / "test"
ARD = DATA / 'ard'
TERRALIVE = Path("/home/konsch/Documents/5-Repos/terralive")
ENVDATA = TERRALIVE / 'data_output' / 'tb_merged'


GROUND = 'subcanopy/MOz1_Grnd'
TOWER = 'tower/MOz2_Twr'