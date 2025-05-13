#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from matplotlib import pyplot as plt, rcParams

rcParams['figure.figsize'] = 7, 5
rcParams['figure.dpi'] = 300
plt.style.use('seaborn-v0_8-whitegrid')

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
AUX = DATA / 'orbit'
TEST = DATA / "test"
ZIP = ROOT / 'zip_archive'

GROUND = 'subcanopy/MOz1_Grnd'
TOWER = 'tower/MOz2_Twr'