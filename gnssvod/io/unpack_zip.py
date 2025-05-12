#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import gzip
import re
import shutil
from pathlib import Path
from typing import Union


def unpack_gz_files(search_dir: Union[str, Path], out_dir: Union[str, Path]) -> None:
    """
    Recursively finds all files ending with 'o.gz', unpacks them and saves them
    in the output directory organized by year based on the file name pattern.

    Parameters
    ----------
    search_dir : str or Path
        Directory to recursively search for .o.gz files
    out_dir : str or Path
        Directory where unpacked files will be saved, organized by year

    Notes
    -----
    The function extracts the year from the filename using the pattern "\d{2}o.gz"
    (two digits followed by 'o.gz'). Files are saved in YYYY subdirectories.
    """
    # Convert inputs to Path objects
    search_dir = Path(search_dir)
    out_dir = Path(out_dir)
    
    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)
    
    # Pattern to match files ending with o.gz and extract year
    pattern = re.compile(r'(\d{2})o\.gz$')
    
    # Track statistics
    processed_count = 0
    skipped_count = 0
    
    # Walk through all directories recursively
    for root, _, files in os.walk(search_dir):
        for file in files:
            # Check if file matches the pattern
            match = pattern.search(file)
            if match:
                file_path = Path(root) / file
                
                # Extract year (assuming 20XX for two digit year)
                year_suffix = match.group(1)
                # Convert to full year (assuming 2000's)
                year = f"20{year_suffix}"
                
                # Create year directory if needed
                year_dir = out_dir / year
                os.makedirs(year_dir, exist_ok=True)
                
                # Create output file path
                output_file = year_dir / file.replace('.gz', '')
                
                # Skip if output file already exists
                if output_file.exists():
                    skipped_count += 1
                    continue
                
                try:
                    # Decompress the file
                    with gzip.open(file_path, 'rb') as f_in:
                        with open(output_file, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    processed_count += 1
                except Exception as e:
                    print(f"Error unpacking {file_path}: {e}")
    
    print(f"Processed {processed_count} files, skipped {skipped_count} existing files")

