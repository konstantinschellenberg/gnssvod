#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import subprocess
import shutil
import concurrent.futures
from pathlib import Path
from typing import Union, Optional
from tqdm import tqdm


def bin2rin(search_dir: Union[str, Path], driver: str = "convbin", out_dir: Optional[Union[str, Path]] = None,
            overwrite: bool = False, num_workers: int = 15) -> None:
    """
    Recursively finds all BINEX files (.bnx) and converts them to RINEX format
    using the convbin utility. Deletes any .nav files created during conversion.
    Uses multiprocessing for faster processing.

    Parameters
    ----------
    driver: str
        The driver to use for conversion. Currently 'convbin' and 'teqc' are supported.
    search_dir : str or Path
        Directory to recursively search for BINEX files
    out_dir : str or Path, optional
        Directory where to save RINEX files. If None, files will be saved
        in the same location as input files.
    overwrite : bool, optional
        Whether to overwrite existing .obs files (default: False)
    num_workers : int, optional
        Number of parallel worker processes (default: CPU count)

    Notes
    -----
    This function requires the convbin utility from RTKLIB to be installed and
    available in the system PATH.
    """
    
    # Define worker function to process a single file
    def process_file(args):
        root, file = args
        file_path = Path(root) / file
        
        # Determine the expected output file path
        base_name = file_path.stem
        target_dir = out_dir if out_dir else Path(root)
        output_obs_path = target_dir / f"{base_name}.obs"
        
        # Skip if output file already exists and overwrite is False
        if output_obs_path.exists() and not overwrite:
            return {"status": "skipped", "file": file_path}
        
        # -----------------------------------
        # CONVBIN DRIVER (RTKLIB)
        # -----------------------------------
        
        if driver == "convbin":
            # Build the convbin command
            cmd = ["convbin", "-r", "binex", "-os"]
            
            # Add output directory if provided
            if out_dir is not None:
                cmd.extend(["-d", str(out_dir)])
            
            # Add the input file
            cmd.append(str(file_path))
            
            try:
                # Run the command
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=True
                )
                return {"status": "processed", "file": file_path}
            except subprocess.CalledProcessError as e:
                return {
                    "status": "error",
                    "file": file_path,
                    "error": str(e),
                    "stdout": e.stdout,
                    "stderr": e.stderr
                }
            except Exception as e:
                return {"status": "error", "file": file_path, "error": str(e)}
            
        # -----------------------------------
        # TEQC DRIVER
        # -----------------------------------
        elif driver == "teqc":
            # Build the teqc command without redirection
            cmd = ["teqc", "-binex", str(file_path)]
            
            try:
                # Run the command and capture stdout
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=True
                )
                
                # Write the captured stdout to the output file
                with open(output_obs_path, 'w') as f:
                    f.write(result.stdout)
                
                return {"status": "processed", "file": file_path}
            except subprocess.CalledProcessError as e:
                return {
                    "status": "error",
                    "file": file_path,
                    "error": str(e),
                    "stdout": e.stdout,
                    "stderr": e.stderr
                }
            except Exception as e:
                return {"status": "error", "file": file_path, "error": str(e)}
        else:
            raise NotImplementedError(
                f"Driver '{driver}' is not supported. Use 'convbin' or 'teqc'."
            )
    
    # Convert input to Path object
    search_dir = Path(search_dir)
    
    # Convert output dir to Path if provided
    if out_dir is not None:
        out_dir = Path(out_dir)
        os.makedirs(out_dir, exist_ok=True)
    
    # Track statistics
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    # Check if convbin is available
    if driver == "convbin" and shutil.which("convbin") is None:
        raise RuntimeError(
            "convbin utility not found. Please install RTKLIB and ensure "
            "convbin is in your PATH."
        )
    elif driver == "teqc" and shutil.which("teqc") is None:
        raise RuntimeError(
            "teqc utility not found. Please install teqc and ensure "
            "teqc is in your PATH."
        )

    # Find all BINEX files first for the progress bar
    binex_files = []
    for root, _, files in os.walk(search_dir):
        for file in files:
            if file.lower().endswith('.bnx') or file.lower().endswith('.binex'):
                binex_files.append((root, file))
    
    # Process files using parallel workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks and create a future -> task mapping
        future_to_file = {
            executor.submit(process_file, args): args
            for args in binex_files
        }
        
        # Process results as they complete with a progress bar
        for future in tqdm(
                concurrent.futures.as_completed(future_to_file),
                total=len(binex_files),
                desc="Converting BINEX to RINEX"
        ):
            args = future_to_file[future]
            try:
                result = future.result()
                if result["status"] == "processed":
                    processed_count += 1
                elif result["status"] == "skipped":
                    skipped_count += 1
                    print(f"Skipping {result['file']} - output file already exists")
                else:  # error
                    error_count += 1
                    print(f"Error converting {result['file']}: {result['error']}")
                    if "stdout" in result:
                        print(f"Command output: {result['stdout']}")
                    if "stderr" in result:
                        print(f"Command error: {result['stderr']}")
            except Exception as e:
                error_count += 1
                print(f"Error processing {args[1]}: {e}")
    
    # Delete all .nav files in the output directory
    nav_deleted_count = 0
    target_dir = out_dir if out_dir else search_dir
    
    # Find all navigation files first
    nav_files = []
    for nav_pattern in ["*.nav", "*.gnav", "*.hnav", "*.qnav", "*.lnav", "*.cnav", "*.inav"]:
        nav_files.extend(list(target_dir.glob(nav_pattern)))
    
    # Delete navigation files with progress bar
    for nav_file in tqdm(nav_files, desc="Deleting navigation files"):
        try:
            nav_file.unlink()
            nav_deleted_count += 1
        except Exception as e:
            print(f"Error deleting {nav_file}: {e}")
    
    print(
        f"Processed {processed_count} files, skipped {skipped_count} existing files, encountered {error_count} errors")
    print(f"Deleted {nav_deleted_count} navigation files")