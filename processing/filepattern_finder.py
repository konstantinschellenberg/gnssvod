import datetime
from pathlib import Path
from typing import Tuple, Dict, List


def create_time_filter_patterns(time_interval: Tuple[str, str], station: str = "MOz") -> Dict[str, str]:
    """
    Create precise glob patterns to filter files within a given time interval.

    Parameters:
    ----------
    time_interval : Tuple[str, str]
        Start and end dates in format "YYYY-MM-DD"
    station : str, optional
        Station name prefix for NC files (default: "MOz")

    Returns:
    -------
    Dict[str, str]
        Dictionary with 'nc' and 'obs' keys containing glob patterns compatible with get_filelist
    """
    # Parse the time interval
    start_date = datetime.datetime.strptime(time_interval[0], "%Y-%m-%d")
    end_date = datetime.datetime.strptime(time_interval[1], "%Y-%m-%d")
    
    # For NC files (gather files)
    # Format: MOz_20240509080000_20240509090000.nc
    # Create a pattern that will match all files with timestamps in the range
    nc_pattern = f"{station}_*.nc"
    
    # For OBS files
    # Format: SEPT281v.23.obs
    # Create a pattern that matches SEPT files with any DOY and the correct year(s)
    start_year = start_date.strftime("%y")
    end_year = end_date.strftime("%y")
    
    if start_year == end_year:
        obs_pattern = f"SEPT*.{start_year}.obs"
    else:
        # If spanning multiple years, we'll need to do post-filtering
        obs_pattern = "SEPT*.??.obs"
    
    return {
        'nc': nc_pattern,
        'obs': obs_pattern
    }


def filter_files_by_date(files: List[str], time_interval: Tuple[str, str]) -> List[str]:
    """
    Filter a list of files to only include those within a specific date range.

    Parameters:
    ----------
    files : List[str]
        List of file paths to filter
    time_interval : Tuple[str, str]
        Start and end dates in format "YYYY-MM-DD"

    Returns:
    -------
    List[str]
        Filtered list of file paths
    """
    # Parse the time interval
    start_date = datetime.datetime.strptime(time_interval[0], "%Y-%m-%d")
    end_date = datetime.datetime.strptime(time_interval[1], "%Y-%m-%d")
    # Add one day to include end date fully
    end_date = end_date + datetime.timedelta(days=1)
    
    filtered_files = []
    
    for file_path in files:
        file_name = Path(file_path).name
        
        # Handle NC files (MOz_20240509080000_20240509090000.nc)
        if file_name.endswith('.nc'):
            # Extract the timestamp from the filename
            try:
                # Extract date parts from filename (assumes format like MOz_20240509080000_...)
                date_part = file_name.split('_')[1]
                if len(date_part) >= 8:  # At least YYYYMMDD
                    file_date = datetime.datetime.strptime(date_part[:8], "%Y%m%d")
                    if start_date <= file_date < end_date:
                        filtered_files.append(file_path)
            except (IndexError, ValueError):
                # If we can't parse the date, skip this file
                continue
        
        # Handle OBS files (SEPT281v.23.obs)
        elif file_name.endswith('.obs') and file_name.startswith('SEPT'):
            try:
                # Extract DOY (3 digits) and year (2 digits) from filename
                doy_part = file_name[4:7]  # 3-digit DOY
                year_part = file_name.split('.')[1]  # 2-digit year
                
                # Construct date from year and DOY
                file_year = 2000 + int(year_part)  # Assuming years are in the 2000s
                file_date = datetime.datetime(file_year, 1, 1) + datetime.timedelta(days=int(doy_part) - 1)
                
                if start_date <= file_date < end_date:
                    filtered_files.append(file_path)
            except (IndexError, ValueError):
                # If we can't parse the date, skip this file
                continue
    
    return filtered_files