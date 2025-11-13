#!/usr/bin/env python
# -*- coding: utf-8 -*-

def create_time_intervals(start_date, end_date, subset_months=6):
    """
    Create a list of time intervals between start_date and end_date.
    The function tries to align intervals with calendar periods when possible.

    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        subset_months (int): Size of each interval in months

    Returns:
        list: List of tuples with start and end dates for each interval
    """
    import pandas as pd
    from dateutil.relativedelta import relativedelta
    
    # Convert to datetime objects
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    intervals = []
    current_start = start
    
    while current_start < end:
        # Calculate the end of this interval
        current_end = current_start + relativedelta(months=subset_months) - relativedelta(days=1)
        
        # If the interval would go beyond the overall end date, use the overall end date
        if current_end > end:
            current_end = end
        
        # Format dates as strings and add to intervals
        interval_start = current_start.strftime('%Y-%m-%d')
        interval_end = current_end.strftime('%Y-%m-%d')
        intervals.append((interval_start, interval_end))
        
        # Move to the next interval
        current_start = current_end + relativedelta(days=1)
    
    return intervals

def print_color(text, color='green'):
    """
    Print text in specified color in the terminal.

    Args:
        text (str): Text to print
        color (str): Color name (e.g., 'red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'white')
    """
    color_codes = {
        'black': '\033[30m',
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'reset': '\033[0m',
    }
    
    color_code = color_codes.get(color.lower(), color_codes['reset'])
    reset_code = color_codes['reset']
    
    print(f"{color_code}{text}{reset_code}")
    
def check_instance(param, param_name, expected_type):
    if isinstance(param, list):
        for p in param:
            if not isinstance(p, expected_type):
                raise ValueError(f"All elements in {param_name} must be of type {expected_type.__name__}")
        return param
    else:
        if not isinstance(param, expected_type):
            raise ValueError(f"{param_name} must be of type {expected_type.__name__}")
        return param