#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from definitions import get_repo_root

def setup_logger(
        name=None,
        level=logging.INFO,
        log_file=None,
        log_dir=None,
        file_level=None,
        console=True,
        console_level=None,
        format_string=None
):
    """
    Configure and return a logger for the GNSS VOD project.

    Parameters
    ----------
    name : str, optional
        Name of the logger. If None, returns root logger.
    level : int, optional
        Master logging level (applied to both handlers if specific levels not provided)
    log_file : str, optional
        Name of log file. If None and log_dir provided, a timestamped file is created.
    log_dir : str or Path, optional
        Directory to store log files. Will be created if it doesn't exist.
    file_level : int, optional
        Specific logging level for file handler. Defaults to level if not provided.
    console : bool, optional
        Whether to output logs to console.
    console_level : int, optional
        Specific logging level for console handler. Defaults to level if not provided.
    format_string : str, optional
        Custom format string for log messages.

    Returns
    -------
    logging.Logger
        Configured logger object
    """
    logger = logging.getLogger(name)
    
    # Clear any existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()
    
    logger.setLevel(level)
    
    # Default format if not specified
    if format_string is None:
        format_string = "[%(asctime)s] %(levelname)s - %(name)s - %(message)s"
    formatter = logging.Formatter(format_string)
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(console_level if console_level is not None else level)
        logger.addHandler(console_handler)
    
    # File handler
    if log_dir is not None:
        log_dir = Path(log_dir)
        os.makedirs(log_dir, exist_ok=True)
        
        if log_file is None:
            # Create a timestamped log file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"gnssvod_{timestamp}.log"
        
        file_path = log_dir / log_file
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(file_level if file_level is not None else level)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name=None, **kwargs):
    """
    Get a logger with the specified name. Use this throughout the project
    for consistent logging.

    Parameters
    ----------
    name : str, optional
        Name of the logger, typically __name__ in each module
    **kwargs : dict
        Additional arguments passed to setup_logger

    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    # Default log directory to the logs folder in the repo root
    if 'log_dir' not in kwargs:
        kwargs['log_dir'] = get_repo_root() / "logs"
    
    return setup_logger(name, **kwargs)


# Global default logger
logger = get_logger("gnssvod")