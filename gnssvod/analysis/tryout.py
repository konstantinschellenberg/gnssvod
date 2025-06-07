#!/usr/bin/env python
# -*- coding: utf-8 -*-

# !/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import dask.array as da
import dask.dataframe as dd
import xarray as xr
import time
import matplotlib.pyplot as plt
# At the top of your file, add:
import dask
from multiprocessing import cpu_count


# Set the number of threads for parallel computation
def set_dask_threads(n_threads=None):
    """Set the number of threads for Dask to use."""
    if n_threads is None:
        n_threads = cpu_count()
    
    print(f"Setting Dask to use {n_threads} threads")
    dask.config.set(num_workers=n_threads)
    dask.config.set(scheduler='threads')  # 'threads' for IO-bound, 'processes' for CPU-bound


def setup_dask_cluster(n_workers=None, threads_per_worker=1):
    """Set up a local Dask distributed cluster."""
    from dask.distributed import Client, LocalCluster
    
    if n_workers is None:
        n_workers = cpu_count()
    
    print(f"Setting up Dask cluster with {n_workers} workers, {threads_per_worker} threads per worker")
    cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker)
    client = Client(cluster)
    print(f"Dashboard link: {client.dashboard_link}")
    return client


def dask_arithmetic():
    print("Testing arithmetic operations on chunked multidimensional data")
    
    # Create sample data
    print("\n1. Creating test data...")
    # Generate a large random dataset
    size = 10000000  # 10 million rows
    chunk_size = 1000000  # 1 million rows per chunk
    
    # Create pandas DataFrame
    print("Creating pandas DataFrame...")
    start = time.time()
    df = pd.DataFrame({
        'A': np.random.randn(size),
        'B': np.random.randn(size),
        'C': np.random.randn(size),
        'key': np.random.choice(['x', 'y', 'z'], size)
    })
    print(f"Pandas creation time: {time.time() - start:.2f} seconds")
    
    # Create Dask DataFrame
    print("Converting to Dask DataFrame...")
    start = time.time()
    ddf = dd.from_pandas(df, chunksize=chunk_size)
    print(f"Dask conversion time: {time.time() - start:.2f} seconds")
    
    # Create multidimensional data with xarray + dask
    print("Creating multidimensional data with xarray...")
    start = time.time()
    times = pd.date_range('2023-01-01', periods=10000, freq='H')
    lats = np.linspace(0, 90, 100)
    lons = np.linspace(-180, 180, 180)
    
    # Random data array with dims (time, lat, lon)
    data = np.random.rand(len(times), len(lats), len(lons))
    
    # Create xarray Dataset with dask chunks
    ds = xr.Dataset(
        data_vars={
            'temperature': (['time', 'lat', 'lon'], data)
        },
        coords={
            'time': times,
            'lat': lats,
            'lon': lons
        }
    )
    ds = ds.chunk({'time': 100, 'lat': 25, 'lon': 45})
    print(f"Xarray+Dask creation time: {time.time() - start:.2f} seconds")
    
    # 2. Basic arithmetic operations on Dask DataFrame
    print("\n2. Testing basic arithmetic on Dask DataFrame...")
    start = time.time()
    result1 = ddf['A'] + ddf['B']
    result2 = ddf['A'] * ddf['C']
    result3 = ddf['A'] / (ddf['B'] + 1e-5)  # Avoid division by zero
    
    # Force computation to measure time
    result1.compute()
    result2.compute()
    result3.compute()
    print(f"Basic DataFrame arithmetic time: {time.time() - start:.2f} seconds")
    
    # 3. Grouped operations
    print("\n3. Testing grouped operations...")
    start = time.time()
    grouped_result = ddf.groupby('key').mean().compute()
    print(f"Grouped mean computation time: {time.time() - start:.2f} seconds")
    print(grouped_result)
    
    # 4. Operations on multidimensional data
    print("\n4. Testing operations on multidimensional data...")
    start = time.time()
    # Calculate mean temperature over time
    temp_mean = ds.temperature.mean(dim='time').compute()
    print(f"Mean over time dimension: {time.time() - start:.2f} seconds")
    
    start = time.time()
    # Calculate daily temperature anomalies
    daily_mean = ds.temperature.mean(dim=['lat', 'lon'])
    anomalies = (ds.temperature - daily_mean).compute()
    print(f"Anomaly calculation time: {time.time() - start:.2f} seconds")
    
    # 5. Compare with non-chunked operations
    print("\n5. Comparing chunked vs non-chunked (small sample)...")
    # Use smaller sample for this comparison
    small_df = df.iloc[:100000].copy()
    small_ddf = dd.from_pandas(small_df, chunksize=10000)
    
    # Pandas operation
    start = time.time()
    pd_result = small_df['A'] + small_df['B'] * small_df['C']
    pd_time = time.time() - start
    print(f"Pandas computation time: {pd_time:.4f} seconds")
    
    # Dask operation
    start = time.time()
    dask_result = small_ddf['A'] + small_ddf['B'] * small_ddf['C']
    dask_result = dask_result.compute()
    dask_time = time.time() - start
    print(f"Dask computation time: {dask_time:.4f} seconds")
    
    # Verify results match
    np.testing.assert_allclose(pd_result.values, dask_result.values)
    print("Results match between Pandas and Dask!")
    
    print("\nAll tests completed successfully!")


if __name__ == "__main__":
    # Option 1: Simple threading
    set_dask_threads(4)  # Use 4 threads
    
    # OR Option 2: Distributed (more features, dashboard)
    # client = setup_dask_cluster(n_workers=4, threads_per_worker=2)
    
    # Run your tests
    dask_arithmetic()
    
    # If using distributed, you can close the client
