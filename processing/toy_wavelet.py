#!/usr/bin/env python
# -*- coding: utf-8 -*-
from processing.inspect_vod_funs import analyze_wavelets, plot_vod_fingerprint
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

from definitions import FIG

FIG = FIG / 'toy_wavelet'
FIG.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":

    freq = '10min'
    # Create datetime index with 10-minute frequency for one year
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 1, 1)
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    n_points = len(date_range)
    
    # calculate points per day by freq
    minutes = pd.Timedelta(freq).total_seconds() / 60
    points_per_day = int(24 * 60 / minutes)  # e.g., 144 points for 10-min frequency
    
    # 1. Daily sinus curve (24-hour cycle)
    daily_freq = 2 * np.pi / points_per_day  # 144 points per day (24h × 6 per hour)
    daily_sine = 0.5 * np.sin(daily_freq * np.arange(n_points))
    
    # 2. Seasonal trend (annual cycle)
    seasonal_freq = 2 * np.pi / n_points
    # Make it slightly asymmetric for a more natural pattern
    x = np.arange(n_points)
    seasonal_trend = 1.2 * np.sin(seasonal_freq * x + 0.3 * np.sin(seasonal_freq * x))
    
    # 3. High-frequency white noise
    high_freq_noise = 0.1 * np.random.normal(0, 1, n_points)
    
    # 4. Low-frequency noise
    raw_noise = np.random.normal(0, 1, n_points)
    window_size = 144  # 24 hours in 10-min intervals
    low_freq_noise = pd.Series(raw_noise).rolling(window=window_size, center=True).mean().fillna(0).values
    low_freq_noise = 0.3 * low_freq_noise / np.std(low_freq_noise)  # Normalize and scale
    
    # 5. Random events (e.g., precipitation)
    random_events = np.zeros(n_points)
    # Add ~20 random spike events throughout the year
    event_indices = np.random.choice(range(n_points), size=20, replace=False)
    for idx in event_indices:
        # Create a spike that decays over ~1 day
        decay_length = np.random.randint(100, 200)
        event_magnitude = np.random.uniform(0.5, 1.5)
        for i in range(min(decay_length, n_points - idx)):
            random_events[idx + i] = event_magnitude * np.exp(-i / 30)
    
    # Combine all components
    combined_signal = daily_sine + seasonal_trend + high_freq_noise + low_freq_noise + random_events
    
    # Create the final dataset
    df = pd.DataFrame({
        'daily_sine': daily_sine,
        'seasonal_trend': seasonal_trend,
        'high_freq_noise': high_freq_noise,
        'low_freq_noise': low_freq_noise,
        'random_events': random_events,
        'combined_signal': combined_signal
    }, index=date_range)
    
    # name index "datetime"
    df.index.name = 'datetime'
    
    # Display basic info
    print(f"Dataset spans from {df.index.min()} to {df.index.max()}")
    print(f"Total data points: {len(df)}")
    print(df.head())
    
    # -----------------------------------
    # plotting
    
    
    make_plot = False
    if make_plot:
        from statsmodels.graphics.tsaplots import plot_acf
        
        # 1. Time series plot of all components
        plt.figure(figsize=(8, 8))
        components = ['daily_sine', 'seasonal_trend', 'high_freq_noise',
                      'low_freq_noise', 'random_events', 'combined_signal']
        
        # Plot full year
        for i, comp in enumerate(components, 1):
            plt.subplot(len(components), 1, i)
            plt.plot(df.index, df[comp], linewidth=1)
            plt.ylabel(comp)
            plt.title(f"{comp} - Full Year")
            if i < len(components):  # Only show x-axis for the bottom plot
                plt.tick_params(labelbottom=False)
        
        plt.tight_layout()
        plt.savefig(FIG / 'timeseries_full_year.png', dpi=300)
        
        # Plot just one month for better visibility of daily patterns
        one_month = df['2023-01-01':'2023-02-01']
        plt.figure(figsize=(8, 8))
        for i, comp in enumerate(components, 1):
            plt.subplot(len(components), 1, i)
            plt.plot(one_month.index, one_month[comp], linewidth=1)
            plt.ylabel(comp)
            plt.title(f"{comp} - January 2023")
            if i < len(components):
                plt.tick_params(labelbottom=False)
        
        plt.tight_layout()
        plt.savefig(FIG / 'timeseries_one_month.png', dpi=300)
        
        # 2. Mean diurnal cycle (average daily pattern)
        plt.figure(figsize=(8, 8))
        # Extract hour and minute for grouping
        df['hour'] = df.index.hour
        df['minute'] = df.index.minute
        # Create a time-of-day identifier (0 to 143 for 10-minute intervals)
        df['time_of_day'] = df['hour'] * 6 + df['minute'] // 10
        
        # Calculate mean diurnal cycle for each component
        diurnal_means = df.groupby('time_of_day')[components].mean()
        diurnal_std = df.groupby('time_of_day')[components].std()
        
        # Create x-axis labels for hours
        x_ticks = np.arange(0, 144, 6)  # Every hour
        x_labels = [f"{h:02d}:00" for h in range(0, 24)]
        
        for i, comp in enumerate(components, 1):
            plt.subplot(3, 2, i)
            plt.plot(diurnal_means.index, diurnal_means[comp], 'b-', linewidth=2)
            plt.fill_between(diurnal_means.index,
                             diurnal_means[comp] - diurnal_std[comp],
                             diurnal_means[comp] + diurnal_std[comp],
                             alpha=0.2)
            plt.title(f"Mean Diurnal Cycle: {comp}")
            plt.xticks(x_ticks, x_labels, rotation=45)
            plt.grid(True, alpha=0.3)
            plt.xlim(0, 143)
        
        plt.tight_layout()
        plt.savefig(FIG / 'diurnal_cycles.png', dpi=300)
        
        # 3. Autocorrelation function
        # plt.figure(figsize=(15, 10))
        # max_lags = 1440  # 10 days (10 days * 144 points per day)
        #
        # for i, comp in enumerate(components, 1):
        #     plt.subplot(3, 2, i)
        #     plot_acf(df[comp].values, lags=max_lags, title=f"ACF: {comp}")
        #     # Add vertical lines at daily intervals
        #     for day in range(1, 11):
        #         plt.axvline(x=day * 144, color='r', linestyle='--', alpha=0.3)
        #     plt.grid(True, alpha=0.3)
        #
        # plt.tight_layout()
        # plt.savefig('autocorrelation.png', dpi=300)
        #
        # # Show plots
        plt.show()
    
    # -----------------------------------
    # Wavelet analysis
    
    # analyze_wavelets(df, 'daily_sine', save_dir=FIG, smoothing=False)
    # analyze_wavelets(df, 'seasonal_trend', save_dir=FIG, smoothing=False)
    # analyze_wavelets(df, 'high_freq_noise', save_dir=FIG, smoothing=False)
    # analyze_wavelets(df, 'low_freq_noise', save_dir=FIG, smoothing=False)
    # analyze_wavelets(df, 'random_events', save_dir=FIG, smoothing=False)
    analyze_wavelets(df, 'combined_signal', save_dir=FIG, smoothing=False)
    
    # -----------------------------------
    # fingerprint
    plot_vod_fingerprint(df, 'combined_signal', title="Synthetic data – 5 components", save_dir=FIG)
    plot_vod_fingerprint(df, 'daily_sine', title="Daily Sine Wave", save_dir=FIG)
    plot_vod_fingerprint(df, 'seasonal_trend', title="Seasonal Trend", save_dir=FIG)
    plot_vod_fingerprint(df, 'high_freq_noise', title="High-Frequency Noise", save_dir=FIG)
    plot_vod_fingerprint(df, 'low_freq_noise', title="Low-Frequency Noise", save_dir=FIG)
    plot_vod_fingerprint(df, 'random_events', title="Random Events", save_dir=FIG)
