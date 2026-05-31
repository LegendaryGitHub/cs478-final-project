# Author: Michael Wilde
# This script performs the following:
# 1. Loads in the AP and STA CSI data from CSV files.
# 2. Extracts the first N samples for a specific channel.
# 3. Defines a fixed AP window and slides a STA window across the data.
# 4. Calculates the Pearson correlation coefficient for each lag.
# 5. Plots the correlation coefficients against the lags and identifies the peak correlation.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Load in the csv files
print("Loading data...")
ap_data_full = pd.read_csv('ap.csv', header=None)
sta_data_full = pd.read_csv('sta.csv', header=None)

# Setup parameters for analysis
N = 20000
w = 5000
# Sample the data by picking one of the available 64 channels
channel_idx = 15

# Extract the first N samples for the chosen channel
ap_data = ap_data_full.iloc[:N, channel_idx].values
sta_data = sta_data_full.iloc[:N, channel_idx].values

# Define the fixed AP windows (centered in the middle of the N samples)
ap_start = (N // 2) - (w // 2) # Should be 7500\
ap_end = (N // 2) + (w // 2) # Should be 12500
ap_window = ap_data[ap_start:ap_end] 

# Slide the STA windows and calculate correlations
lags = range(-(N // 2 - w // 2), (N // 2 - w // 2) + 1)
correlations = []

print("Working on the calculating sliding windows in Pearson correlations...")
for lag in lags:
    # Shift the STA window by the lag amount
    sta_start = ap_start + lag
    sta_end = ap_end + lag
    sta_window = sta_data[sta_start:sta_end]

    # Calculate Pearson correlation coefficient
    coef, _ = pearsonr(ap_window, sta_window)
    correlations.append(coef)

# Plot the graph
plt.figure(figsize=(10,6))
plt.plot(lags, correlations, color='blue')
plt.title('Task 1: Pearson Correlation vs. Time Lag')
plt.xlabel('Lag (samples)')
plt.ylabel('Pearson Correlation Coefficient')
plt.grid(True)
plt.show()

# Find and print the peak correlation for the written comments
max_corr = max(correlations)
best_lag = lags[correlations.index(max_corr)]
print(f"Peak correlation is {max_corr:.4f} at lag {best_lag}")