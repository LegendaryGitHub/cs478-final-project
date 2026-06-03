import numpy as np
import pandas as pd # https://www.geeksforgeeks.org/python/how-to-calculate-moving-averages-in-python/
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html
from scipy.stats import pearsonr # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html

# -------- Part 1: CSI data visualization (Nathaniel) ----------------------------------------
SUBCARRIER   = 6   # Select the first subcarrier for visualization
MA_WINDOW    = 20  # Moving average window size
SG_WINDOW    = 21  # Savitzky-Golay filter window size (must be odd)
SG_POLYORDER = 3   # Savitzky-Golay filter polynomial order

# load data
apData  = pd.read_csv('ap.csv',  header=None).values.astype(float)
staData = pd.read_csv('sta.csv', header=None).values.astype(float)

# Extract values for the selected subcarrier
apRaw = apData[:, SUBCARRIER]
staRaw = staData[:, SUBCARRIER]

# smoothing
apSeries = pd.Series(apRaw)
apWindows = apSeries.rolling(window=MA_WINDOW, min_periods=1, center=True)
apMovingAverages = apWindows.mean()
apMA = apMovingAverages.to_numpy()

staSeries = pd.Series(staRaw)
staWindows = staSeries.rolling(window=MA_WINDOW, min_periods=1, center=True)
staMovingAverages = staWindows.mean()
staMA = staMovingAverages.to_numpy()

# SavGol smoothing
apSG  = savgol_filter(apRaw, window_length=SG_WINDOW, polyorder=SG_POLYORDER, mode='interp')
staSG = savgol_filter(staRaw, window_length=SG_WINDOW, polyorder=SG_POLYORDER, mode='interp')

# Plot the raw, moving-averaged, and SavGol-smoothed CSI data
samples = np.arange(0, 1000)

fig, subplots = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
fig.suptitle(f'Task 2 – CSI Amplitude Visualization (Subcarrier {SUBCARRIER})', fontsize=13, fontweight='bold')

y_min = min(apRaw[0:1000].min(), staRaw[0:1000].min())
y_max = max(apRaw[0:1000].max(), staRaw[0:1000].max())

# Subplot 1 - Raw
subplots[0].plot(samples, apRaw[0:1000],  color='steelblue', linewidth=1.0, label='AP')
subplots[0].plot(samples, staRaw[0:1000], color='crimson',   linewidth=1.0, label='STA')
subplots[0].set_title('Raw CSI Data')
subplots[0].set_ylabel('Amplitude')
subplots[0].legend(loc='upper right')
subplots[0].grid(True, alpha=0.3)
subplots[0].set_ylim(y_min - 2, y_max + 2)

# Subplot 2 - Moving Average
subplots[1].plot(samples, apMA[0:1000],  color='steelblue', linewidth=1.0, label='AP')
subplots[1].plot(samples, staMA[0:1000], color='crimson',   linewidth=1.0, label='STA')
subplots[1].set_title(f'Moving Average (window={MA_WINDOW})')
subplots[1].set_ylabel('Amplitude')
subplots[1].legend(loc='upper right')
subplots[1].grid(True, alpha=0.3)
subplots[1].set_ylim(y_min - 2, y_max + 2)

# Subplot 3 - Savitzky-Golay
subplots[2].plot(samples, apSG[0:1000],  color='steelblue', linewidth=1.0, label='AP')
subplots[2].plot(samples, staSG[0:1000], color='crimson',   linewidth=1.0, label='STA')
subplots[2].set_title(f'Savitzky-Golay (window={SG_WINDOW}, polyorder={SG_POLYORDER})')
subplots[2].set_ylabel('Amplitude')
subplots[2].legend(loc='upper right')
subplots[2].grid(True, alpha=0.3)
subplots[2].set_xlabel('Sample Index')
subplots[2].set_ylim(y_min - 2, y_max + 2)

plt.tight_layout()
plt.savefig('task2-csi.png', dpi=150, bbox_inches='tight')



# -------- Part 2: Pearson coefficients (Michael) ----------------------------------------
# Setup parameters for analysis
N = 20000
w = 5000

# Define the fixed AP windows for both smoothed datasets (centered in the middle of the N samples)
ap_start = (N // 2) - (w // 2) # Should be 7500
ap_end = (N // 2) + (w // 2) # Should be 12500

ap_window_ma = apMA[ap_start:ap_end]
ap_window_sg = apSG[ap_start:ap_end]

# Slide the STA windows and calculate correlations for both smoothed datasets
lags = range(-(N // 2 - w // 2), (N // 2 - w // 2) + 1)
correlations_ma = []
correlations_sg = []

print("Calculating sliding windows in Pearson correlations...")
for lag in lags:
    # Shift the STA window by the lag amount
    sta_start = ap_start + lag
    sta_end = ap_end + lag
    
    sta_window_ma = staMA[sta_start:sta_end]
    sta_window_sg = staSG[sta_start:sta_end]

    # Calculate Pearson correlation coefficient for moving average
    coef_ma, _ = pearsonr(ap_window_ma, sta_window_ma)
    correlations_ma.append(coef_ma)

    # Calculate Pearson correlation coefficient for Savitzky-Golay
    coef_sg, _ = pearsonr(ap_window_sg, sta_window_sg)
    correlations_sg.append(coef_sg)

# Plot the graph for the Pearson coefficients
plt.figure(figsize=(10,6))
plt.plot(lags, correlations_ma, color='darkorange', linewidth=1.5, label=f'Moving Average (window={MA_WINDOW})')
plt.plot(lags, correlations_sg, color='forestgreen', linewidth=1.5, label=f'Savitzky-Golay (window={SG_WINDOW})')
plt.title(f'Task 2: Smoothed Pearson Correlation vs. Time Lag (Subcarrier {SUBCARRIER})')
plt.xlabel('Lag (samples)')
plt.ylabel('Pearson Correlation Coefficient')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.5)
plt.tight_layout()
plt.savefig('task2-correlations.png', dpi=150, bbox_inches='tight')
plt.show()

# Find and print the peak correlation for the written comments
max_corr_ma = max(correlations_ma)
best_lag_ma = lags[correlations_ma.index(max_corr_ma)]
max_corr_sg = max(correlations_sg)
best_lag_sg = lags[correlations_sg.index(max_corr_sg)]
print(f"Peak Pearson Correlation (Moving Average): {max_corr_ma:.4f} at lag {best_lag_ma} samples")
print(f"Peak Pearson Correlation (Savitzky-Golay): {max_corr_sg:.4f} at lag {best_lag_sg} samples")