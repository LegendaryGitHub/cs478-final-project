import numpy as np
import pandas as pd # https://www.geeksforgeeks.org/python/how-to-calculate-moving-averages-in-python/
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html


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

# Subplot 1 - Raw
subplots[0].plot(samples, apRaw[0:1000],  color='steelblue', linewidth=1.0, label='AP')
subplots[0].plot(samples, staRaw[0:1000], color='crimson',   linewidth=1.0, label='STA')
subplots[0].set_title('Raw CSI Data')
subplots[0].set_ylabel('Amplitude')
subplots[0].legend(loc='upper right')
subplots[0].grid(True, alpha=0.3)

# Subplot 2 - Moving Average
subplots[1].plot(samples, apMA[0:1000],  color='steelblue', linewidth=1.0, label='AP')
subplots[1].plot(samples, staMA[0:1000], color='crimson',   linewidth=1.0, label='STA')
subplots[1].set_title(f'Moving Average (window={MA_WINDOW})')
subplots[1].set_ylabel('Amplitude')
subplots[1].legend(loc='upper right')
subplots[1].grid(True, alpha=0.3)

# Subplot 3 - Savitzky-Golay
subplots[2].plot(samples, apSG[0:1000],  color='steelblue', linewidth=1.0, label='AP')
subplots[2].plot(samples, staSG[0:1000], color='crimson',   linewidth=1.0, label='STA')
subplots[2].set_title(f'Savitzky-Golay (window={SG_WINDOW}, polyorder={SG_POLYORDER})')
subplots[2].set_ylabel('Amplitude')
subplots[2].legend(loc='upper right')
subplots[2].grid(True, alpha=0.3)
subplots[2].set_xlabel('Sample Index')

plt.tight_layout()
plt.savefig('task2-csi.png', dpi=150, bbox_inches='tight')



# -------- Part 2: Pearson coefficients (Michael) ----------------------------------------
