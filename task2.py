import numpy as np
import pandas as pd # https://www.geeksforgeeks.org/python/how-to-calculate-moving-averages-in-python/
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html


# -------- Part 1: CSI data visualization ----------------------------------------
SUBCARRIER   = 0   # Select the first subcarrier for visualization
MA_WINDOW    = 20  # Moving average window size
SG_WINDOW    = 21  # Savitzky-Golay filter window size (must be odd)
SG_POLYORDER = 3   # Savitzky-Golay filter polynomial order

# load data
apData  = pd.read_csv(AP_FILE,  header=None).values.astype(float)
staData = pd.read_csv(STA_FILE, header=None).values.astype(float)

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

# TODO: Plot the raw, moving-averaged, and SavGol-smoothed CSI data

