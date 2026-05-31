import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

KEY_LENGTH       = 128
ERROR_THRESHOLD  = 0.1
SAMPLE_COUNT     = 20000
SUBCARRIER_COUNT = 64
WINDOW_SIZE      = 20

# Load CSI data
apData  = pd.read_csv('ap.csv', header=None).values.astype(float)
staData = pd.read_csv('sta.csv', header=None).values.astype(float)

# Extract amplitude columns 
apAmp  = apData[:,  :SUBCARRIER_COUNT]
staAmp = staData[:, :SUBCARRIER_COUNT]

# Smoothing using moving average
apDataframe  = pd.DataFrame(apAmp)
staDataframe = pd.DataFrame(staAmp)

apWindows  = apDataframe.rolling(window=WINDOW_SIZE, min_periods=1, center=True)
staWindows = staDataframe.rolling(window=WINDOW_SIZE, min_periods=1, center=True)

apMovingAverages  = apWindows.mean()
staMovingAverages = staWindows.mean()

apSmooth  = apMovingAverages.to_numpy()
staSmooth = staMovingAverages.to_numpy()

# Quantization
apThreshold  = np.mean(apSmooth)
staThreshold = np.mean(staSmooth)

apBinary  = np.where(apSmooth > apThreshold, 1, -1)
apBits    = apBinary.flatten()

staBinary = np.where(staSmooth > staThreshold, 1, -1)
staBits   = staBinary.flatten()

print(f"AP  bit mean: {apBits.mean():.4f}")
print(f"STA bit mean: {staBits.mean():.4f}")

# Key splitting
keys = len(apBits) // KEY_LENGTH

apBitsTrimmed = apBits[:keys * KEY_LENGTH]
apKeys        = apBitsTrimmed.reshape(keys, KEY_LENGTH)

staBitsTrimmed = staBits[:keys * KEY_LENGTH]
staKeys        = staBitsTrimmed.reshape(keys, KEY_LENGTH)

# Key Generation Rate (KGR) 
bitMismatches = (apKeys != staKeys)
mismatchRates = np.mean(bitMismatches, axis=1)

successFlags = (mismatchRates < ERROR_THRESHOLD)
successes    = int(np.sum(successFlags))
kgr          = successes / keys

print(f"Key Generation Rate:   {kgr:.4f}  ({kgr*100:.2f}%)")
print(f"Avg bit mismatch rate: {np.mean(mismatchRates):.4f}  ({np.mean(mismatchRates)*100:.2f}%)")

# Visualization
fig, subplots = plt.subplots(1, 2, figsize=(13, 5))

# Subplot 1 - Bit mismatch distribution
subplots[0].hist(mismatchRates, bins=50, color='steelblue', edgecolor='white', linewidth=0.5)
subplots[0].axvline(ERROR_THRESHOLD, color='crimson', linestyle='--', linewidth=1.8,label=f'{int(ERROR_THRESHOLD*100)}% threshold')
subplots[0].set_xlabel('Bit Mismatch Rate per Key')
subplots[0].set_ylabel('Number of Keys')
subplots[0].set_title('Bit Mismatch Rate Distribution')
subplots[0].legend()

# Subplot 2 - Success / Failure bar chart
bars = subplots[1].bar(
    ['Successful\nKeys', 'Failed\nKeys'],
    [successes, keys - successes],
    color=['steelblue', 'salmon'],
    edgecolor='white', width=0.5
)
subplots[1].set_ylabel('Number of Keys')
subplots[1].set_title(f'Key Generation Rate: {kgr*100:.2f}%')

# Annotate successful bar
subplots[1].text(bars[0].get_x() + bars[0].get_width() / 2,
                 bars[0].get_height() + keys * 0.005,
                 f'{successes:,}', ha='center', fontsize=11, fontweight='bold')

# Annotate failed bar
subplots[1].text(bars[1].get_x() + bars[1].get_width() / 2,
                 bars[1].get_height() + keys * 0.005,
                 f'{keys - successes:,}', ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('task3-results.png', dpi=150, bbox_inches='tight')


