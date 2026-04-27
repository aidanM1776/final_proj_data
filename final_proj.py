import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
import mne
import pandas as pd

# Create raw object from PSG edf data
# raw has diff channels, number of time points, time points, and duration of the data
# ch_names, n_times, times, duration
# Two EEG channels that will be our main data to determine sleep stages
# One EOG and one EMG useful for identifying REM sleep
# --------------------------------------------------EXCLUDE---------------------------------------------------------------
# Resp oro-nasal is used to identify sleep apneas and hypopneas
# temp rectal is body temp useful for identifying sleep quality and sleep onset (wake to sleep)
# Event markers is specific to how each study was ran I think
raw1 = mne.io.read_raw_edf('final_proj/SC4001E0-PSG.edf', exclude=('Resp oro-nasal', 'Temp rectal','Event marker'))
raw1.set_channel_types({'EOG horizontal':'eog','EMG submental':'emg'})

raw2 = mne.io.read_raw_edf('final_proj/SC4002E0-PSG.edf', exclude=('Resp oro-nasal', 'Temp rectal','Event marker'))
raw2.set_channel_types({'EOG horizontal':'eog','EMG submental':'emg'})

raw3 = mne.io.read_raw_edf('final_proj/SC4011E0-PSG.edf', exclude=('Resp oro-nasal', 'Temp rectal','Event marker'))
raw3.set_channel_types({'EOG horizontal':'eog','EMG submental':'emg'})

# the hypnogram.edf file only has annotations, must add to raw data
ann1 = mne.read_annotations('final_proj/SC4001EC-Hypnogram.edf')
raw1.set_annotations(ann1)

ann2 = mne.read_annotations('final_proj/SC4002EC-Hypnogram.edf')
raw2.set_annotations(ann2)

ann3 = mne.read_annotations('final_proj/SC4011EH-Hypnogram.edf')
raw3.set_annotations(ann3)

raw = mne.concatenate_raws([raw1, raw2, raw3])

# Epoch object has an (X,Y,Z) numpy matrix
# X: number of epochs (each epoch is 30 seconds as specified below)
# Y: number of channels (the EDF data has 4 channels)
# Z: data points in the 30 second intervals
epochs = mne.make_fixed_length_epochs(raw, duration=30.0, preload=True)

psd = epochs.compute_psd(fmin=0.5, fmax=30, picks=['eeg', 'eog', 'emg'])

# get labels and match length to epochs
labels = raw.annotations.description
n = min(len(epochs), len(labels))
epochs = epochs[:n]
labels = labels[:n]

# encoding hypnogram labels
class_map = {
    'Sleep stage W': 0,
    'Sleep stage 1': 1,
    'Sleep stage 2': 2,
    'Sleep stage 3': 3,
    'Sleep stage 4': 3,
    'Sleep stage R': 4,
}

X = epochs.get_data()
Y = np.array([class_map.get(i,i) for i in labels])

print(X.shape)
print(Y.shape)

fig, (ax1, ax2) = plt.subplots(2, 1)
#epochs.plot(n_epochs=436, n_channels=4)
#psd.plot(spatial_colors=True, dB=True)

# Plot of first epoch (30 seconds) for 4 channels over time
data = X[0]
times = epochs.times
ax1.plot(times, data.T)
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Amplitude (µV)')
ax1.set_title('First Epoch')
ax1.legend(epochs.ch_names)


# Plotting psd of first epoch
psd_data, freqs = psd.get_data(picks=['eeg', 'eog', 'emg'],return_freqs=True)
psd_db = 10 * np.log10(psd_data[0])
ax2.plot(freqs, psd_db.T)
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Amplitude')
ax2.set_title('First Epoch')
ax2.legend(psd.ch_names)

plt.tight_layout()
plt.show()
