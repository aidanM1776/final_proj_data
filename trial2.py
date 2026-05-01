import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
import os
import mne
import pandas as pd

## RON DID THESE IMPORTS 
from math import sqrt

import lightgbm as lgb

from lightgbm import LGBMClassifier, LGBMRegressor
import warnings
warnings.filterwarnings('ignore')
import catboost as cb
from catboost import CatBoostRegressor, CatBoostClassifier

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, StackingRegressor, StackingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, ElasticNetCV, Ridge, Lasso
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, classification_report, r2_score, mean_squared_error, roc_curve
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from tqdm import tqdm

from mlxtend.feature_selection import SequentialFeatureSelector as SFS

try:
    import xgboost as xgb
    print("XGBoost imported successfully!")
except ImportError:
    print("XGBoost not installed. Installing...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"])
    import xgboost as xgb
    print("XGBoost installed and imported successfully!")

from xgboost import XGBClassifier, XGBRegressor
## END RON IMPORTS


# Create raw object from PSG edf data
# raw has diff channels, number of time points, time points, and duration of the data
# ch_names, n_times, times, duration
# Two EEG channels that will be our main data to determine sleep stages
# One EOG and one EMG useful for identifying REM sleep
# --------------------------------------------------EXCLUDE---------------------------------------------------------------
# Resp oro-nasal is used to identify sleep apneas and hypopneas
# temp rectal is body temp useful for identifying sleep quality and sleep onset (wake to sleep)
# Event markers is specific to how each study was ran I think

## Created a variable for the project destination since it will be different for everyone
project_destination = 'C:\\Users\\ronsh\\Downloads\\datascience_workspace_folder'

raw1 = mne.io.read_raw_edf(f'{project_destination}/SC4001E0-PSG.edf', exclude=('Resp oro-nasal', 'Temp rectal','Event marker'))
raw1.set_channel_types({'EOG horizontal':'eog','EMG submental':'emg'})

raw2 = mne.io.read_raw_edf(f'{project_destination}/SC4002E0-PSG.edf', exclude=('Resp oro-nasal', 'Temp rectal','Event marker'))
raw2.set_channel_types({'EOG horizontal':'eog','EMG submental':'emg'})

raw3 = mne.io.read_raw_edf(f'{project_destination}/SC4011E0-PSG.edf', exclude=('Resp oro-nasal', 'Temp rectal','Event marker'))
raw3.set_channel_types({'EOG horizontal':'eog','EMG submental':'emg'})

raw4 = mne.io.read_raw_edf(f'{project_destination}/SC4012E0-PSG.edf', exclude=('Resp oro-nasal', 'Temp rectal','Event marker'))
raw4.set_channel_types({'EOG horizontal':'eog','EMG submental':'emg'})

## Testing data not used for training the model, just to see how it performs on new data
test_raw1 = mne.io.read_raw_edf(f'{project_destination}/SC4021E0-PSG.edf', exclude=('Resp oro-nasal', 'Temp rectal','Event marker'))
test_raw1.set_channel_types({'EOG horizontal':'eog','EMG submental':'emg'})

# the hypnogram.edf file only has annotations, must add to raw data
ann1 = mne.read_annotations(f'{project_destination}/SC4001EC-Hypnogram.edf')
raw1.set_annotations(ann1)

ann2 = mne.read_annotations(f'{project_destination}/SC4002EC-Hypnogram.edf')
raw2.set_annotations(ann2)

ann3 = mne.read_annotations(f'{project_destination}/SC4011EH-Hypnogram.edf') # make sure that after SC4011 it's H NOT C !!
raw3.set_annotations(ann3)

ann4 = mne.read_annotations(f'{project_destination}/SC4012EC-Hypnogram.edf')
raw4.set_annotations(ann4)

## For the test data
test_ann1 = mne.read_annotations(f'{project_destination}/SC4021EH-Hypnogram.edf')
test_raw1.set_annotations(test_ann1)


raw = mne.concatenate_raws([raw1, raw2, raw3, raw4]) # Data used to train the model
raw.resample(100.0, npad='auto')  # lower sampling rate before epoching to speed processing

# Epoch object has an (X,Y,Z) numpy matrix
# X: number of epochs (each epoch is 30 seconds as specified below)
# Y: number of channels (the EDF data has 4 channels)
# Z: data points in the 30 second intervals
epochs = mne.make_fixed_length_epochs(raw, duration=30.0, preload=False)

psd = epochs.compute_psd(fmin=0.5, fmax=30, picks=['eeg', 'eog', 'emg'])


## OLD WAY OF GETTING LABELS
# # get labels and match length to epochs
# labels = raw.annotations.description
# n = min(len(epochs), len(labels))
# epochs = epochs[:n]
# labels = labels[:n]

##  NEW WAY THAT MATCHES EPOCHS TO ANNOTATIONS
def stage_at_time(t, ann):
    idx = np.where((ann.onset <= t) & (t < ann.onset + ann.duration))[0]
    return ann.description[idx[0]] if len(idx) else None

epoch_midpoints = np.arange(len(epochs)) * 30.0 + 15.0
labels = [stage_at_time(t, raw.annotations) for t in epoch_midpoints]
##

# encoding hypnogram labels
class_map = {
    'Sleep stage W': 0,
    'Sleep stage 1': 1,
    'Sleep stage 2': 2,
    'Sleep stage 3': 3,
    'Sleep stage 4': 3,
    'Sleep stage R': 4,
    #'Sleep stage ?': 5, # this needs to be added for unknown sleep stages ??
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
#plt.show() # don't need this and it stops the rest of the code from running


## XGBoost Classifier
X_data = epochs.get_data()
epoch_midpoints = np.arange(len(epochs)) * 30.0 + 15.0
epoch_labels = [stage_at_time(t, raw.annotations) for t in epoch_midpoints]

valid_mask = [lbl in class_map for lbl in epoch_labels]
X_data = X_data[valid_mask]
y = np.array([class_map[lbl] for lbl in epoch_labels if lbl in class_map], dtype=int)
times = epoch_midpoints[valid_mask]

# Statistics for each channel in each epoch
feature_rows = []
for idx, epoch in enumerate(X_data):
    row = {'time': times[idx]}
    for ch_idx, ch_name in enumerate(epochs.ch_names):
        data = epoch[ch_idx]
        prefix = ch_name.replace(' ', '_').replace('-', '_')
        row[f'{prefix}_mean'] = data.mean()
        row[f'{prefix}_std'] = data.std()
        row[f'{prefix}_min'] = data.min()
        row[f'{prefix}_max'] = data.max()
    feature_rows.append(row)

features_df = pd.DataFrame(feature_rows)
TRAINING_FEATURE_COLUMNS = features_df.drop(columns=['time']).columns.tolist()

# Train-test split and XGBoost classifier
X_train, X_test, y_train, y_test = train_test_split(features_df.drop(columns=['time']), y, test_size=0.5, random_state=42, stratify=y)
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'XGBoost Classifier Accuracy: {accuracy:.4f}')
print(classification_report(y_test, y_pred))

## Cross-validation with XGBoost and StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
cv_model.fit(features_df.drop(columns=['time']), y)
scores = cross_val_score(cv_model, features_df.drop(columns=['time']), y, cv=cv, scoring='accuracy')
print("CV scores: ", scores)
print("Mean CV accuracy: ", scores.mean())

# Testing model on new data set (test_raw1 and test_ann1)
test_epochs = mne.make_fixed_length_epochs(test_raw1, duration=30.0, preload=True)
test_psd = test_epochs.compute_psd(fmin=0.5, fmax=30, picks=['eeg', 'eog', 'emg'])
test_epoch_midpoints = np.arange(len(test_epochs)) * 30.0 + 15.0
test_labels = [stage_at_time(t, test_raw1.annotations) for t in test_epoch_midpoints]

test_valid_mask = [lbl in class_map for lbl in test_labels]
test_X_data = test_epochs.get_data()[test_valid_mask]
test_y = np.array([class_map[lbl] for lbl in test_labels if lbl in class_map], dtype=int)
test_times = test_epoch_midpoints[test_valid_mask]
test_feature_rows = []
for idx, epoch in enumerate(test_X_data):
    row = {'time': test_times[idx]}
    for ch_idx, ch_name in enumerate(test_epochs.ch_names):
        data = epoch[ch_idx]
        prefix = ch_name.replace(' ', '_').replace('-', '_')
        row[f'{prefix}_mean'] = data.mean()
        row[f'{prefix}_std'] = data.std()
        row[f'{prefix}_min'] = data.min()
        row[f'{prefix}_max'] = data.max()
    test_feature_rows.append(row)

test_features_df = pd.DataFrame(test_feature_rows)
test_y_pred = cv_model.predict(test_features_df.drop(columns=['time']))
test_accuracy = accuracy_score(test_y, test_y_pred)
print(f'Test Accuracy: {test_accuracy:.4f}')
print(classification_report(test_y, test_y_pred))


# def _match_features_to_training_columns(features_df):
#     features_df = features_df.copy()
#     if 'time' in features_df.columns:
#         features_df = features_df.drop(columns=['time'])

#     missing_cols = [col for col in TRAINING_FEATURE_COLUMNS if col not in features_df.columns]
#     if missing_cols:
#         print(f"Warning: missing feature columns from new PSG file, filling zeros for: {missing_cols}")
#         for col in missing_cols:
#             features_df[col] = 0.0

#     return features_df[TRAINING_FEATURE_COLUMNS]


# def _safe_set_channel_types(raw):
#     map_candidates = {}
#     for ch in raw.ch_names:
#         upper = ch.upper()
#         if 'EOG' in upper:
#             map_candidates[ch] = 'eog'
#         elif 'EMG' in upper:
#             map_candidates[ch] = 'emg'
#     if map_candidates:
#         raw.set_channel_types(map_candidates)


# def extract_epoch_features_from_raw(raw, epoch_duration=30.0):
#     """Extract per-epoch feature rows from a raw PSG object."""
#     if raw.annotations is None or len(raw.annotations) == 0:
#         raise ValueError('Raw PSG object has no annotations.')

#     epochs = mne.make_fixed_length_epochs(raw, duration=epoch_duration, preload=False)

#     try:
#         epoch_data = epochs.get_data()
#     except Exception as e:
#         if 'bad epochs have not been dropped' in str(e):
#             print('Retrying epoch extraction with preload=True because current Epochs length is unknown.')
#             epochs = mne.make_fixed_length_epochs(raw, duration=epoch_duration, preload=True)
#             epoch_data = epochs.get_data()
#         else:
#             raise

#     epoch_count = epoch_data.shape[0]
#     epoch_midpoints = np.arange(epoch_count) * epoch_duration + epoch_duration / 2
#     labels = [stage_at_time(t, raw.annotations) for t in epoch_midpoints]
#     valid_mask = [lbl in class_map for lbl in labels]

#     if not any(valid_mask):
#         return pd.DataFrame(columns=['time'] + TRAINING_FEATURE_COLUMNS)

#     feature_rows = []
#     for idx, epoch in enumerate(epoch_data[valid_mask]):
#         row = {'time': epoch_midpoints[valid_mask][idx]}
#         for ch_idx, ch_name in enumerate(epochs.ch_names):
#             data = epoch[ch_idx]
#             prefix = ch_name.replace(' ', '_').replace('-', '_')
#             row[f'{prefix}_mean'] = data.mean()
#             row[f'{prefix}_std'] = data.std()
#             row[f'{prefix}_min'] = data.min()
#             row[f'{prefix}_max'] = data.max()
#         feature_rows.append(row)

#     features_df = pd.DataFrame(feature_rows)
#     return _match_features_to_training_columns(features_df)


# def predict_sleep_stages(raw, model=cv_model, epoch_duration=30.0):
#     """Predict sleep stage labels for every 30-second epoch in a raw PSG recording."""
#     features_df = extract_epoch_features_from_raw(raw, epoch_duration=epoch_duration)
#     if features_df.empty:
#         return np.array([], dtype=int)
#     return model.predict(features_df)


# def predict_sleep_stages_from_files(psg_path, hypnogram_path, model=cv_model, epoch_duration=30.0, exclude_channels=('Resp oro-nasal', 'Temp rectal', 'Event marker')):
#     """Load a PSG EDF and hypnogram EDF, then predict sleep stages."""
#     raw = mne.io.read_raw_edf(psg_path, exclude=exclude_channels)
#     _safe_set_channel_types(raw)
#     ann = mne.read_annotations(hypnogram_path)
#     raw.set_annotations(ann)
#     return predict_sleep_stages(raw, model=model, epoch_duration=epoch_duration)


# # Example usage:
# # new_psg_path = 'C:/path/to/your/PSG.edf'
# # new_hypnogram_path = 'C:/path/to/your/Hypnogram.edf'
# # sleep_stage_array = predict_sleep_stages_from_files(new_psg_path, new_hypnogram_path)
# # print(sleep_stage_array)

# new_psg_path = f'{project_destination}/SC4021E0-PSG.edf'
# new_hypnogram_path = f'{project_destination}/SC4021EH-Hypnogram.edf'
# if os.path.exists(new_psg_path) and os.path.exists(new_hypnogram_path):
#     try:
#         sleep_stage_array = predict_sleep_stages_from_files(new_psg_path, new_hypnogram_path)
#         print(sleep_stage_array)
#     except Exception as e:
#         print(f'Could not predict sleep stages for example file: {e}')
# else:
#     print('Example PSG or hypnogram file not found, skipping example prediction.')
