import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
import mne
import pandas as pd
from yasa import Hypnogram

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
project_destination = 'C:\\Users\\cyber\\OneDrive\\Documents\\1college\\softDes\\coding\\final_proj'

raw1 = mne.io.read_raw_edf(f'{project_destination}/SC4001E0-PSG.edf', exclude=('Resp oro-nasal', 'Temp rectal','Event marker'))
raw1.set_channel_types({'EOG horizontal':'eog','EMG submental':'emg'})

raw2 = mne.io.read_raw_edf(f'{project_destination}/SC4002E0-PSG.edf', exclude=('Resp oro-nasal', 'Temp rectal','Event marker'))
raw2.set_channel_types({'EOG horizontal':'eog','EMG submental':'emg'})

raw3 = mne.io.read_raw_edf(f'{project_destination}/SC4011E0-PSG.edf', exclude=('Resp oro-nasal', 'Temp rectal','Event marker'))
raw3.set_channel_types({'EOG horizontal':'eog','EMG submental':'emg'})

raw4 = mne.io.read_raw_edf(f'{project_destination}/SC4012E0-PSG.edf', exclude=('Resp oro-nasal', 'Temp rectal','Event marker'))
raw4.set_channel_types({'EOG horizontal':'eog','EMG submental':'emg'})

raw5 = mne.io.read_raw_edf(f'{project_destination}/SC4022E0-PSG.edf', exclude=('Resp oro-nasal', 'Temp rectal','Event marker'))
raw5.set_channel_types({'EOG horizontal':'eog','EMG submental':'emg'})

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

ann5 = mne.read_annotations(f'{project_destination}/SC4022EJ-Hypnogram.edf')
raw5.set_annotations(ann5)

## For the test data
test_ann1 = mne.read_annotations(f'{project_destination}/SC4021EH-Hypnogram.edf')
test_raw1.set_annotations(test_ann1)

raw = mne.concatenate_raws([raw1, raw2, raw3, raw4, raw5]) # Data used to train the model
raw.resample(100.0, npad='auto')  # lower sampling rate before epoching to speed processing

# Epoch object has an (X,Y,Z) numpy matrix
# X: number of epochs (each epoch is 30 seconds as specified below)
# Y: number of channels (the EDF data has 4 channels)
# Z: data points in the 30 second intervals
epochs = mne.make_fixed_length_epochs(raw, duration=30.0, preload=True)

psd = epochs.compute_psd(fmin=0.5, fmax=30, picks=['eeg', 'eog', 'emg'])

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
}

X = epochs.get_data()
Y = np.array([class_map.get(i,i) for i in labels])

print(X.shape)
print(Y.shape)

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


# ------------------------------------------HYPNOGRAM CREATION-------------------------------------
# replace annotations (strings) with integers for yasa.from_integers function
#true_hypnogram = np.array([class_map.get(i,i) for i in test_ann1.description])

# replace unknown sleep stages with
#true_hypnogram = [-2 if x == 'Sleep stage ?' else x for x in true_hypnogram]

print(test_y_pred.shape)
print(test_y.shape)
#print(len(true_hypnogram))

# plot correct hypnogram
hyp2 = Hypnogram.from_integers(test_y) # or use test_y_pred to see model predicted hypnogram
hyp2.plot_hypnogram()
plt.show()

# ------------------------------------------CONFUSION MATRIX-------------------------------------------
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(test_y, test_y_pred)
classes = np.unique(np.concatenate([test_y, test_y_pred]))

stage_names = {
    0: 'W',
    1: 'N1',
    2: 'N2',
    3: 'N3',
    4: 'REM'
}
labels = [stage_names[c] for c in classes]

fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap='Blues', ax=ax, xticks_rotation=45)
plt.show()
