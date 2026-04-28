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

# the hypnogram.edf file only has annotations, must add to raw data
ann1 = mne.read_annotations(f'{project_destination}/SC4001EC-Hypnogram.edf')
raw1.set_annotations(ann1)

ann2 = mne.read_annotations(f'{project_destination}/SC4002EC-Hypnogram.edf')
raw2.set_annotations(ann2)

ann3 = mne.read_annotations(f'{project_destination}/SC4011EH-Hypnogram.edf') # make sure that after SC4011 it's H NOT C !!
raw3.set_annotations(ann3)

raw = mne.concatenate_raws([raw1, raw2, raw3])

# Epoch object has an (X,Y,Z) numpy matrix
# X: number of epochs (each epoch is 30 seconds as specified below)
# Y: number of channels (the EDF data has 4 channels)
# Z: data points in the 30 second intervals
epochs = mne.make_fixed_length_epochs(raw, duration=30.0, preload=True)

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

# Train-test split and XGBoost classifier
X_train, X_test, y_train, y_test = train_test_split(features_df.drop(columns=['time']), y, test_size=0.3, random_state=42, stratify=y)
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'XGBoost Classifier Accuracy: {accuracy:.4f}')
print(classification_report(y_test, y_pred))

## Cross-validation with XGBoost and StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
scores = cross_val_score(model, features_df.drop(columns=['time']), y, cv=cv, scoring='accuracy')
print("CV scores: ", scores)
print("Mean CV accuracy: ", scores.mean())


# ------------------------------------------HYPNOGRAM CREATION-------------------------------------

raw_test = mne.io.read_raw_edf(f'{project_destination}/SC4012E0-PSG.edf', exclude=('Resp oro-nasal', 'Temp rectal','Event marker'))
raw_test.set_channel_types({'EOG horizontal':'eog','EMG submental':'emg'})

# the hypnogram.edf file only has annotations, must add to raw data
ann_test = mne.read_annotations(f'{project_destination}/SC4012EC-Hypnogram.edf')

# replace annotations (strings) with integers for yasa.from_integers function
true_hypnogram = np.array([class_map.get(i,i) for i in ann_test.description])

# replace unknown sleep stages with
true_hypnogram = [-2 if x == 'Sleep stage ?' else x for x in true_hypnogram]

# plot correct hypnogram
hyp2 = Hypnogram.from_integers(true_hypnogram)
hyp2.plot_hypnogram()
plt.show()

epoch_test = mne.make_fixed_length_epochs(raw_test, duration=30.0, preload=True)
