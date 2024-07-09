import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import time

from utils import get_patient_by_id_original, get_patient_by_id_imputed, get_patient_by_id_normalized, get_patient_by_id_standardized, prepare_hdf5
from config import *

## Write sepsis indices to file
sepsis_indices = []
for i in tqdm(range(40336), desc="Inspecting sepsis", ascii=False, ncols=75):
    patient = get_patient_by_id_original(i)
    if 1 in patient["SepsisLabel"].unique():
        sepsis_indices.append(i)

print('Sepsis count:', len(sepsis_indices))

with open("../data/sepsis_indices.json", "w") as fp:
    json.dump(sepsis_indices, fp)

with open(normals_filepath, 'r') as fp:
    normals = json.load(fp)


### Imputation

dirs = ['imputed', 'normalized', 'standardized']
for dirname in dirs:
    dirname = '../data/' + dirname
    if not os.path.exists(dirname):
        os.mkdir(dirname)

def impute(patient):
    for col in patient.columns.values[:-7]:
        if col in ['Hct', 'Hgb', 'Creatinine']:
            normal = normals[col]['avg'][patient['Gender'][0]]
        else:
            normal = normals[col]['avg']
        patient.ffill(inplace=True)
        patient[col].fillna(normal, inplace=True)
    return patient

def sanity_check(patient):
    patient['DBP'] = patient.apply(lambda row: np.nan if row['SBP'] <= row['DBP'] else row['DBP'], axis=1)
    for column, info in normals.items():
        patient[column] = patient[column].apply(lambda x: np.nan if (x < info['accept'][0] or x > info['accept'][1]) else x)
    return patient

for idx in tqdm(range(40336), desc="Imputing", ascii=False, ncols=75):
    p = get_patient_by_id_original(idx)
    p = sanity_check(p)
    p = impute(p)
    assert p.iloc[:,:-5].isnull().values.any() == False
    p.to_csv('../data/imputed/p'+str(idx).zfill(6)+'.csv', index=False)


### Normalization and Standardization
p = get_patient_by_id_imputed(sepsis_indices[74])
measurements = p.columns.drop(['Unit1','Unit2','Gender','HospAdmTime','SepsisLabel'])
mstat = {m:{'max':-np.inf, 'min':np.inf, 'sum':0, 'mean':0, 'std':0} for m in measurements}

count = 0
for i in tqdm(range(40336), desc="Stat 1", ascii=False, ncols=75):
    p = get_patient_by_id_imputed(i)
    assert p.iloc[:,:-5].isnull().values.any() == False
    p = p[measurements]
    count += len(p)
    vsum = p.sum()
    vmax = p.max()
    vmin = p.min()
    for m in measurements:
        mstat[m]['sum'] += vsum[m]
        if vmax[m] > mstat[m]['max']:
            mstat[m]['max'] = vmax[m]
        if vmin[m] < mstat[m]['min']:
            mstat[m]['min'] = vmin[m]

for m in measurements:
    mstat[m]['mean'] = mstat[m]['sum']/count
    
for i in tqdm(range(40336), desc="Stat 2", ascii=False, ncols=75):
    p = get_patient_by_id_imputed(i)
    for m in measurements:
        mstat[m]['std'] += sum(p[m] - mstat[m]['mean']) ** 2

for m in measurements:
    mstat[m]['std'] = (mstat[m]['std'] / (count-1)) ** 0.5
    
with open("../data/mstat.json", "w") as fp:
    json.dump(mstat, fp)
    

### Normalization Min Max
for i in tqdm(range(40336), desc="Normalizing", ascii=False, ncols=75):
    p = get_patient_by_id_imputed(i)
    for m in measurements:
        p[m] = p[m].apply(lambda x: (x - mstat[m]['min']) / (mstat[m]['max'] - mstat[m]['min']))
    p.to_csv('../data/normalized/p'+str(i).zfill(6)+'.csv', index=False)
    
### Standardization zero mean 1 std
for i in tqdm(range(40336), desc="Standardizing", ascii=False, ncols=75):
    p = get_patient_by_id_imputed(i)
    for m in measurements:
        p[m] = p[m].apply(lambda x: (x - mstat[m]['mean']) / mstat[m]['std'])
    p.to_csv('../data/standardized/p'+str(i).zfill(6)+'.csv', index=False)


### Compute Utility Series
def compute_utility_series(labels, dt_early=-12, dt_optimal=-6, dt_late=3.0, max_u_tp=1, min_u_fn=-2, u_fp=-0.05, u_tn=0):
    # Ensure labels is a numpy array
    labels = np.array(labels)
    
    # Check if the patient eventually has sepsis
    if np.any(labels):
        is_septic = True
        t_sepsis = np.argmax(labels) - dt_optimal
    else:
        is_septic = False
        t_sepsis = float('inf')

    n = len(labels)

    # Define slopes and intercepts for utility functions of the form u = m * t + b
    m_1 = max_u_tp / (dt_optimal - dt_early)
    b_1 = -m_1 * dt_early
    m_2 = -max_u_tp / (dt_late - dt_optimal)
    b_2 = -m_2 * dt_late
    m_3 = min_u_fn / (dt_late - dt_optimal)
    b_3 = -m_3 * dt_optimal

    u_pred_pos = np.zeros(n)
    u_pred_neg = np.zeros(n)

    if is_septic:
        t_diff = np.arange(n) - t_sepsis

        # Assign values for u_pred_pos and u_pred_neg using for loops
        for t in range(n):
            if t < t_sepsis + dt_optimal + 1:
                u_pred_pos[t] = max(m_1 * t_diff[t] + b_1, u_fp)
            elif t_sepsis + dt_optimal < t <= t_sepsis + dt_late:
                u_pred_pos[t] = m_2 * t_diff[t] + b_2
                u_pred_neg[t] = m_3 * t_diff[t] + b_3
            else:
                u_pred_pos[t] = 0
                u_pred_neg[t] = 0
    else:
        u_pred_pos[:] = u_fp
        u_pred_neg[:] = u_tn
    
    # Convert numpy arrays to lists if necessary
    u_pred_pos = u_pred_pos.tolist()
    u_pred_neg = u_pred_neg.tolist()
    
    return list(zip(u_pred_neg, u_pred_pos))

for i in tqdm(range(40336), desc="Computing Utility Series", ascii=False, ncols=75):
    p = get_patient_by_id_standardized(i)
    utility_weights = compute_utility_series(p['SepsisLabel'])
    p['UtilityWeights'] = utility_weights
    p.to_csv('../data/standardized/p'+str(i).zfill(6)+'.csv', index=False)

### Use hdf5 for more efficient data accessing
prepare_hdf5()
