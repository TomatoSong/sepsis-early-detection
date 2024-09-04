import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

from utils_biogear import get_patient_by_id_original_biogear, get_patient_by_id_imputed_biogear, get_patient_by_id_standardized_biogear, prepare_hdf5_biogear, normals
#from preprocessing import compute_utility_series

## Write sepsis indices to file
sepsis_indices = []
for i in tqdm(range(190), desc="Inspecting sepsis", ascii=False, ncols=36):
    patient = get_patient_by_id_original_biogear(i)
    if 1 in patient["sepsis"].unique():
        sepsis_indices.append(i)

#print('Sepsis count:', len(sepsis_indices))


file_path = "../data/sepsis_indices_biogear.json"
directory = os.path.dirname(file_path)
if not os.path.exists(directory):
    os.makedirs(directory, exist_ok=True)

with open(file_path, "w") as fp:
    json.dump(sepsis_indices, fp)

### Imputation

dirs = ['imputed_biogear', 'normalized_biogear', 'standardized_biogear']
for dirname in dirs:
    dirname = '../data/' + dirname
    if not os.path.exists(dirname):
        os.mkdir(dirname)

def impute_biogear(patient):
    for col in patient.columns.values[2:-2]:
        #if col in ['Hct', 'Hgb', 'Creatinine']:
        #    normal = normals[col]['avg'][patient['Gender'][0]]
        #else:
        normal = normals[col]['avg']
        #print(p["PTT"])
        patient.ffill(inplace=True)
        #print(p["PTT"])
        patient[col].fillna(normal, inplace=True)
        #print(p["PTT"])
        #sys.exit()
    return patient

def sanity_check_biogear(patient):
    for column, info in normals.items():
        if (column == "PTT"):
            #print(p["PTT"])
            patient[column] = patient[column].apply(lambda x: info['accept'][0] if x < info['accept'][0] else (info['accept'][1] if x > info['accept'][1] else x)) # problem is here
            #print(p["PTT"])
            #sys.exit()
    return patient


for idx in tqdm(range(190), desc="Imputing", ascii=False, ncols=36):
    p = get_patient_by_id_original_biogear(idx)
    p = sanity_check_biogear(p)
    p = impute_biogear(p)
    assert p.iloc[:,:].isnull().values.any() == False
    p.to_csv('../data/imputed_biogear/'+str(idx)+'.csv', index=False)

### Normalization and Standardization
p = get_patient_by_id_imputed_biogear(sepsis_indices[0])
measurements = p.columns.drop(['patient_id', 'time', 'qsofa', 'sepsis']) # should we drop qsofa as well
mstat = {m:{'max':-np.inf, 'min':np.inf, 'sum':0, 'mean':0, 'std':0} for m in measurements} # this still leaves the patient id column

count = 0
for i in tqdm(range(190), desc="Stat 1", ascii=False, ncols=36):
    p = get_patient_by_id_imputed_biogear(i)
    assert p.iloc[:,:].isnull().values.any() == False
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
    
for i in tqdm(range(190), desc="Stat 2", ascii=False, ncols=36):
    p = get_patient_by_id_imputed_biogear(i)
    for m in measurements:
        mstat[m]['std'] += sum(p[m] - mstat[m]['mean']) ** 2

for m in measurements:
    mstat[m]['std'] = (mstat[m]['std'] / (count-1)) ** 0.5
    
with open("../data/mstat_biogear.json", "w") as fp:
    json.dump(mstat, fp)
    

### Normalization Min Max
for i in tqdm(range(190), desc="Normalizing", ascii=False, ncols=36):
    p = get_patient_by_id_imputed_biogear(i)
    for m in measurements:
        if (m == "CTSresistance"):
            print(p[m])
            if mstat[m]['max'] != mstat[m]['min']:
                p[m] = p[m].apply(lambda x: (x - mstat[m]['min']) / (mstat[m]['max'] - mstat[m]['min']))
            else:
                p[m] = p[m].apply(lambda x: 1)
            print(p[m])
    p.to_csv('../data/normalized_biogear/'+str(i)+'.csv', index=False)


### Standardization zero mean 1 std
for i in tqdm(range(190), desc="Standardizing", ascii=False, ncols=36):
    p = get_patient_by_id_imputed_biogear(i)
    for m in measurements:
        if mstat[m]['std'] != 0:
            p[m] = p[m].apply(lambda x: (x - mstat[m]['mean']) / mstat[m]['std'])
        else:
            p[m] = p[m].apply(lambda x: 1)
    p.to_csv('../data/standardized_biogear/'+str(i)+'.csv', index=False)
    
# ### Padding
# for i in tqdm(range(40336), desc="Padding", ascii=False, ncols=75):
#     p = get_patient_by_id_standardized(i)
#     new_rows = pd.DataFrame(0, index=np.arange(100), columns=p.columns, dtype='float32')
#     padded_sequence = pd.concat([new_rows, p])
#     padded_sequence.to_csv('../data/standardized_padded/p'+str(i).zfill(6)+'.csv', index=False)

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
    u_pred = [(pos, neg) for pos, neg in zip(u_pred_pos, u_pred_neg)]
    #print(u_pred)
    
    return u_pred

# Utility series calculation for biogear data should be same as normal

for i in tqdm(range(190), desc="Computing Utility Series", ascii=False, ncols=36):
    p = get_patient_by_id_standardized_biogear(i)
    u_pred = compute_utility_series(p['sepsis'])
    # u_pred_pos = compute_utility_series(p['sepsis'])[0]
    # u_pred_neg = compute_utility_series(p['sepsis'])[1]
    # p['UtilityPos'] = u_pred_pos
    # p['UtilityNeg'] = u_pred_neg
    p['UtilityScores'] = u_pred
    p.to_csv('../data/standardized_biogear/'+str(i)+'.csv', index=False)

### Use hdf5 for more efficient data accessing
prepare_hdf5_biogear() # should I only include the columns from the normals dictionary