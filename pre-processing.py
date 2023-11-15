import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import time

path1 = "../data/training/"
path2 = "../data/training_setB/"
fnames1 = os.listdir(path1)
fnames2 = os.listdir(path2)
fnames1.sort()
fnames2.sort()
print(len(fnames1),len(fnames2))

### id from 0 to 40335
def get_patient_by_id_original(idx):
    path   = path1   if idx < 20336 else path2
    fnames = fnames1 if idx < 20336 else fnames2
    idx    = idx     if idx < 20336 else idx-20336
    return pd.read_csv(path + fnames[idx],sep='|')

def plot(patient):
    sepsis = 1 in patient.SepsisLabel.unique()
    if sepsis:
        predict_hour = patient.SepsisLabel.ne(0).idxmax()
        sepsis_hour = predict_hour + 6
        print("Pateint developed sepis at hour ", sepsis_hour)
    else:
        print("Patient did not develop sepsis")
    patient = patient.drop(columns=['Unit1','Unit2','Gender','Age', 'HospAdmTime', 'SepsisLabel'])
    fig, axs = plt.subplots(len(patient.columns.values)-1, 1, sharex=True, figsize=(10,100))
    for i, col in enumerate(patient.columns.values[:-1]):
        patient.plot(x=['ICULOS'], y=[col], kind='scatter', ax=axs[i])
        plt.sca(axs[i])
        if sepsis:
            plt.axvline(x=predict_hour, color='b')
            plt.axvline(x=sepsis_hour, color='r')
            
            
## Write sepsis indices to file
sepsis_indices = []
for i in tqdm(range(40336), desc="Inspecting sepsis", ascii=False, ncols=75):
    patient = get_patient_by_id_original(i)
    if 1 in patient.SepsisLabel.unique():
        sepsis_indices.append(i)

print('Sepsis count:', len(sepsis_indices))

with open("../data/sepsis_indices.json", "w") as fp:
    json.dump(sepsis_indices, fp)
    

normals = {
    'HR': { 'avg':80, 'range': [60,100], 'accept':[20,250] },
    'O2Sat': { 'avg':98, 'range': [95,100], 'accept':[0,100] },
    'Temp': { 'avg':37, 'range': [36.1,37.2], 'accept':[25,45] },
    'SBP': { 'avg':110, 'range': [90,120], 'accept':[60,250] },
    'MAP': { 'avg':85, 'range': [70,100], 'accept':[50,200] },
    'DBP': { 'avg':70, 'range': [60,80], 'accept':[40,120] },
    'Resp': { 'avg':14, 'range': [12,16], 'accept':[10,60] },
    'EtCO2': { 'avg':40, 'range': [35,45], 'accept':[20,60] },
    'BaseExcess': { 'avg':0, 'range': [-2,2], 'accept':[-50,50] },
    'HCO3': { 'avg':24, 'range': [22,26], 'accept':[0,100] },
    'FiO2': { 'avg':0.21, 'range': [0.21,1],'accept':[0,1] },
    'pH': { 'avg':7.4, 'range': [7.35,7.45],'accept':[5,9] },
    'PaCO2': { 'avg':40, 'range': [35,45], 'accept':[0,100] },
    'SaO2': { 'avg':97, 'range': [94,100], 'accept':[0,100] },
    'AST': { 'avg':28, 'range': [8,48],'accept':[0,100] },
    'BUN': { 'avg':12.5, 'range': [5, 20], 'accept':[0,100] },
    'Alkalinephos': { 'avg':84.5, 'range': [40,129], 'accept':[0,200] },
    'Calcium': { 'avg':9.45, 'range': [8.5,10.2], 'accept':[0,100] },
    'Chloride': { 'avg':100, 'range': [96,106], 'accept':[0,500] },
    'Creatinine': { 'avg':[0.85,1.1], 'range': [[0.6,1.1],[0.9,1.3]], 'accept':[0,50] },
    'Bilirubin_direct': { 'avg':0.3, 'range': [0,0.3], 'accept':[0,20] },
    'Glucose': { 'avg':106, 'range': [72,140], 'accept':[0,1000] },
    'Lactate': { 'avg':12.15, 'range': [4.5,19.8], 'accept':[0,200] },
    'Magnesium': { 'avg':0.975, 'range': [0.85,1.1], 'accept':[0,20] },
    'Phosphate': { 'avg':3.75, 'range': [3.0,4.5], 'accept':[0,50] },
    'Potassium': { 'avg':4.4, 'range': [3.6,5.2], 'accept':[0,50] },
    'Bilirubin_total': { 'avg':1.2, 'range': [0.3,1.2], 'accept':[0,100] },
    'TroponinI': { 'avg':0.2, 'range': [0,0.4], 'accept':[0,20] },
    'Hct': { 'avg':[42.5, 48.6], 'range': [[35.5,44.9],[38.3,48.6]], 'accept':[0,200] },
    'Hgb': { 'avg':[13.75, 15.05], 'range': [[12.0,15.5],[13.5,17.5]], 'accept':[0,200] },
    'PTT': { 'avg':30, 'range': [25,35], 'accept':[0,200] },
    'WBC': { 'avg':7.75, 'range': [4.5,11], 'accept':[0,100] },
    'Fibrinogen': { 'avg':300, 'range': [200,400], 'accept':[0,2000] },
    'Platelets': { 'avg':300, 'range': [150,450], 'accept':[0,2000] }
}

### Imputation

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
    for column, info in normals.items():
        patient[column] = patient[column].apply(lambda x: info['accept'][0] if x < info['accept'][0] else (info['accept'][1] if x > info['accept'][1] else x))
    return patient


for idx in tqdm(range(40336), desc="Imputing", ascii=False, ncols=75):
    p = get_patient_by_id_original(idx)
    p = sanity_check(p)
    p = impute(p)
    assert p.iloc[:,:-5].isnull().values.any() == False
    p.to_csv('../data/imputed/p'+str(idx).zfill(6)+'.csv', index=False)
    
def get_patient_by_id_imputed(idx):
    return pd.read_csv('../data/imputed/p'+str(idx).zfill(6)+'.csv')


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
    

def get_patient_by_id_normalized(idx):
    return pd.read_csv('../data/normalized3/p'+str(idx).zfill(6)+'.csv')

def get_patient_by_id_standardized(idx):
    return pd.read_csv('../data/standardized/p'+str(idx).zfill(6)+'.csv')