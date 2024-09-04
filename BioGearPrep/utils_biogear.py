import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix
from tqdm import tqdm
import json
import torch
import h5py
import re

COLS_BIOGEAR = ['patient_id','time','BaseExcess(mmol/L)','BUN(mg/dL)','PTT','CTSresistance','HeartRate(1/min)','MeanArterialPressure(mmHg)','SystolicArterialPressure(mmHg)','DiastolicArterialPressure(mmHg)','CardiacOutput(L/min)','HemoglobinContent(g)','CentralVenousPressure(mmHg)','Hematocrit','ArterialBloodPH','UrinationRate(mL/hr)','WhiteBloodCellCount(ct/uL)','UrineProductionRate(mL/min)','RespirationRate(1/min)','OxygenSaturation','CarbonDioxideSaturation','CoreTemperature(degC)','SkinTemperature(degC)','TotalBilirubin(mg/dL)','Phosphate(mmol/mL)','Bicarbonate-BloodConcentration(ug/mL)','Creatinine-BloodConcentration(ug/mL)','Lactate-BloodConcentration(ug/mL)','Calcium-BloodConcentration(ug/mL)','Chloride-BloodConcentration(ug/mL)','Glucose-BloodConcentration(ug/mL)','Potassium-BloodConcentration(ug/mL)','Hemoglobin-BloodConcentration(g/mL)','ArterialCarbonDioxidePressure(mmHg)','qsofa']

path_biogear = "C:/Users/prave/CPSIL Research/SepsisDetection/data/BioGearData/BioGearDataCSV/"
fnames_biogear = os.listdir(path_biogear)

normals = {
    'BaseExcess(mmol/L)': { 'avg':0, 'range': [-2,2], 'accept':[-50,50] },
    'BUN(mg/dL)': { 'avg':12.5, 'range': [5, 20], 'accept':[0,300] },
    'PTT': { 'avg':30, 'range': [25,35], 'accept':[0,250] },
    'CTSresistance': { 'avg':1.35, 'range':[0.1,1.8], 'accept':[0,2] },
    'HeartRate(1/min)': { 'avg':80, 'range': [60,100], 'accept':[0,300] },
    'MeanArterialPressure(mmHg)': { 'avg':85, 'range': [70,100], 'accept':[20,300] },
    'SystolicArterialPressure(mmHg)': { 'avg':110, 'range': [90,120], 'accept':[0,300] },
    'DiastolicArterialPressure(mmHg)': { 'avg':70, 'range': [60,80], 'accept':[10,180] },
    'CardiacOutput(L/min)': { 'avg':5.5, 'range': [4, 8], 'accept': [4, 8] },
    'HemoglobinContent(g)': { 'avg': 15.5, 'range': [13.8,17.2], 'accept':[12.1,17.2] },
    'CentralVenousPressure(mmHg)': { 'avg': 5, 'range': [2,8], 'accept':[2,8] },
    'Hematocrit': { 'avg': 45.55, 'range': [[35.5,44.9],[38.3,48.6]], 'accept':[0,75] },
    'ArterialBloodPH': { 'avg': 7.4, 'range': [7.35,7.45], 'accept': [5,9] },
    'UrinationRate(mL/hr)': { 'avg': 0, 'range': [0,0], 'accept': [0,0] },
    'WhiteBloodCellCount(ct/uL)': { 'avg':7.75, 'range': [4.5,11], 'accept':[0,200] },
    'UrineProductionRate(mL/min)': { 'avg': 58, 'range': [33,83], 'accept': [33,83] },
    'RespirationRate(1/min)': { 'avg':14, 'range': [12,16], 'accept':[0,100] },
    'OxygenSaturation': { 'avg':98, 'range': [95,100], 'accept':[0,100] },
    'CarbonDioxideSaturation': { 'avg': 3.5, 'range': [2,5], 'accept': [0,5] },
    'CoreTemperature(degC)': { 'avg': 37, 'range': [36.1,37.2], 'accept': [20,45] },
    'SkinTemperature(degC)': { 'avg':37, 'range': [36.1,37.2], 'accept':[20,45] },
    'TotalBilirubin(mg/dL)': { 'avg':1.2, 'range': [0.3,1.2], 'accept': [0,50] },
    'Phosphate(mmol/mL)': { 'avg':3.75, 'range': [3.0,4.5], 'accept':[0,50] },
    'Bicarbonate-BloodConcentration(ug/mL)': { 'avg':24, 'range': [22,26], 'accept':[0,100] },
    'Creatinine-BloodConcentration(ug/mL)': { 'avg':0.975, 'range': [[0.6,1.1],[0.9,1.3]], 'accept':[0,50] },
    'Lactate-BloodConcentration(ug/mL)': { 'avg':12.15, 'range': [4.5,19.8], 'accept':[0,35] },
    'Calcium-BloodConcentration(ug/mL)': { 'avg':9.45, 'range': [8.5,10.2], 'accept':[0,100] },
    'Chloride-BloodConcentration(ug/mL)': { 'avg':100, 'range': [96,106], 'accept':[0,500] },
    'Glucose-BloodConcentration(ug/mL)': { 'avg':106, 'range': [72,140], 'accept':[0,1000] },
    'Potassium-BloodConcentration(ug/mL)': { 'avg':4.4, 'range': [3.6,5.2], 'accept':[0,10] },
    'Hemoglobin-BloodConcentration(g/mL)': { 'avg':14.4, 'range': [[12.0,15.5],[13.5,17.5]], 'accept':[0,35] },
    'ArterialCarbonDioxidePressure(mmHg)': { 'avg':40, 'range': [35,45], 'accept':[10,100] }
}
#print(len(fnames_biogear))

for fname in fnames_biogear:
    match = re.search(r'\.([^.]*)$', fname)
    if not match or match.group(1) != 'csv':
        fnames_biogear.remove(fname)
        
fnames_biogear.sort()

print(len(fnames_biogear))
with open('C:/Users/prave/CPSIL Research/SepsisDetection/data/fnames_biogear.json', 'w') as f:
    json.dump(fnames_biogear, f)

def get_patient_by_id_original_biogear(idx):
    path   = path_biogear
    fnames = fnames_biogear

    file_path = path + fnames[idx]
    match = re.search(r'\.([^.]*)$', file_path)
    if match and match.group(1) == 'csv':
        return pd.read_csv(file_path)
    else:
        print('Error loading original data {}'.format(file_path))
        sys.exit()

def get_patient_by_id_imputed_biogear(idx):
    return pd.read_csv('../data/imputed_biogear/'+str(idx)+'.csv')

def get_patient_by_id_standardized_biogear(idx):
    return pd.read_csv('C:/Users/prave/CPSIL Research/SepsisDetection/src/data/standardized_biogear/'+str(idx)+'.csv')

def prepare_hdf5_biogear():
    with h5py.File('../data/patient_data_biogear.h5', 'w') as f:
        count = 0
        for pid in tqdm(range(190)):
            p = get_patient_by_id_standardized_biogear(pid)[COLS_BIOGEAR]
            #assert p.isnull().values.any() == False
            if p.isnull().values.any() == True:
                count+=1
                print("pid: ")
                print(pid)
                print(p)
            grp = f.create_group(f'patient_{pid}')
            grp.create_dataset('data', data=p.to_numpy(), compression='gzip')
        print(count)

def get_patient_data_biogear(pid, start, end):
    with h5py.File('C:/Users/prave/CPSIL Research/SepsisDetection/src/data/patient_data_biogear.h5', 'r') as f:
        data = f[f'patient_{pid}/data'][start:end+1]
    return data.tolist()