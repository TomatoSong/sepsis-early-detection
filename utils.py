import os
import pandas as pd
import matplotlib.pyplot as plt

path1 = "../data/training/"
path2 = "../data/training_setB/"
fnames1 = os.listdir(path1)
fnames2 = os.listdir(path2)
fnames1.sort()
fnames2.sort()

### id from 0 to 40335
def get_patient_by_id_original(idx):
    path   = path1   if idx < 20336 else path2
    fnames = fnames1 if idx < 20336 else fnames2
    idx    = idx     if idx < 20336 else idx-20336
    return pd.read_csv(path + fnames[idx],sep='|')

def get_patient_by_id_imputed(idx):
    return pd.read_csv('../data/imputed/p'+str(idx).zfill(6)+'.csv')

def get_patient_by_id_normalized(idx):
    return pd.read_csv('../data/normalized3/p'+str(idx).zfill(6)+'.csv')

def get_patient_by_id_standardized(idx):
    return pd.read_csv('../data/standardized/p'+str(idx).zfill(6)+'.csv')

def get_patient_by_id_standardized_padded(idx):
    return pd.read_csv('../data/standardized_padded/p'+str(idx).zfill(6)+'.csv')

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