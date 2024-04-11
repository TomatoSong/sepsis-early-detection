import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import json
import numpy as np
from tqdm import tqdm
from collections import Counter

from utils import get_patient_by_id_original, get_patient_by_id_standardized
from config import padding_offset

DATA_FOLDER = "../data"
processing = {
    "original"    : "/",
    "imputed"     : "/imputed3/",
    "normalized"  : "/normalized3/",
    "standardized": "/standardized/",
    "standardized_padded": "/standardized_padded/"
}

COLS = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
       'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
       'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine',
       'Bilirubin_direct', 'Glucose', 'Lactate', 'Magnesium', 'Phosphate',
       'Potassium', 'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT',
       'WBC', 'Fibrinogen', 'Platelets', 'Age', 'Gender', 'ICULOS']

class SepsisDataset(Dataset):
    def __init__(self, patient_ids, seq_len=72, starting_offset=24, cols=COLS, method='standardized_padded',padding_offset=padding_offset):
        self.folder_path = DATA_FOLDER + processing[method]
        self.cols = cols
        self.patient_ids = patient_ids
        self.method = method
        self.seq_len = seq_len
        self.padding_offset = padding_offset
        self.ratio = [0,0]
        self.idxmap, self.idxmap_subset = self.build_index_map(seq_len, starting_offset)
        
    def build_index_map(self, seq_len, starting_offset):
        path = '../data/idxmap_'+str(seq_len)+'_'+str(starting_offset)+".json"
        index_map = []
        index_map_subset = []
        patients_subset = set()
        patients_all = set()
        if not os.path.exists(path):
            for pid in tqdm(range(40336), desc="Building idx map", ascii=False, ncols=75):
                p = get_patient_by_id_original(pid)
                if len(p) < starting_offset:
                    t = len(p)-1
                    label = int(p.at[t, 'SepsisLabel'])
                    hist = (pid,0,t,label)
                    assert t < len(p)
                    assert hist[2]-hist[1]+1 <= seq_len
                    index_map.append(hist)
                    patients_all.add(pid)
                    if pid in self.patient_ids:
                        index_map_subset.append(hist)
                        self.ratio[label] += 1
                        patients_subset.add(pid)
                else:
                    for t in range(starting_offset-1,len(p)):
                        label = int(p.at[t, 'SepsisLabel'])
                        hist = (pid,0,t,label) if t < seq_len else (pid,t-seq_len+1,t,label)
                        assert t < len(p)
                        assert hist[2]-hist[1]+1 <= seq_len
                        index_map.append(hist)
                        patients_all.add(pid)
                        if pid in self.patient_ids:
                            index_map_subset.append(hist)
                            self.ratio[label] += 1
                            patients_subset.add(pid)
            print('populated {} patients into {} timeseries'.format(len(patients_all), len(index_map)))
            with open(path, "w") as fp:
                json.dump(index_map, fp)
        else:
            with open(path, "r") as fp:
                index_map = json.load(fp)
                for item in tqdm(index_map, desc="Building idx map subset", ascii=False, ncols=75):
                    pid = item[0]
                    label = item[-1]
                    if pid in self.patient_ids:
                        index_map_subset.append(item)
                        self.ratio[label] += 1
                        patients_subset.add(pid)
        print('using {} patients in current subset'.format(len(patients_subset)))
        
        assert len(patients_subset) == len(self.patient_ids)
        assert patients_subset == set(self.patient_ids)
        print('len idxmap {}, len idxmap subset {}'.format(len(index_map), len(index_map_subset)))
        return index_map, index_map_subset
    
    def get_ratio(self):
        return self.ratio
        
    def __len__(self):
        return (len(self.idxmap_subset))

    def __getitem__(self, idx):
        pid, start, end, label = self.idxmap_subset[idx]
        p = pd.read_csv(self.folder_path+'p'+str(pid).zfill(6)+'.csv')
        p = p[self.cols]
        padded_start = end+self.padding_offset-self.seq_len+1
        padded_end   = end+self.padding_offset
        assert padded_start >= 0, 'padding length not enough'
        assert padded_end < len(p)
        assert padded_end-padded_start+1 == self.seq_len
        seq = p.iloc[padded_start:padded_end+1]
        offset = self.seq_len - (end-start+1)
        mask = [True if i < offset else False for i in range(self.seq_len)]
        data = seq.values.tolist()
        return pid, end, Tensor(data), Tensor(mask), label

    
class RawDataset(Dataset):
    def __init__(self, pids):
        self.build_index_map(pids)
        
    def build_index_map(self, pids):
        self.x = []
        self.y = []
        for pid in tqdm(pids, desc="Preparing data", ascii=False, ncols=75):
            p = get_patient_by_id_standardized(pid)
            self.x.extend(p[COLS].values.tolist())
            self.y.extend(p['SepsisLabel'].tolist())
        print('Populated {} dps from {} patients'.format(len(self.y), len(pids)))
        return
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return torch.tensor(self.x[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)


class WeibullCoxDataset(Dataset):
    def __init__(self, pids):
        self.build_index_map(pids)
        
    def build_index_map(self, pids):
        self.x = []
        self.tau = []
        self.S = []
        for pid in tqdm(pids, desc="Preparing data", ascii=False, ncols=75):
            p = get_patient_by_id_standardized(pid)
            for rid in range(len(p)-5):
                window = p.iloc[rid:min(len(p),rid+6),:]
                S = 0 if (window['SepsisLabel'] == 1).any() else 1
                tau = max(0.1, window['SepsisLabel'].idxmax()-rid if (window['SepsisLabel'] == 1).any() else 7)
                x = p.loc[rid, COLS].tolist()
                self.x.append(x)
                self.tau.append(tau)
                self.S.append(S)
        return
        
    def __len__(self):
        return len(self.S)
    
    def __getitem__(self, idx):
        return torch.tensor(self.x[idx], dtype=torch.float32), torch.tensor(self.tau[idx], dtype=torch.float32), torch.tensor(self.S[idx], dtype=torch.int32)