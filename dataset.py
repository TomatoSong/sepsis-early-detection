import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import json
import numpy as np
from tqdm import tqdm
from collections import Counter

from utils import get_patient_by_id_original
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
        if not os.path.exists(path):
            for idx in tqdm(range(40336), desc="Building idx map", ascii=False, ncols=75):
                p = get_patient_by_id_original(idx)
                for t in range(starting_offset-1,len(p)):
                    label = int(p.at[t, 'SepsisLabel'])
                    hist = (idx,0,t,label) if t < seq_len else (idx,t-seq_len+1,t,label)
                    assert t < len(p)
                    assert hist[2]-hist[1]+1 <= seq_len
                    index_map.append(hist)
                    if idx in self.patient_ids:
                        index_map_subset.append(hist)
                        self.ratio[label] += 1
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
