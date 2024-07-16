import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import json
import numpy as np
from tqdm import tqdm
from collections import Counter
import re
import ast
import sys

from utils import get_patient_by_id_original, get_patient_by_id_standardized, get_patient_data, get_synthetic_patient_by_id
from config import All_COLS

DATA_FOLDER = "../data"
processing = {
    "original"    : "/",
    "imputed"     : "/imputed3/",
    "normalized"  : "/normalized3/",
    "standardized": "/standardized/",
    "standardized_padded": "/standardized_padded/"
}

class SepsisDataset(Dataset):
    def __init__(self, patient_ids, seq_len, starting_offset, columns, method='standardized'):
        self.patient_ids = patient_ids
        self.columns = columns
        self.method = method
        self.seq_len = seq_len
        self.ratio = [0,0]
        self.idxmap_subset = self.build_index_map(seq_len, starting_offset)
        
    def check_store(self, seq_len, starting_offset):
        try:
            directory = '../data/idxmap_subset/'
            for filename in os.listdir(directory):
                pattern = r'_(\d+)'
                matches = re.findall(pattern, filename)
                if len(matches) < 2:
                    print("Illegal idx map file found {}!".format(filename))
                    continue
                if int(matches[0]) == seq_len and int(matches[1]) == starting_offset:
                    f = os.path.join(directory, filename)
                    with open(f, "r") as fp:
                        info = json.load(fp)
                        if info['patient_ids'] == self.patient_ids:
                            self.ratio = info['ratio']
                            return info['index_map_subset']
        except Exception as e:
            print(f"An error occurred during file search: {e}")
        return None
        
    def build_index_map(self, seq_len, starting_offset):
        index_map_subset = self.check_store(seq_len, starting_offset)
        if not index_map_subset:
            path = '../data/idxmap_'+str(seq_len)+'_'+str(starting_offset)+".json"
            index_map = []
            index_map_subset = []
            patients_subset = set()
            patients_all = set()
            if not os.path.exists(path):
                for pid in tqdm(range(40336), desc="Building idx map", ascii=False, ncols=75):
                    p = get_patient_by_id_standardized(pid)
                    if len(p) < starting_offset:
                        t = len(p)-1
                        label = int(p.at[t, 'SepsisLabel'])
                        u_weights = [p.loc[t, 'UtilityNeg'], p.loc[t, 'UtilityPos']]
                        hist = (pid,0,t,label,u_weights,seq_len-t-1) # patient id, start, end, label, utility, padding
                        assert hist[2] < len(p)
                        assert hist[2]-hist[1]+1+hist[5] == seq_len
                        index_map.append(hist)
                        patients_all.add(pid)
                        if pid in self.patient_ids:
                            index_map_subset.append(hist)
                            self.ratio[label] += 1
                            patients_subset.add(pid)
                    else:
                        for t in range(starting_offset-1,len(p)):
                            label = int(p.at[t, 'SepsisLabel'])
                            u_weights = [p.loc[t, 'UtilityNeg'], p.loc[t, 'UtilityPos']]
                            hist = (pid,0,t,label,u_weights,seq_len-t-1) if t < seq_len else (pid,t-seq_len+1,t,label,u_weights,0)
                            assert hist[2] < len(p)
                            assert hist[2]-hist[1]+1+hist[5] == seq_len
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
                        label = item[3]
                        if pid in self.patient_ids:
                            index_map_subset.append(item)
                            self.ratio[label] += 1
                            patients_subset.add(pid)
            print('using {} patients in current subset'.format(len(patients_subset)))

            assert len(patients_subset) == len(self.patient_ids)
            assert patients_subset == set(self.patient_ids)
            print('len idxmap {}'.format(len(index_map)))
            dirname = '../data/idxmap_subset/'
            if not os.path.exists(dirname):
                os.mkdir(dirname)
            path = dirname + 'idxmap_subset_'+str(seq_len)+'_'+str(starting_offset)+'_'+str(len(index_map_subset))+".json"
            with open(path, "w") as fp:
                json.dump(dict(patient_ids=self.patient_ids,
                               ratio = self.ratio,
                               index_map_subset=index_map_subset
                              ), 
                          fp)
        
        print('len idxmap subset {}'.format(len(index_map_subset)))
        return index_map_subset
    
    def get_ratio(self):
        return self.ratio
        
    def __len__(self):
        return (len(self.idxmap_subset))

    def __getitem__(self, idx):
        pid, start, end, label, u_weights, padding = self.idxmap_subset[idx]
        data = get_patient_data(pid, start, end)
        data = [[0]*len(self.columns)]*padding + data
        mask = [True]*padding + [False]*(self.seq_len-padding)
        assert len(data) == self.seq_len
        assert len(mask) == self.seq_len
        # Return: patient_id, latest_hour, clinical_data, label, utility_weights, mask, empty_tensor
        return pid, end, torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.float32), torch.tensor(u_weights), torch.tensor(mask), torch.tensor([])


class RawDataset(Dataset):
    def __init__(self, pids):
        self.build_index_map(pids)
        
    def build_index_map(self, pids):
        self.x = []
        self.y = []
        self.u = []
        self.ids = []
        for pid in tqdm(pids, desc="Preparing raw data", ascii=False, ncols=75):
            p = get_patient_by_id_standardized(pid)
            self.x.extend(p[All_COLS].values.tolist())
            self.y.extend(p['SepsisLabel'].tolist())
            u = p.apply(lambda row: [row['UtilityNeg'], row['UtilityPos']], axis=1).tolist()
            self.u.extend(u)
            self.ids.extend([(pid, rid) for rid in range(len(p))])
        print('Populated {} dps from {} patients'.format(len(self.y), len(pids)))
        return
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        # Return: patient_id, latest_hour, clinical_data, label, utility_weights, empty_tensor, empty_tensor
        return self.ids[idx][0], self.ids[idx][1], torch.tensor(self.x[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32), torch.tensor(self.u[idx], dtype=torch.float32), torch.tensor([]), torch.tensor([])


class SyntheticDataset(Dataset):
    def __init__(self, pids, seq_len, starting_offset, columns):
        self.seq_len = seq_len
        self.patient_ids = pids
        self.columns = columns
        self.ratio = [0,0]
        self.idxmap_subset = self.build_index_map(seq_len, starting_offset)

    def check_store(self, seq_len, starting_offset):
        try:
            directory = '../data/idxmap_subset/'
            for filename in os.listdir(directory):
                pattern = r'syn_[^0-9]*(_\d+)+'
                matches = re.findall(pattern, filename)
                if len(matches) < 2:
                    print("Illegal idx map file found {}!".format(filename))
                    continue
                if int(matches[0]) == seq_len and int(matches[1]) == starting_offset:
                    f = os.path.join(directory, filename)
                    with open(f, "r") as fp:
                        info = json.load(fp)
                        if info['patient_ids'] == self.patient_ids:
                            return info['index_map_subset']
        except Exception as e:
            print(f"An error occurred during file search: {e}")
        return None
        
    def build_index_map(self, seq_len, starting_offset):
        index_map_subset = self.check_store(seq_len, starting_offset)
        if not index_map_subset:
            path = '../data/syn_idxmap_'+str(seq_len)+'_'+str(starting_offset)+".json"
            index_map = []
            index_map_subset = []
            patients_subset = set()
            patients_all = set()
            if not os.path.exists(path):
                for pid in tqdm(range(10000), desc="Building idx map", ascii=False, ncols=75):
                    p = get_patient_by_id_original(pid)
                    if len(p) < starting_offset:
                        t = len(p)-1
                        label = int(p.at[t, 'SepsisLabel'])
                        hist = (pid,0,t,label,seq_len-t-1) # patient id, start, end, label, padding
                        assert hist[2] < len(p)
                        assert hist[2]-hist[1]+1+hist[4] == seq_len
                        index_map.append(hist)
                        patients_all.add(pid)
                        if pid in self.patient_ids:
                            index_map_subset.append(hist)
                            self.ratio[label] += 1
                            patients_subset.add(pid)
                    else:
                        for t in range(starting_offset-1,len(p)):
                            label = int(p.at[t, 'SepsisLabel'])
                            hist = (pid,0,t,label, seq_len-t-1) if t < seq_len else (pid,t-seq_len+1,t,label,0)
                            assert hist[2] < len(p)
                            assert hist[2]-hist[1]+1+hist[4] == seq_len
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
            print('len idxmap {}'.format(len(index_map)))
            path = '../data/idxmap_subset/syn_idxmap_subset_'+str(seq_len)+'_'+str(starting_offset)+'_'+str(len(index_map_subset))+".json"
            with open(path, "w") as fp:
                json.dump(dict(patient_ids=self.patient_ids, index_map_subset=index_map_subset), fp)
        
        print('len idxmap subset {}'.format(len(index_map_subset)))
        return index_map_subset
    
    def get_ratio(self):
        return self.ratio
        
    def __len__(self):
        return (len(self.idxmap_subset))

    def __getitem__(self, idx):
        pid, start, end, label, padding = self.idxmap_subset[idx]
        data = [0]*padding + get_patient_data(pid, start, end)
        mask = [True]*padding + [False]*(self.seq_len-padding)
        assert len(data) == self.seq_len
        assert len(mask) == self.seq_len
        return pid, end, torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.float32), torch.tensor(mask), torch.tensor([])


class WeibullCoxDataset(Dataset):
    def __init__(self, pids):
        self.build_index_map(pids)
        
    def build_index_map(self, pids):
        path = '../data/wc_dataset.json'
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                if set(data['pids']) == set(pids):
                    self.x = data['x']
                    self.tau = data['tau']
                    self.S = data['S']
                    self.ids = data['ids']
                    self.y = data['y']
                    self.u = data['u']
                    return
        except Exception as err:
            print('Failed matching saved Weibull-Cox dataset: ', err)
        
        self.x = []
        self.tau = []
        self.S = []
        self.ids = []
        self.y = []
        self.u = []
        for pid in tqdm(pids, desc="Preparing data", ascii=False, ncols=75):
            p = get_patient_by_id_standardized(pid)
            for rid in range(len(p)-5):
                window = p.iloc[rid:min(len(p),rid+6),:]
                S = 0 if (window['SepsisLabel'] == 1).any() else 1
                tau = max(0.1, window['SepsisLabel'].idxmax()-rid if (window['SepsisLabel'] == 1).any() else 7)
                x = p.loc[rid, All_COLS].astype(float).tolist()
                y = p.loc[rid, ['SepsisLabel']].astype(int).tolist()
                u = [p.loc[rid, 'UtilityNeg'], p.loc[rid, 'UtilityPos']]
                self.x.append(x)
                self.y.append(y)
                self.u.append(u)
                self.tau.append(tau)
                self.S.append(S)
                self.ids.append((pid, rid))

        data = {
            'pids': pids,
            'x'   : self.x,
            'tau' : self.tau,
            'S'   : self.S,
            'y'   : self.y,
            'u'   : self.u,
            'ids' : self.ids
        }
        with open(path, "w") as f:
            json.dump(data, f)
        return
        
    def __len__(self):
        return len(self.S)
    
    def __getitem__(self, idx):
        # Return: patient_id, latest_hour, clinical_data, label, utility_weights, tau, S
        return self.ids[idx][0], self.ids[idx][1], torch.tensor(self.x[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32), torch.tensor(self.u[idx], dtype=torch.float32), torch.tensor(self.tau[idx], dtype=torch.float32), torch.tensor(self.S[idx], dtype=torch.int32)