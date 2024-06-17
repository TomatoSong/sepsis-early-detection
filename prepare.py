import json
import random
import os
from config import sepsis_ids_filepath, train_ids_filepath, test_ids_filepath
from utils import get_patient_by_id_original, get_patient_by_id_standardized

def train_test_split():
    if not (os.path.exists(sepsis_ids_filepath)):
        print(f"Finding sepsis ids.")
        sepsis_ids = []
        nonsepsis_ids = []
        for pid in range(40336):
            p = get_patient_by_id_original(pid)
            if (p['SepsisLabel'] == 0).all():
                sepsis_ids.append(pid)
            else:
                nonsepsis_ids.append(pid)
        assert len(sepsis_ids) + len(nonsepsis_ids) == 40336
        assert set(sepsis_ids + nonsepsis_ids) == set(range(40336))
        with open(sepsis_ids_filepath, "w") as f:
            json.dump(sepsis_ids, f)
        
    with open(sepsis_ids_filepath, "r") as fp:
        sepsis_ids = json.load(fp)
    nonsepsis_ids = list(set(range(40336))-set(sepsis_ids))
    print('sepsis {} nonsepsis {}'.format(len(sepsis_ids), len(nonsepsis_ids)))

    # Train Test Split
    train_pos = random.sample(sepsis_ids, int(0.85*len(sepsis_ids)))
    train_neg = random.sample(nonsepsis_ids, int(0.85*len(nonsepsis_ids)))
    train_ids = train_pos + train_neg
    test_ids = list(set(range(40336))-set(train_ids))
    train_ids.sort()
    test_ids.sort()
    assert len(train_ids) + len(test_ids) == 40336
    assert set(test_ids).isdisjoint(train_ids) == True
    assert set(train_ids).union(set(test_ids)) == set(range(40336))
    
    test_pos  = [value for value in test_ids  if value in sepsis_ids]
    test_neg  = [value for value in test_ids  if value in nonsepsis_ids]
    print('       neg    pos    total')
    print('train: {}  {}   {}'.format(len(train_neg),len(train_pos),len(train_pos)+len(train_neg)))
    print(' test: {}   {}    {}'.format(len(test_neg),len(test_pos), len(test_pos)+len(test_neg)))

    with open(train_ids_filepath, "w") as f:
        json.dump(train_ids, f)
    with open(test_ids_filepath, "w") as f:
        json.dump(test_ids, f)
        
