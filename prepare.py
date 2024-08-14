import json
import random
import os
from config import sepsis_ids_filepath, train_ids_filepath, test_ids_filepath, pos_pos_idxmap_prefix
from utils import get_patient_by_id_original, get_patient_by_id_standardized

def train_test_split():
    if not (os.path.exists(sepsis_ids_filepath)):
        print(f"Finding sepsis ids.")
        sepsis_ids = []
        nonsepsis_ids = []
        for pid in range(40336):
            p = get_patient_by_id_original(pid)
            if not (p['SepsisLabel'] == 0).all():
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

    ### Save label for test set
    dirpath = '../results/labels/'
    os.mkdir(dirpath)
    for pid in test_ids:
        p = get_patient_by_id_original(pid)
        label = p['SepsisLabel']
        filename = dirpath + '/p' + str(pid).zfill(6) + '.psv'
        label.to_csv(filename, mode='w+', index=False, header=True, sep='|')


def find_train_pos():
    with open(sepsis_ids_filepath, "r") as fp:
        sepsis_ids = json.load(fp)

    with open(train_ids_filepath, "r") as f:
        train_ids = json.load(f)

    train_pos = [i for i in train_ids if i in sepsis_ids]
    with open(train_pos_filepath, "w") as f:
        json.dump(train_pos, f)


def build_pos_pos_idxmap(seq_len, starting_offset):
    with open(sepsis_ids_filepath, "r") as fp:
        sepsis_ids = json.load(fp)

    fname = pos_pos_idxmap_prefix + str(seq_len) + '_' + str(starting_offset)
    if os.path.exists(fname):
      return

    idxmap = []
    for pid in sepsis_ids:
        p = get_patient_by_id_standardized(pid)
        sepsis_time_ad6 = p['SepsisLabel'].idxmax()
        for t in range(sepsis_time_ad6, len(p)):
            u_weights = [p.loc[t, 'UtilityNeg'], p.loc[t, 'UtilityPos']]
            start = t-seq_len+1
            padding = seq_len-t-1 if t < seq_len else 0
            if start < 0:
                continue
            hist = (
                pid,
                start,
                t,
                1,
                u_weights,
                padding
            )
            assert hist[2] < len(p)
            assert hist[2]-hist[1]+1+hist[5] == seq_len
            idxmap.append(hist)
    print('populated {} patients into {} timeseries'.format(len(sepsis_ids), len(idxmap)))
    with open(fname, "w") as fp:
        json.dump(idxmap, fp)
    
