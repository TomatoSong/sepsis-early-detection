import json
import sys
#sys.path.insert(1, './BioGearPrep')
import random
import os
from config import sepsis_ids_filepath_biogear, biogear_train_ids_filepath, biogear_test_ids_filepath
from BioGearPrep.utils_biogear import get_patient_by_id_original_biogear

def train_test_split_biogear():
    if not (os.path.exists(sepsis_ids_filepath_biogear)):
        print(f"Finding sepsis ids.")
        sepsis_ids = []
        nonsepsis_ids = []
        for pid in range(190):
            p = get_patient_by_id_original_biogear(pid)
            if (p['sepsis'] == 0).all():
                sepsis_ids.append(pid)
            else:
                nonsepsis_ids.append(pid)
        assert len(sepsis_ids) + len(nonsepsis_ids) == 190
        assert set(sepsis_ids + nonsepsis_ids) == set(range(190))
        with open(sepsis_ids_filepath_biogear, "w") as f:
            json.dump(sepsis_ids, f)
        
    with open(sepsis_ids_filepath_biogear, "r") as fp:
        sepsis_ids = json.load(fp)
    nonsepsis_ids = list(set(range(190))-set(sepsis_ids))
    print('sepsis {} nonsepsis {}'.format(len(sepsis_ids), len(nonsepsis_ids)))

    # Train Test Split
    train_pos = random.sample(sepsis_ids, int(0.85*len(sepsis_ids)))
    train_neg = random.sample(nonsepsis_ids, int(0.85*len(nonsepsis_ids)))
    train_ids = train_pos + train_neg
    test_ids = list(set(range(190))-set(train_ids))
    train_ids.sort()
    test_ids.sort()
    assert len(train_ids) + len(test_ids) == 190
    assert set(test_ids).isdisjoint(train_ids) == True
    assert set(train_ids).union(set(test_ids)) == set(range(190))
    
    test_pos  = [value for value in test_ids  if value in sepsis_ids]
    test_neg  = [value for value in test_ids  if value in nonsepsis_ids]
    print('       neg    pos    total')
    print('train: {}  {}   {}'.format(len(train_neg),len(train_pos),len(train_pos)+len(train_neg)))
    print(' test: {}   {}    {}'.format(len(test_neg),len(test_pos), len(test_pos)+len(test_neg)))

    with open(biogear_train_ids_filepath, "w") as f:
        json.dump(train_ids, f)
    with open(biogear_test_ids_filepath, "w") as f:
        json.dump(test_ids, f)

    ### Save label for test set
    dirpath = '../../results_biogear/'
    os.mkdir(dirpath)
    for pid in test_ids:
        p = get_patient_by_id_original_biogear(pid)
        label = p['sepsis']
        filename = dirpath + str(pid) + '_biogear.csv'
        label.to_csv(filename, mode='w+', index=False, header=True, sep='|')