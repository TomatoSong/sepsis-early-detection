import json
import random

def train_test_split():
    with open("../data/sepsis_indices.json", "r") as fp:
        sepsis_ids = json.load(fp)
    nonsepsis_ids = list(set(range(40336))-set(sepsis_ids))
    print('sepsis {} nonsepsis {}'.format(len(sepsis_ids), len(nonsepsis_ids)))

    # Train Test Split
    train_ids = random.sample(sepsis_ids, int(0.85*len(sepsis_ids)))
    train_ids += random.sample(nonsepsis_ids, int(0.85*len(nonsepsis_ids)))
    test_ids = list(set(range(40336))-set(train_ids))
    train_ids.sort()
    test_ids.sort()
    assert len(train_ids) + len(test_ids) == 40336
    assert set(test_ids).isdisjoint(train_ids) == True
    assert set(train_ids).union(set(test_ids)) == set(range(40336))

    with open("train_ids.json", "w") as f:
        json.dump(train_ids, f)
    with open("test_ids.json", "w") as f:
        json.dump(test_ids, f)
