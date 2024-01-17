SEQ_LEN    = 72
SEQ_OFFSET = 6
BATCH_SIZE = 1024
LR         = 0.001
EPOCHS     = 75
INPUT_DIM  = 37
HIDDEN_DIM = 64
FF_DIM     = 1024
N_CLASSES  = 2
N_HEADS    = 16
N_LAYERS   = 8

ORGAN_HIDDEN_DIM = 16
ORGAN_N_HEADS    = 4
ORGAN_N_LAYERS   = 4
ORGAN_FF_DIM     = 128

train_ids_filepath = "train_ids.json"
test_ids_filepath = "test_ids.json"

preprocess_method = "standardized_padded"
default_model = "../models/happy-cloud-13311_18_2023_12_15_49_44_0.31252.pth"

padding_offset = 100

clinical_measurements_dict = {
    "Cardiovascular": ["HR", "SBP", "MAP", "DBP", "TroponinI"],
    "Respiratory": ["O2Sat", "Resp", "EtCO2", "FiO2", "PaCO2", "SaO2", "pH"],
    "Immune": ["Temp", "WBC"],
    "Coagulation": ["Platelets", "PTT", "Fibrinogen"],
    "Renal": ["BUN", "Creatinine", "BaseExcess", "HCO3", "pH", "Chloride", "Magnesium", "Phosphate", "Potassium"],
    "Liver": ["AST", "Alkalinephos", "Bilirubin_direct", "Bilirubin_total"],
    "Hematologic": ["Hct", "Hgb", "PTT", "WBC", "Fibrinogen", "Platelets"],
    "Metabolism": ["BaseExcess", "HCO3", "pH", "Glucose", "Lactate", "Calcium"],
    "Demographic": ["Age", "Gender", "ICULOS"]
}

COLS = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
       'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
       'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine',
       'Bilirubin_direct', 'Glucose', 'Lactate', 'Magnesium', 'Phosphate',
       'Potassium', 'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT',
       'WBC', 'Fibrinogen', 'Platelets', 'Age', 'Gender', 'ICULOS']


organ_system_ids = {
    'Cardiovascular': [0, 3, 4, 5, 27],
    'Respiratory': [1, 6, 7, 10, 12, 13, 11],
    'Immune': [2, 31],
    'Coagulation': [33, 30, 32],
    'Renal': [15, 19, 8, 9, 11, 18, 23, 24, 25],
    'Liver': [14, 16, 20, 26],
    'Hematologic': [28, 29, 30, 31, 32, 33],
    'Metabolism': [8, 9, 11, 21, 22, 17],
    "Demographic": [34, 35, 36]
}