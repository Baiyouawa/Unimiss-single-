ETT = "electricity_transformer_temperature"
IAQ = "italy_air_quality"

MISSING_LABEL_NONE = 0
MISSING_LABEL_MAR = 1
MISSING_LABEL_MNAR = 2

DATASET_MASK_TYPES = {
    ETT: {"mar", "mnar_t", "mix"},
    IAQ: {"mar", "mnar_x", "mix"},
}

MAIN_MISSING_RATES = {0.2, 0.3, 0.4}
