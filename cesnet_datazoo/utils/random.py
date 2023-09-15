from enum import IntEnum, auto

import numpy as np

from cesnet_datazoo.config import DatasetConfig


class RandomizedSection(IntEnum):
    INIT_TRAIN_INDICES = auto()
    INIT_VAL_INIDICES = auto()
    INIT_TEST_INDICES = auto()
    DATE_WEIGHT_SAMPLING = auto()
    TRAIN_VAL_SPLIT = auto()
    FIT_STANDARDIZATION_SAMPLE = auto()

def get_fresh_random_generator(dataset_config: DatasetConfig, section: RandomizedSection) -> np.random.RandomState:
    return np.random.RandomState(seed=dataset_config.random_state + 1_000 * dataset_config.fold_id + section.value)
