import dataclasses
import logging
import os
import time
import warnings
from collections import namedtuple
from enum import Enum

import numpy as np
import pandas as pd

from cesnet_datazoo.config import DatasetConfig
from cesnet_datazoo.constants import (INDICES_APP_FIELD, INDICES_DTYPE, INDICES_INDEX_FIELD,
                                      INDICES_TABLE_FIELD)
from cesnet_datazoo.pytables_data.pytables_dataset import init_test_indices, init_train_indices
from cesnet_datazoo.utils.fileutils import yaml_dump, yaml_load
from cesnet_datazoo.utils.random import RandomizedSection, get_fresh_random_generator

log = logging.getLogger(__name__)
TRAIN_DATA_PARAMS_FILE = "train_data_params.yaml"
TEST_DATA_PARAMS_FILE = "test_data_params.yaml"
IndicesTuple = namedtuple("IndicesTuple", ["train_indices", "val_known_indices", "val_unknown_indices", "test_known_indices", "test_unknown_indices"])


def sort_indices(indices: np.ndarray) -> np.ndarray:
    idxs = np.argsort(indices[INDICES_INDEX_FIELD])
    res = idxs[np.argsort(indices[idxs][INDICES_TABLE_FIELD], kind="stable")]
    return indices[res]

def subset_and_sort_indices(dataset_config: DatasetConfig, dataset_indices: IndicesTuple) -> IndicesTuple:
    if dataset_config.train_size == "all":
        dataset_config.train_size = len(dataset_indices.train_indices)
    if dataset_config.val_known_size == "all":
        dataset_config.val_known_size = len(dataset_indices.val_known_indices)
    if dataset_config.val_unknown_size == "all":
        dataset_config.val_unknown_size = len(dataset_indices.val_unknown_indices)
    if dataset_config.test_known_size == "all":
        dataset_config.test_known_size = len(dataset_indices.test_known_indices)
    if dataset_config.test_unknown_size == "all":
        dataset_config.test_unknown_size = len(dataset_indices.test_unknown_indices)
    train_indices = sort_indices(dataset_indices.train_indices[:dataset_config.train_size])
    val_known_indices = sort_indices(dataset_indices.val_known_indices[:dataset_config.val_known_size])
    val_unknown_indices = sort_indices(dataset_indices.val_unknown_indices[:dataset_config.val_unknown_size])
    test_known_indices = sort_indices(dataset_indices.test_known_indices[:dataset_config.test_known_size])
    test_unknown_indices = sort_indices(dataset_indices.test_unknown_indices[:dataset_config.test_unknown_size])
    if dataset_config.train_size != len(train_indices):
        warnings.warn(f"Requested train size {dataset_config.train_size} is larger than the number of available samples {len(train_indices)}.")
        dataset_config.train_size = len(train_indices)
    if dataset_config.val_known_size != len(val_known_indices):
        warnings.warn(f"Requested validation known size {dataset_config.val_known_size} is larger than the number of available samples {len(val_known_indices)}.")
        dataset_config.val_known_size = len(val_known_indices)
    if dataset_config.val_unknown_size != len(val_unknown_indices):
        warnings.warn(f"Requested validation unknown size {dataset_config.val_unknown_size} is larger than the number of available samples {len(val_unknown_indices)}.")
        dataset_config.val_unknown_size = len(val_unknown_indices)
    if dataset_config.test_known_size != len(test_known_indices):
        warnings.warn(f"Requested test known size {dataset_config.test_known_size} is larger than the number of available samples {len(test_known_indices)}.")
        dataset_config.test_known_size = len(test_known_indices)
    if dataset_config.test_unknown_size != len(test_unknown_indices):
        warnings.warn(f"Requested test unknown size {dataset_config.test_unknown_size} is larger than the number of available samples {len(test_unknown_indices)}.")
        dataset_config.test_unknown_size = len(test_unknown_indices)
    dataset_indices = IndicesTuple(train_indices=train_indices, val_known_indices=val_known_indices, val_unknown_indices=val_unknown_indices, test_known_indices=test_known_indices, test_unknown_indices=test_unknown_indices)
    return dataset_indices

def date_weight_sample_train_indices(dataset_config: DatasetConfig, train_indices: np.ndarray, num_samples: int) -> np.ndarray:
    rng = get_fresh_random_generator(dataset_config=dataset_config, section=RandomizedSection.DATE_WEIGHT_SAMPLING)
    indices_per_date = [train_indices[train_indices[INDICES_TABLE_FIELD] == i] for i in np.unique(train_indices[INDICES_TABLE_FIELD])]
    weights = np.array(dataset_config.train_dates_weights)
    weights = weights / weights.sum()
    samples_per_date = np.ceil((weights * (num_samples))).astype(int)
    samples_per_date_clipped = np.clip(samples_per_date, a_max=list(map(len, indices_per_date)), a_min=0)
    df = pd.DataFrame(data={"Dates": dataset_config.train_dates, "Weights": dataset_config.train_dates_weights, "Requested Samples": samples_per_date, "Available Samples": samples_per_date_clipped})
    log.info(f"Weight sampling per date with requsted total number of samples {num_samples} (train_size + val_known_size when using the split-from-train validation approach; train_size otherwise)")
    for l in df.to_string(index=False).splitlines():
        log.info(l)
    if not all(samples_per_date == samples_per_date_clipped):
        warnings.warn("Some dates have not enough samples, resulting train size will be smaller")
    sampled_indicies_per_date = [indices[rng.choice(len(indices), size=n, replace=False)] for indices, n in zip(indices_per_date, samples_per_date_clipped)]
    sampled_train_indices = np.concatenate(sampled_indicies_per_date)
    return sampled_train_indices

def indices_to_app_counts(indices: np.ndarray, tables_app_enum: dict[int, str]) -> pd.Series:
    app_counts = pd.Series(indices[INDICES_APP_FIELD]).value_counts()
    app_counts.index = app_counts.index.map(lambda x: tables_app_enum[x])
    return app_counts

def compute_known_app_counts(dataset_indices: IndicesTuple, tables_app_enum: dict[int, str]) -> pd.DataFrame:
    train_app_counts = indices_to_app_counts(dataset_indices.train_indices, tables_app_enum)
    val_known_app_counts = indices_to_app_counts(dataset_indices.val_known_indices, tables_app_enum)
    test_known_app_counts = indices_to_app_counts(dataset_indices.test_known_indices, tables_app_enum)
    df = pd.DataFrame(data={"Train": train_app_counts, "Validation": val_known_app_counts, "Test": test_known_app_counts}).fillna(0).astype("int64")
    return df

def compute_unknown_app_counts(dataset_indices: IndicesTuple, tables_app_enum: dict[int, str]) -> pd.DataFrame:
    val_unknown_app_counts = indices_to_app_counts(dataset_indices.val_unknown_indices, tables_app_enum)
    test_unknown_app_counts = indices_to_app_counts(dataset_indices.test_unknown_indices, tables_app_enum)
    df = pd.DataFrame(data={"Validation": val_unknown_app_counts, "Test": test_unknown_app_counts}).fillna(0).astype("int64")
    return df

def init_or_load_train_indices(dataset_config: DatasetConfig, tables_app_enum: dict[int, str], servicemap: pd.DataFrame, disable_indices_cache: bool) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    train_data_path = dataset_config._get_train_data_path()
    init_train_data(train_data_path)
    if not os.path.isfile(os.path.join(train_data_path, TRAIN_DATA_PARAMS_FILE)):
        log.info("Processing train indices")
        start_time = time.time()
        train_data_params = dataset_config._get_train_data_params()
        train_known_indices, train_unknown_indices, known_apps, unknown_apps = init_train_indices(train_data_params=train_data_params,
                                                                                                  database_path=dataset_config.database_path,
                                                                                                  tables_app_enum=tables_app_enum,
                                                                                                  sni_column=dataset_config.sni_column,
                                                                                                  servicemap=servicemap,
                                                                                                  rng=get_fresh_random_generator(dataset_config=dataset_config, section=RandomizedSection.INIT_TRAIN_INDICES))
        if not disable_indices_cache:
            yaml_dump({k: str(v) if isinstance(v, Enum) else list(v) if isinstance(v, tuple) else v for k, v in dataclasses.asdict(train_data_params).items()}, os.path.join(train_data_path, TRAIN_DATA_PARAMS_FILE))
            yaml_dump(known_apps, os.path.join(train_data_path, "known_apps.yaml"))
            yaml_dump(unknown_apps, os.path.join(train_data_path, "unknown_apps.yaml"))
            np.savez_compressed(os.path.join(train_data_path, "train_indices.npz"), train_known_indices=train_known_indices, train_unknown_indices=train_unknown_indices)
        log.info(f"Processing indices took {time.time() - start_time:.2f} seconds")
    else:
        known_apps = yaml_load(os.path.join(train_data_path, "known_apps.yaml"))
        unknown_apps = yaml_load(os.path.join(train_data_path, "unknown_apps.yaml"))
        loaded = np.load(os.path.join(train_data_path, "train_indices.npz"))
        train_known_indices = loaded["train_known_indices"]
        train_unknown_indices = loaded["train_unknown_indices"]
    return train_known_indices, train_unknown_indices, known_apps, unknown_apps

def init_or_load_val_indices(dataset_config: DatasetConfig, known_apps: list[str], unknown_apps: list[str], tables_app_enum: dict[int, str], disable_indices_cache: bool) -> tuple[np.ndarray, np.ndarray, str]:
    val_data_params, val_data_path = dataset_config._get_val_data_params_and_path(known_apps=known_apps, unknown_apps=unknown_apps)
    init_test_data(val_data_path)
    if not os.path.isfile(os.path.join(val_data_path, TEST_DATA_PARAMS_FILE)):
        log.info("Processing validation indices")
        start_time = time.time()
        val_known_indices, val_unknown_indices = init_test_indices(test_data_params=val_data_params,
                                                                   database_path=dataset_config.database_path,
                                                                   tables_app_enum=tables_app_enum,
                                                                   sni_column=dataset_config.sni_column,
                                                                   rng=get_fresh_random_generator(dataset_config=dataset_config, section=RandomizedSection.INIT_VAL_INIDICES))
        if not disable_indices_cache:
            yaml_dump(dataclasses.asdict(val_data_params), os.path.join(val_data_path, TEST_DATA_PARAMS_FILE))
            np.savez_compressed(os.path.join(val_data_path, "val_indices.npz"), val_known_indices=val_known_indices, val_unknown_indices=val_unknown_indices)
        log.info(f"Processing indices took {time.time() - start_time:.2f} seconds")
    else:
        loaded = np.load(os.path.join(val_data_path, "val_indices.npz"))
        val_known_indices = loaded["val_known_indices"]
        val_unknown_indices = loaded["val_unknown_indices"]
    return val_known_indices, val_unknown_indices, val_data_path

def init_or_load_test_indices(dataset_config: DatasetConfig, known_apps: list[str], unknown_apps: list[str], tables_app_enum: dict[int, str], disable_indices_cache: bool) -> tuple[np.ndarray, np.ndarray, str]:
    test_data_params, test_data_path = dataset_config._get_test_data_params_and_path(known_apps=known_apps, unknown_apps=unknown_apps)
    init_test_data(test_data_path)
    if not os.path.isfile(os.path.join(test_data_path, TEST_DATA_PARAMS_FILE)):
        log.info("Processing test indices")
        start_time = time.time()
        test_known_indices, test_unknown_indices = init_test_indices(test_data_params=test_data_params,
                                                                     database_path=dataset_config.database_path,
                                                                     tables_app_enum=tables_app_enum,
                                                                     sni_column=dataset_config.sni_column,
                                                                     rng=get_fresh_random_generator(dataset_config=dataset_config, section=RandomizedSection.INIT_TEST_INDICES))
        if not disable_indices_cache:
            yaml_dump(dataclasses.asdict(test_data_params), os.path.join(test_data_path, TEST_DATA_PARAMS_FILE))
            np.savez_compressed(os.path.join(test_data_path, "test_indices.npz"), test_known_indices=test_known_indices, test_unknown_indices=test_unknown_indices)
        log.info(f"Processing indices took {time.time() - start_time:.2f} seconds")
    else:
        loaded = np.load(os.path.join(test_data_path, "test_indices.npz"))
        test_known_indices = loaded["test_known_indices"]
        test_unknown_indices = loaded["test_unknown_indices"]
    return test_known_indices, test_unknown_indices, test_data_path

def init_train_data(train_data_path: str):
    os.makedirs(train_data_path, exist_ok=True)
    os.makedirs(os.path.join(train_data_path, "transforms"), exist_ok=True)
    os.makedirs(os.path.join(train_data_path, "preload"), exist_ok=True)

def init_test_data(test_data_path: str):
    os.makedirs(test_data_path, exist_ok=True)
    os.makedirs(os.path.join(test_data_path, "preload"), exist_ok=True)

def no_indices() -> np.ndarray:
    return np.empty(shape=(0,), dtype=INDICES_DTYPE)
