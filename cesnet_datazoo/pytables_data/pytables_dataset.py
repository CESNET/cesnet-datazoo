import atexit
import logging
import os
import time
import warnings
from datetime import datetime
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import tables as tb
import torch
from numpy.lib.recfunctions import structured_to_unstructured
from torch.utils.data import Dataset
from typing_extensions import assert_never

from cesnet_datazoo.config import (AppSelection, MinTrainSamplesCheck, TestDataParams,
                                   TrainDataParams)
from cesnet_datazoo.constants import (APP_COLUMN, INDICES_DTYPE, INDICES_INDEX_FIELD,
                                      INDICES_TABLE_FIELD, PPI_COLUMN)
from cesnet_datazoo.pytables_data.apps_split import (is_background_app,
                                                     split_apps_topx_with_provider_groups)

log = logging.getLogger(__name__)


class PyTablesDataset(Dataset):
    def __init__(self,
                 database_path: str,
                 tables_paths: list[str],
                 indices: Optional[np.ndarray],
                 tables_app_enum: dict[int, str],
                 tables_cat_enum: dict[int, str],
                 flowstats_features: list[str],
                 flowstats_features_boolean: list[str],
                 flowstats_features_phist: list[str],
                 other_fields: list[str],
                 ppi_channels: list[int],
                 sni_column: Optional[str] = None,
                 ppi_transform: Optional[Callable] = None,
                 flowstats_transform: Optional[Callable] = None,
                 flowstats_phist_transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 return_tensors: bool = False,
                 return_raw_fields: bool = False,
                 preload: bool = False,
                 preload_blob: Optional[str] = None,
                 disabled_apps: Optional[list[str]] = None,):
        self.database_path = database_path
        self.tables_paths = tables_paths
        self.tables = {}
        self.tables_app_enum = tables_app_enum
        self.tables_app_arr = np.array(list(tables_app_enum.values()))
        self.tables_cat_enum = tables_cat_enum
        if indices is None:
            self.set_all_indices(disabled_apps=disabled_apps)
        else:
            self.indices = indices

        self.flowstats_features = flowstats_features
        self.flowstats_features_boolean = flowstats_features_boolean
        self.flowstats_features_phist = flowstats_features_phist
        self.other_fields = other_fields
        self.sni_column = sni_column
        self.ppi_channels = ppi_channels
        self.ppi_transform = ppi_transform
        self.flowstats_transform = flowstats_transform
        self.flowstats_phist_transform = flowstats_phist_transform
        self.target_transform = target_transform
        self.return_tensors = return_tensors
        self.return_raw_fields = return_raw_fields

        self.preload = preload
        self.preload_blob = preload_blob

    def __getitem__(self, batch_idx):
        # log.debug(f"worker {self.worker_id}: __getitem__")
        if self.preload:
            batch_data = self.data[batch_idx]
        else:
            batch_data = load_data_from_tables(tables=self.tables, indices=self.indices[batch_idx], data_dtype=self.data_dtype)
        if self.return_raw_fields:
            return (batch_data, batch_idx)

        # Prepare data
        x_ppi = batch_data[PPI_COLUMN].astype("float32")
        x_ppi = x_ppi[:, self.ppi_channels, :]
        x_flowstats = structured_to_unstructured(batch_data[self.flowstats_features], dtype="float32")
        if self.flowstats_features_boolean:
            x_flowstats_boolean = structured_to_unstructured(batch_data[self.flowstats_features_boolean], dtype="float32")
        else:
            x_flowstats_boolean = np.zeros(shape=(x_flowstats.shape[0], 0), dtype="float32")
        if self.flowstats_features_phist:
            x_flowstats_phist = structured_to_unstructured(batch_data[self.flowstats_features_phist], dtype="float32")
        else:
            x_flowstats_phist = np.zeros(shape=(x_flowstats.shape[0], 0), dtype="float32")
        # Feature transformations
        if self.ppi_transform:
            x_ppi = self.ppi_transform(x_ppi)
        if self.flowstats_transform:
            x_flowstats = self.flowstats_transform(x_flowstats)
        if self.flowstats_phist_transform:
            x_flowstats_phist = self.flowstats_phist_transform(x_flowstats_phist)
        x_flowstats = np.concatenate([x_flowstats, x_flowstats_boolean, x_flowstats_phist], axis=1).astype("float32")
        # Labels transformation
        labels = self.tables_app_arr[batch_data[APP_COLUMN]]
        if self.target_transform:
            labels = self.target_transform(labels)
        # Prepare dataframe with other fields
        other_fields_df = pd.DataFrame(batch_data[self.other_fields]) if len(self.other_fields) > 0 else pd.DataFrame()
        for column in other_fields_df.columns:
            if other_fields_df[column].dtype.kind == "O":
                other_fields_df[column] = other_fields_df[column].astype(str)
            elif column.startswith("TIME_"):
                other_fields_df[column] = other_fields_df[column].map(lambda x: datetime.fromtimestamp(x))

        if self.return_tensors:
            x_ppi = torch.from_numpy(x_ppi)
            x_flowstats = torch.from_numpy(x_flowstats)
            labels = torch.from_numpy(labels).long() # PyTorch loss functions require long type for labels
        return_data = (other_fields_df, x_ppi, x_flowstats, labels)
        return return_data

    def __len__(self):
        return len(self.indices)

    def pytables_worker_init(self, worker_id=0):
        self.worker_id = worker_id
        log.debug(f"Initializing dataloader worker id {self.worker_id}")
        self.database, self.tables = load_database(database_path=self.database_path, tables_paths=self.tables_paths)
        atexit.register(self.cleanup)
        self.data_dtype = self.tables[0].dtype
        if self.preload:
            data = None
            if self.preload_blob and os.path.isfile(self.preload_blob):
                try:
                    data = np.load(self.preload_blob)["data"]
                    log.info(f"Found dataset blob to preload: {self.preload_blob}")
                except:
                    pass # ignore if the file is corrupted (or being written at the moment)
            if data is None:
                data = load_data_from_tables(tables=self.tables, indices=self.indices, data_dtype=self.data_dtype)
            self.data = data
            if self.preload_blob and not os.path.isfile(self.preload_blob):
                np.savez_compressed(self.preload_blob, data=self.data)
        log.debug(f"Finish initialization worker id {self.worker_id}")

    def set_all_indices(self, disabled_apps: Optional[list[str]] = None):
        """
        This should be called from the main process, before dataloader workers split the work.
        Does no filter apps with not enough samples.
        """
        database, tables = load_database(database_path=self.database_path, tables_paths=self.tables_paths)
        inverted_tables_app_enum = {v: k for k, v in self.tables_app_enum.items()}
        disabled_apps_ids = [inverted_tables_app_enum[app] for app in disabled_apps] if disabled_apps is not None else []
        base_labels = {}
        base_indices = {}
        for i in range(len(tables)):
            base_labels[i] = tables[i].read(field=APP_COLUMN)
            base_indices[i] = np.nonzero(np.isin(base_labels[i], disabled_apps_ids, invert=True))[0]
        indices = np.array(list(zip(
            np.concatenate([[table_id] * len(base_indices[table_id]) for table_id in tables]),
            np.concatenate(list(base_indices.values())))), dtype=[field for field in INDICES_DTYPE if field[0] in [INDICES_INDEX_FIELD, INDICES_TABLE_FIELD]])
        self.indices = indices
        database.close()

    def cleanup(self):
        self.database.close()

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info() # type: ignore
    dataset = worker_info.dataset
    dataset.pytables_worker_init(worker_id)

def init_train_indices(train_data_params: TrainDataParams, database_path: str, tables_app_enum: dict[int, str], sni_column: Optional[str], servicemap: pd.DataFrame, rng: np.random.RandomState) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    database, train_tables = load_database(database_path, tables_paths=train_data_params.train_tables_paths)
    inverted_tables_app_enum = {v: k for k, v in tables_app_enum.items()}
    all_labels = {}
    all_sni_domains = {}
    app_counts = pd.Series(dtype="int64")
    start_time = time.time()
    for i, table_path in enumerate(train_data_params.train_tables_paths):
        all_labels[i] = train_tables[i].read(field=APP_COLUMN)
        if sni_column is not None:
            all_sni_domains[i] = train_tables[i].read(field=sni_column)
        else:
            all_sni_domains[i] = np.full_like(all_labels[i], "", dtype="U1")
        log.info(f"Reading app and SNI columns for table {table_path} took {time.time() - start_time:.2f} seconds"); start_time = time.time()
        app_counts = app_counts.add(pd.Series(all_labels[i]).value_counts(), fill_value=0)
    database.close()
    # Handle disabled apps and apps with less than min_samples_per_app samples
    if len(train_data_params.disabled_apps) > 0:
        log.info(f"Disabled applications in dataset config: {sorted(train_data_params.disabled_apps)}")
    disabled_apps_ids = [inverted_tables_app_enum[app] for app in train_data_params.disabled_apps]
    min_samples_apps_ids = set(app_counts[app_counts<train_data_params.min_train_samples_per_app].index.tolist())
    if len(min_samples_apps_ids) > 0:
        min_samples_apps_names = sorted([tables_app_enum[app_id] for app_id in min_samples_apps_ids])
        if train_data_params.min_train_samples_check == MinTrainSamplesCheck.WARN_AND_EXIT:
            warnings.warn(f"Found applications with less than {train_data_params.min_train_samples_per_app} train samples: {min_samples_apps_names}. " +
                            "To disable these applications, add them to config.disabled_apps or set config.min_train_samples_check to disable-apps. To turn off this check, set config.min_train_samples_per_app to zero. Exiting")
            exit()
        elif train_data_params.min_train_samples_check == MinTrainSamplesCheck.DISABLE_APPS:
            log.info(f"Found applications with less than {train_data_params.min_train_samples_per_app} train samples: {min_samples_apps_names}. " +
                       "Disabling these applications")
            disabled_apps_ids.extend(min_samples_apps_ids)
    # Base indices are indices of samples that are not disabled and have enough samples
    base_indices = {}
    for i, table_path in enumerate(train_data_params.train_tables_paths):
        base_indices[i] = np.nonzero(np.isin(all_labels[i], disabled_apps_ids, invert=True))[0]
    base_labels = {table_id: arr[base_indices[table_id]] for table_id, arr in all_labels.items()}
    base_sni_domains = {table_id: arr[base_indices[table_id]] for table_id, arr in all_sni_domains.items()}
    # Apps selection
    if train_data_params.apps_selection != AppSelection.FIXED:
        app_counts = app_counts[[app for app in app_counts.index.tolist() if app not in disabled_apps_ids]]
        app_counts.index = app_counts.index.map(tables_app_enum)
        app_counts = app_counts.sort_values(ascending=False).astype("int64")
        sorted_apps = app_counts.index.to_list()
        if train_data_params.apps_selection == AppSelection.ALL_KNOWN:
            known_apps = [app for app in sorted_apps if not is_background_app(app)]
            unknown_apps = []
        elif train_data_params.apps_selection == AppSelection.TOPX_KNOWN:
            known_apps, unknown_apps = split_apps_topx_with_provider_groups(sorted_apps=sorted_apps, known_count=train_data_params.apps_selection_topx, servicemap=servicemap)
            if len(known_apps) < train_data_params.apps_selection_topx:
                warnings.warn(f"The number of known applications ({len(known_apps)}) is lower than requested in config.apps_selection_topx ({train_data_params.apps_selection_topx}).")
        elif train_data_params.apps_selection == AppSelection.BACKGROUND_UNKNOWN:
                unknown_apps = train_data_params.apps_selection_background_unknown
                known_apps = [app for app in sorted_apps if not (is_background_app(app) or app in unknown_apps)]
        else: assert_never(train_data_params.apps_selection)
        log.info(f"Selected {len(known_apps)} known applications and {len(unknown_apps)} unknown applications")
    else:
        known_apps = train_data_params.apps_selection_fixed_known
        unknown_apps = train_data_params.apps_selection_fixed_unknown
    known_apps = sorted(known_apps)
    unknown_apps = sorted(unknown_apps)
    known_apps_ids = [inverted_tables_app_enum[app] for app in known_apps]
    unknown_apps_ids = [inverted_tables_app_enum[app] for app in unknown_apps]

    train_known_indices, train_unknown_indices = convert_dict_indices(base_indices=base_indices,
                                                                      base_labels=base_labels,
                                                                      base_sni_domains=base_sni_domains,
                                                                      known_apps_ids=known_apps_ids,
                                                                      unknown_apps_ids=unknown_apps_ids)
    rng.shuffle(train_known_indices)
    rng.shuffle(train_unknown_indices)
    return train_known_indices, train_unknown_indices, known_apps, unknown_apps

def init_test_indices(test_data_params: TestDataParams, database_path: str, tables_app_enum: dict[int, str], sni_column: Optional[str], rng: np.random.RandomState) -> tuple[np.ndarray, np.ndarray]:
    database, test_tables = load_database(database_path, tables_paths=test_data_params.test_tables_paths)
    inverted_tables_app_enum = {v: k for k, v in tables_app_enum.items()}
    base_labels = {}
    base_sni_domains = {}
    base_indices = {}
    start_time = time.time()
    for i, table_path in enumerate(test_data_params.test_tables_paths):
        base_labels[i] = test_tables[i].read(field=APP_COLUMN)
        if sni_column is not None:
            base_sni_domains[i] = test_tables[i].read(field=sni_column)
        else:
            base_sni_domains[i] = np.full_like(base_labels[i], "", dtype="U1")
        log.info(f"Reading app and SNI columns for table {table_path} took {time.time() - start_time:.2f} seconds"); start_time = time.time()
        base_indices[i] = np.arange(len(test_tables[i]))
    database.close()
    known_apps_ids = [inverted_tables_app_enum[app] for app in test_data_params.known_apps]
    unknown_apps_ids = [inverted_tables_app_enum[app] for app in test_data_params.unknown_apps]
    test_known_indices, test_unknown_indices = convert_dict_indices(base_indices=base_indices,
                                                                    base_labels=base_labels,
                                                                    base_sni_domains=base_sni_domains,
                                                                    known_apps_ids=known_apps_ids,
                                                                    unknown_apps_ids=unknown_apps_ids)
    rng.shuffle(test_known_indices)
    rng.shuffle(test_unknown_indices)
    log.info(f"Processing indices took {time.time() - start_time:.2f} seconds"); start_time = time.time()
    return test_known_indices, test_unknown_indices

def load_database(database_path: str, tables_paths: Optional[list[str]] = None, mode: str = "r") -> tuple[tb.File, dict[int, Any]]: # dict[int, tb.Table]
    database = tb.open_file(database_path, mode=mode)
    if tables_paths is None:
        tables_paths = list(map(lambda x: x._v_pathname, iter(database.get_node(f"/flows"))))
    tables = {}
    try:
        for i, table_path in enumerate(tables_paths):
            tables[i] = database.get_node(table_path)
    except tb.NoSuchNodeError as e:
        raise e
    return database, tables

def list_all_tables(database_path: str) -> list[str]:
    with tb.open_file(database_path, mode="r") as database:
        return list(map(lambda x: x._v_pathname, iter(database.get_node(f"/flows"))))

def convert_dict_indices(base_indices: dict[int, np.ndarray], base_labels: dict[int, np.ndarray], base_sni_domains: dict[int, np.ndarray], known_apps_ids: list[int], unknown_apps_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    is_known = {table_id: np.isin(table_arr, known_apps_ids) for table_id, table_arr in base_labels.items()}
    is_unknown = {table_id: np.isin(table_arr, unknown_apps_ids) for table_id, table_arr in base_labels.items()}
    known_indices_dict = {table_id: table_arr[is_known[table_id]] for table_id, table_arr in base_indices.items()}
    unknown_indices_dict = {table_id: table_arr[is_unknown[table_id]] for table_id, table_arr in base_indices.items()}
    known_labels_dict = {table_id: table_arr[is_known[table_id]] for table_id, table_arr in base_labels.items()}
    unknown_labels_dict = {table_id: table_arr[is_unknown[table_id]] for table_id, table_arr in base_labels.items()}
    known_sni_domains_dict = {table_id: table_arr[is_known[table_id]] for table_id, table_arr in base_sni_domains.items()}
    unknown_sni_domains_dict = {table_id: table_arr[is_unknown[table_id]] for table_id, table_arr in base_sni_domains.items()}
    known_indices = np.array(list(zip(
        np.concatenate([[table_id] * table_arr.sum() for table_id, table_arr in is_known.items()]),
        np.concatenate(list(known_indices_dict.values())),
        np.concatenate(list(known_labels_dict.values())),
        np.concatenate(list(known_sni_domains_dict.values())))), dtype=INDICES_DTYPE)
    unknown_indices = np.array(list(zip(
        np.concatenate([[table_id] * table_arr.sum() for table_id, table_arr in is_unknown.items()]),
        np.concatenate(list(unknown_indices_dict.values())),
        np.concatenate(list(unknown_labels_dict.values())),
        np.concatenate(list(unknown_sni_domains_dict.values())))), dtype=INDICES_DTYPE)
    return known_indices, unknown_indices

def load_data_from_tables(tables, indices: np.ndarray, data_dtype: np.dtype) -> np.ndarray:
    sorted_indices = indices[indices[INDICES_TABLE_FIELD].argsort(kind="stable")]
    unique_tables, split_bounderies = np.unique(sorted_indices[INDICES_TABLE_FIELD], return_index=True)
    indices_per_table = np.split(sorted_indices, split_bounderies[1:])
    data = np.zeros(len(indices), dtype=data_dtype)
    for table_id, table_indices in zip(unique_tables, indices_per_table):
        data[np.where(indices[INDICES_TABLE_FIELD] == table_id)[0]] = tables[table_id].read_coordinates(table_indices[INDICES_INDEX_FIELD])
    return data
