import atexit
import logging
import os
import time
import warnings
from typing import Any, Optional

import numpy as np
import pandas as pd
import tables as tb
import torch
from numpy.lib.recfunctions import drop_fields, structured_to_unstructured
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, LabelEncoder
from torch.utils.data import Dataset
from typing_extensions import assert_never

from cesnet_datazoo.config import (AppsSelection, DatasetConfig, MinTrainSamplesCheck, ScalerEnum,
                                   TestDataParams, TrainDataParams, Scaler)
from cesnet_datazoo.constants import (APP_COLUMN, CATEGORY_COLUMN, DIR_POS, FLOWSTATS_NO_CLIP,
                                      FLOWSTATS_TO_SCALE, IPT_POS, INDICES_INDEX_POS,
                                      INDICES_LABEL_POS, INDICES_TABLE_POS, IP_FEATURES,
                                      PHIST_BIN_COUNT, PHISTS_FEATURES, PPI_COLUMN, SIZE_POS,
                                      UNKNOWN_STR_LABEL)
from cesnet_datazoo.pytables_data.apps_split import (is_background_app,
                                                     split_apps_topx_with_provider_groups)
from cesnet_datazoo.utils.fileutils import pickle_dump, pickle_load
from cesnet_datazoo.utils.random import RandomizedSection, get_fresh_random_generator

log = logging.getLogger(__name__)


class PyTablesDataset(Dataset):
    def __init__(self, database_path: str, tables_paths: list[str], indices: Optional[np.ndarray], flowstats_features: list[str], disabled_apps: Optional[list[str]] = None, preload: bool = False, preload_blob: Optional[str] = None, return_ips: bool = False, return_all_fields: bool = False):
        self.database_path = database_path
        self.tables_paths = tables_paths
        self.tables = {}
        self.flowstats_features = flowstats_features
        self.preload = preload
        self.preload_blob = preload_blob
        self.return_all_fields = return_all_fields
        self.return_ips = return_ips
        if indices is None:
            self.set_all_indices(disabled_apps=disabled_apps)
        else:
            self.indices = indices

    def __getitem__(self, batch_idx):
        # log.debug(f"worker {self.worker_id}: __getitem__")
        if self.preload:
            batch_data = self.data[batch_idx]
        else:
            batch_data = load_data_from_pytables(tables=self.tables, indices=self.indices[batch_idx], data_dtype=self.data_dtype)
        # Assert that app labels are the same between loaded data and indices
        assert np.array_equal(batch_data[APP_COLUMN], self.indices[batch_idx][:, INDICES_LABEL_POS])
        if self.return_all_fields:
            return (batch_data, batch_idx)
        return_data = (batch_data[PPI_COLUMN].astype("float32"), batch_data[self.flowstats_features], list(map(self.app_enum, batch_data[APP_COLUMN])))
        if self.return_ips:
            return_data = (batch_data[IP_FEATURES],) + return_data
        return return_data

    def __len__(self):
        return len(self.indices)

    def pytables_worker_init(self, worker_id=0):
        self.worker_id = worker_id
        log.debug(f"Initializing dataloader worker id {self.worker_id}")
        self.database, self.tables = load_database(database_path=self.database_path, tables_paths=self.tables_paths)
        atexit.register(self.cleanup)
        self.app_enum = self.tables[0].get_enum(APP_COLUMN)
        self.cat_enum = self.tables[0].get_enum(CATEGORY_COLUMN)
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
                data = load_data_from_pytables(tables=self.tables, indices=self.indices, data_dtype=self.data_dtype)
            self.data = data
            if self.preload_blob and not os.path.isfile(self.preload_blob):
                np.savez_compressed(self.preload_blob, data=self.data)
        log.debug(f"Finish initialization worker id {self.worker_id}")

    def get_app_enum(self) -> tb.Enum:
        if self.app_enum:
            return self.app_enum
        database, tables = load_database(database_path=self.database_path, tables_paths=self.tables_paths)
        app_enum = tables[0].get_enum(APP_COLUMN)
        cat_enum = tables[0].get_enum(CATEGORY_COLUMN)
        self.app_enum, self.cat_enum = app_enum, cat_enum
        database.close()
        return app_enum

    def get_cat_enum(self) -> tb.Enum:
        if self.cat_enum:
            return self.cat_enum
        database, tables = load_database(database_path=self.database_path, tables_paths=self.tables_paths)
        app_enum = tables[0].get_enum(APP_COLUMN)
        cat_enum = tables[0].get_enum(CATEGORY_COLUMN)
        self.app_enum, self.cat_enum = app_enum, cat_enum
        database.close()
        return cat_enum

    def set_all_indices(self, disabled_apps: Optional[list[str]] = None):
        """
        This should be called from the main process, before dataloader workers split the work.
        Does no filter apps with not enough samples.
        """
        database, tables = load_database(database_path=self.database_path, tables_paths=self.tables_paths)
        app_enum = tables[0].get_enum(APP_COLUMN)
        disabled_apps_ids = list(map(lambda x: app_enum[x], disabled_apps)) if disabled_apps is not None else []
        base_labels = {}
        base_indices = {}
        for i in range(len(tables)):
            base_labels[i] = tables[i].read(field=APP_COLUMN)
            base_indices[i] = np.nonzero(np.isin(base_labels[i], disabled_apps_ids, invert=True))[0]
        indices = np.column_stack((
            np.concatenate([[table_id] * len(base_indices[table_id]) for table_id in tables]),
            np.concatenate(list(base_indices.values())),
            np.concatenate(list(base_labels.values()))
        )).astype(np.int32)
        self.indices = indices
        database.close()

    def cleanup(self):
        self.database.close()

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info() # type: ignore
    dataset = worker_info.dataset
    dataset.pytables_worker_init(worker_id)

def pytables_ip_collate_fn(batch):
    ips, _, _, apps = batch
    return ips, apps

def pytables_collate_fn(batch: tuple, 
                        flowstats_scaler: Scaler, flowstats_quantiles: pd.Series,
                        psizes_scaler: Scaler, psizes_max: int,
                        ipt_scaler: Scaler, ipt_min: int, ipt_max: int,
                        use_push_flags: bool, use_packet_histograms: bool, zero_ppi_start: int,
                        encoder: LabelEncoder, known_apps: list[str], return_torch: bool = False):
    x_ppi, x_flowstats, labels = batch
    x_ppi = x_ppi.transpose(0, 2, 1)
    orig_shape = x_ppi.shape
    ppi_channels = x_ppi.shape[-1]
    x_ppi = x_ppi.reshape(-1, ppi_channels)
    x_ppi[:, IPT_POS] = x_ppi[:, IPT_POS].clip(max=ipt_max, min=ipt_min)
    x_ppi[:, SIZE_POS] = x_ppi[:, SIZE_POS].clip(max=psizes_max, min=1)
    padding_mask = x_ppi[:, DIR_POS] == 0 # mask of zero padding
    if ipt_scaler:
        x_ppi[:, IPT_POS] = ipt_scaler.transform(x_ppi[:, IPT_POS].reshape(-1, 1)).reshape(-1)
    if psizes_scaler:
        x_ppi[:, SIZE_POS] = psizes_scaler.transform(x_ppi[:, SIZE_POS].reshape(-1, 1)).reshape(-1)
    x_ppi[padding_mask, IPT_POS] = 0
    x_ppi[padding_mask, SIZE_POS] = 0
    x_ppi = x_ppi.reshape(orig_shape).transpose(0, 2, 1)
    if not use_push_flags:
        x_ppi = x_ppi[:, (IPT_POS, DIR_POS, SIZE_POS), :]
    if zero_ppi_start > 0:
        x_ppi[:,:,:zero_ppi_start] = 0

    if use_packet_histograms:
        x_phist = structured_to_unstructured(x_flowstats[PHISTS_FEATURES], dtype="float32")
        src_sizes_pkt_count = x_phist[:, :PHIST_BIN_COUNT].sum(axis=1)[:, np.newaxis]
        dst_sizes_pkt_count = x_phist[:, PHIST_BIN_COUNT:(2*PHIST_BIN_COUNT)].sum(axis=1)[:, np.newaxis]
        np.divide(x_phist[:, :PHIST_BIN_COUNT], src_sizes_pkt_count, out=x_phist[:, :PHIST_BIN_COUNT], where=src_sizes_pkt_count != 0)
        np.divide(x_phist[:, PHIST_BIN_COUNT:(2*PHIST_BIN_COUNT)], dst_sizes_pkt_count, out=x_phist[:, PHIST_BIN_COUNT:(2*PHIST_BIN_COUNT)], where=dst_sizes_pkt_count != 0)
        np.divide(x_phist[:, (2*PHIST_BIN_COUNT):(3*PHIST_BIN_COUNT)], src_sizes_pkt_count - 1, out=x_phist[:, (2*PHIST_BIN_COUNT):(3*PHIST_BIN_COUNT)], where=src_sizes_pkt_count > 1)
        np.divide(x_phist[:, (3*PHIST_BIN_COUNT):(4*PHIST_BIN_COUNT)], dst_sizes_pkt_count - 1, out=x_phist[:, (3*PHIST_BIN_COUNT):(4*PHIST_BIN_COUNT)], where=dst_sizes_pkt_count > 1)
        x_flowstats = structured_to_unstructured(drop_fields(x_flowstats, PHISTS_FEATURES), dtype="float32")
        x_flowstats = np.concatenate([x_flowstats, x_phist], axis=1)
    else:
        x_flowstats = structured_to_unstructured(x_flowstats, dtype="float32")
    np.clip(x_flowstats[:, :len(FLOWSTATS_TO_SCALE)], a_max=flowstats_quantiles, a_min=0, out=x_flowstats[:, :len(FLOWSTATS_TO_SCALE)])
    if flowstats_scaler:
        x_flowstats[:, :len(FLOWSTATS_TO_SCALE)] = flowstats_scaler.transform(x_flowstats[:, :len(FLOWSTATS_TO_SCALE)])

    labels = encoder.transform(np.where(np.isin(labels, known_apps), labels, UNKNOWN_STR_LABEL)).astype("int64") # type: ignore
    if return_torch:
        return torch.from_numpy(x_ppi), torch.from_numpy(x_flowstats), torch.from_numpy(labels)
    return x_ppi, x_flowstats, labels

def init_train_indices(train_data_params: TrainDataParams, servicemap: pd.DataFrame, database_path: str, rng: np.random.RandomState) -> tuple[np.ndarray, np.ndarray, dict[int, str], dict[int, str]]:
    database, train_tables = load_database(database_path, tables_paths=train_data_params.train_tables_paths)
    app_enum = train_tables[0].get_enum(APP_COLUMN)
    all_app_labels = {}
    app_counts = pd.Series(dtype="int64")
    start_time = time.time()
    for i, table_path in enumerate(train_data_params.train_tables_paths):
        all_app_labels[i] = train_tables[i].read(field=APP_COLUMN)
        log.info(f"Reading app column for train table {table_path} took {time.time() - start_time:.2f} seconds"); start_time = time.time()
        app_counts = app_counts.add(pd.Series(all_app_labels[i]).value_counts(), fill_value=0)
    database.close()
    # Handle disabled apps and apps with less than min_samples_per_app samples
    if len(train_data_params.disabled_apps) > 0:
        log.info(f"Disabled applications in dataset config: {sorted(train_data_params.disabled_apps)}")
    disabled_apps_ids = list(map(lambda x: app_enum[x], train_data_params.disabled_apps))
    min_samples_apps_ids = set(app_counts[app_counts<train_data_params.min_train_samples_per_app].index.tolist())
    if len(min_samples_apps_ids) > 0:
        if train_data_params.min_train_samples_check == MinTrainSamplesCheck.WARN_AND_EXIT:
            warnings.warn(f"Found applications with less than {train_data_params.min_train_samples_per_app} train samples: {sorted(map(app_enum, min_samples_apps_ids))}. " +
                            "To disable these applications, add them to config.disabled_apps or set config.min_train_samples_check to disable-apps. To turn off this check, set config.min_train_samples_per_app to zero. Exiting")
            exit()
        elif train_data_params.min_train_samples_check == MinTrainSamplesCheck.DISABLE_APPS:
            log.info(f"Found applications with less than {train_data_params.min_train_samples_per_app} train samples: {sorted(map(app_enum, min_samples_apps_ids))}. " +
                       "Disabling these applications")
            disabled_apps_ids.extend(min_samples_apps_ids)
    # Base indices are indices of samples that are not disabled and have enough samples
    base_indices = {}
    for i, table_path in enumerate(train_data_params.train_tables_paths):
        base_indices[i] = np.nonzero(np.isin(all_app_labels[i], disabled_apps_ids, invert=True))[0]
    base_labels = {table_id: arr[base_indices[table_id]] for table_id, arr in all_app_labels.items()}
    # Apps selection
    if train_data_params.apps_selection != AppsSelection.LONGTERM_FIXED:
        app_counts = app_counts[[app for app in app_counts.index.tolist() if app not in disabled_apps_ids]]
        app_counts.index = app_counts.index.map(app_enum)
        app_counts = app_counts.sort_values(ascending=False).astype("int64")
        sorted_apps = app_counts.index.to_list()
        if train_data_params.apps_selection == AppsSelection.ALL_KNOWN:
            known_apps = [app for app in sorted_apps if not is_background_app(app)]
            unknown_apps = []
        elif train_data_params.apps_selection == AppsSelection.TOPX_KNOWN:
            known_apps, unknown_apps = split_apps_topx_with_provider_groups(sorted_apps=sorted_apps, known_count=train_data_params.apps_selection_topx, servicemap=servicemap)
            if len(known_apps) < train_data_params.apps_selection_topx:
                warnings.warn(f"The number of known applications ({len(known_apps)}) is lower than requested in config.apps_selection_topx ({train_data_params.apps_selection_topx}).")
        elif train_data_params.apps_selection == AppsSelection.EXPLICIT_UNKNOWN:
                unknown_apps = train_data_params.apps_selection_explicit_unknown
                missing_unknown_apps = [app for app in unknown_apps if app not in sorted_apps]
                if len(missing_unknown_apps) > 0:
                    raise ValueError(f"Applications configured in config.apps_selection_explicit_unknown are not present in the dataset (or might be disabled): {sorted(missing_unknown_apps)}")
                known_apps = [app for app in sorted_apps if not (is_background_app(app) or app in unknown_apps)]
        else: assert_never(train_data_params.apps_selection)

        log.info(f"Selected {len(known_apps)} known applications and {len(unknown_apps)} unknown applications")
        known_apps_database_enum: dict[int, str] = {int(app_enum[app]): app for app in known_apps}
        unknown_apps_database_enum: dict[int, str] = {int(app_enum[app]): app for app in unknown_apps}
    else:
        assert train_data_params.apps_selection_fixed_longterm is not None
        known_apps_database_enum, unknown_apps_database_enum = train_data_params.apps_selection_fixed_longterm
    known_apps_ids = list(known_apps_database_enum)
    unknown_apps_ids = list(unknown_apps_database_enum)

    train_known_indices, train_unknown_indices = convert_dict_indices(base_indices=base_indices, base_labels=base_labels, known_apps_ids=known_apps_ids, unknown_apps_ids=unknown_apps_ids)
    rng.shuffle(train_known_indices)
    rng.shuffle(train_unknown_indices)
    log.info(f"Processing train indices took {time.time() - start_time:.2f} seconds"); start_time = time.time()
    return train_known_indices, train_unknown_indices, known_apps_database_enum, unknown_apps_database_enum

def init_test_indices(test_data_params: TestDataParams, database_path: str, rng: np.random.RandomState) -> tuple[np.ndarray, np.ndarray]:
    database, test_tables = load_database(database_path, tables_paths=test_data_params.test_tables_paths)
    base_labels = {}
    base_indices = {}
    start_time = time.time()
    for i, table_path in enumerate(test_data_params.test_tables_paths):
        base_labels[i] = test_tables[i].read(field=APP_COLUMN)
        log.info(f"Reading app column for test table {table_path} took {time.time() - start_time:.2f} seconds"); start_time = time.time()
        base_indices[i] = np.arange(len(test_tables[i]))
    database.close()
    known_apps_ids = list(test_data_params.known_apps_database_enum)
    unknown_apps_ids = list(test_data_params.unknown_apps_database_enum)
    test_known_indices, test_unknown_indices = convert_dict_indices(base_indices=base_indices, base_labels=base_labels, known_apps_ids=known_apps_ids, unknown_apps_ids=unknown_apps_ids)
    rng.shuffle(test_known_indices)
    rng.shuffle(test_unknown_indices)
    log.info(f"Processing test indices took {time.time() - start_time:.2f} seconds"); start_time = time.time()
    return test_known_indices, test_unknown_indices

def fit_or_load_scalers(dataset_config: DatasetConfig, train_indices: np.ndarray) -> tuple[Scaler, pd.Series, Scaler, Scaler]:
    train_data_path = dataset_config._get_train_data_path()
    flowstats_scaler_path = os.path.join(train_data_path, "stand", f"flowstats_scaler-{dataset_config.flowstats_scaler}-q{dataset_config.flowstats_clip}.pickle")
    flowstats_quantiles_path = os.path.join(train_data_path, "stand", f"flowstats_quantiles-q{dataset_config.flowstats_clip}.pickle")
    ipt_scaler_path = os.path.join(train_data_path, "stand", f"ipt_scaler-{dataset_config.ipt_scaler}-ipt_min{dataset_config.ipt_min}-ipt_max{dataset_config.ipt_max}.pickle")
    psizes_sizes_scaler_path = os.path.join(train_data_path, "stand", f"psizes_scaler-{dataset_config.psizes_scaler}-psizes_max{dataset_config.psizes_max}.pickle")
    if os.path.isfile(flowstats_scaler_path) and os.path.isfile(flowstats_quantiles_path) and os.path.isfile(ipt_scaler_path) and os.path.isfile(psizes_sizes_scaler_path):
        flowstats_scaler = pickle_load(flowstats_scaler_path)
        flowstats_quantiles = pickle_load(flowstats_quantiles_path)
        ipt_scaler = pickle_load(ipt_scaler_path)
        psizes_scaler = pickle_load(psizes_sizes_scaler_path)
    else:
        if dataset_config.flowstats_scaler == ScalerEnum.ROBUST:
            flowstats_scaler = RobustScaler()
        elif dataset_config.flowstats_scaler == ScalerEnum.STANDARD:
            flowstats_scaler = StandardScaler()
        elif dataset_config.flowstats_scaler == ScalerEnum.MINMAX:
            flowstats_scaler = MinMaxScaler()
        elif dataset_config.flowstats_scaler == ScalerEnum.NO_SCALER:
            flowstats_scaler = None
        else: assert_never(dataset_config.flowstats_scaler)

        if dataset_config.ipt_scaler == ScalerEnum.ROBUST:
            ipt_scaler = RobustScaler()
        elif dataset_config.ipt_scaler == ScalerEnum.STANDARD:
            ipt_scaler = StandardScaler()
        elif dataset_config.ipt_scaler == ScalerEnum.MINMAX:
            ipt_scaler = MinMaxScaler()
        elif dataset_config.ipt_scaler == ScalerEnum.NO_SCALER:
            ipt_scaler = None
        else: assert_never(dataset_config.ipt_scaler)

        if dataset_config.psizes_scaler == ScalerEnum.ROBUST:
            psizes_scaler = RobustScaler()
        elif dataset_config.psizes_scaler == ScalerEnum.STANDARD:
            psizes_scaler = StandardScaler()
        elif dataset_config.psizes_scaler == ScalerEnum.MINMAX:
            psizes_scaler = MinMaxScaler()
        elif dataset_config.psizes_scaler == ScalerEnum.NO_SCALER:
            psizes_scaler = None
        else: assert_never(dataset_config.psizes_scaler)

        if isinstance(dataset_config.fit_scalers_samples, int) and dataset_config.fit_scalers_samples > len(train_indices):
            warnings.warn(f"The number of samples for fitting scalers ({dataset_config.fit_scalers_samples}) is larger than the number of train samples ({len(train_indices)}), using the number of train samples instead")
            dataset_config.fit_scalers_samples = len(train_indices)
        fit_scalers_rng = get_fresh_random_generator(dataset_config=dataset_config, section=RandomizedSection.FIT_SCALERS_SAMPLE)
        if isinstance(dataset_config.fit_scalers_samples, float):
            num_samples = int(dataset_config.fit_scalers_samples * len(train_indices))
        else:
            num_samples = dataset_config.fit_scalers_samples
        fit_scalers_indices = train_indices[fit_scalers_rng.choice(len(train_indices), size=num_samples, replace=False)]
        flowstats_quantiles = fit_scalers(
            database_path=dataset_config.database_path,
            train_tables_paths=dataset_config._get_train_tables_paths(),
            fit_scalers_indices=fit_scalers_indices,
            flowstats_scaler=flowstats_scaler,
            flowstats_quantile_clip=dataset_config.flowstats_clip,
            ipt_scaler=ipt_scaler,
            psizes_scaler=psizes_scaler,
            ipt_min=dataset_config.ipt_min,
            ipt_max=dataset_config.ipt_max,
            psizes_max=dataset_config.psizes_max)
        pickle_dump(flowstats_scaler, flowstats_scaler_path)
        pickle_dump(flowstats_quantiles, flowstats_quantiles_path)
        pickle_dump(ipt_scaler, ipt_scaler_path)
        pickle_dump(psizes_scaler, psizes_sizes_scaler_path)
    return flowstats_scaler, flowstats_quantiles, ipt_scaler, psizes_scaler

def fit_scalers(database_path: str, train_tables_paths: list[str], fit_scalers_indices: np.ndarray, flowstats_scaler, flowstats_quantile_clip: float, ipt_scaler, psizes_scaler, ipt_min: int, ipt_max: int, psizes_max: int) -> pd.Series:
    start_time = time.time()
    database, tables = load_database(database_path, tables_paths=train_tables_paths)
    data = load_data_from_pytables(tables=tables, indices=fit_scalers_indices, data_dtype=tables[0].dtype)
    database.close()
    # PPI
    data_ppi = data[PPI_COLUMN].astype("float32")
    ppi_channels = data_ppi.shape[1]
    data_ppi = data_ppi.transpose(0, 2, 1).reshape(-1, ppi_channels)
    padding_mask = data_ppi[:, DIR_POS] == 0 # mask of padded packets
    if ipt_scaler:
        train_ipt = data_ppi[:, IPT_POS].clip(max=ipt_max, min=ipt_min)
        train_ipt[padding_mask] = np.nan # nans are ignored in sklearn scalers
        if isinstance(ipt_scaler, MinMaxScaler):
            # let zero be the minimum for minmax scaling
            train_ipt = np.concatenate((train_ipt, [0]))
        ipt_scaler.fit(train_ipt.reshape(-1, 1))
    if psizes_scaler:
        train_psizes = data_ppi[:, SIZE_POS].clip(max=psizes_max, min=1)
        train_psizes[padding_mask] = np.nan
        if isinstance(psizes_scaler, MinMaxScaler):
            train_psizes = np.concatenate((train_psizes, [0]))
        psizes_scaler.fit(train_psizes.reshape(-1, 1))
    # FLOWSTATS
    train_flowstats = pd.DataFrame(data, columns=FLOWSTATS_TO_SCALE)
    upper_quantiles = train_flowstats.quantile(flowstats_quantile_clip)
    upper_quantiles[FLOWSTATS_NO_CLIP] = np.Inf # disable clipping for features with "fixed" range
    if flowstats_scaler:
        train_flowstats = train_flowstats.clip(upper=upper_quantiles, lower=0, axis=1).to_numpy() # type: ignore
        flowstats_scaler.fit(train_flowstats)
    log.info(f"Reading data and fitting scalers took {time.time() - start_time:.2f} seconds")
    return upper_quantiles

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

def convert_dict_indices(base_indices: dict[int, np.ndarray], base_labels: dict[int, np.ndarray], known_apps_ids: list[int], unknown_apps_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    is_known = {table_id: np.isin(table_arr, known_apps_ids) for table_id, table_arr in base_labels.items()}
    is_unknown = {table_id: np.isin(table_arr, unknown_apps_ids) for table_id, table_arr in base_labels.items()}
    known_indices_dict = {table_id: table_arr[is_known[table_id]] for table_id, table_arr in base_indices.items()}
    unknown_indices_dict = {table_id: table_arr[is_unknown[table_id]] for table_id, table_arr in base_indices.items()}
    known_labels_dict = {table_id: table_arr[is_known[table_id]] for table_id, table_arr in base_labels.items()}
    unknown_labels_dict = {table_id: table_arr[is_unknown[table_id]] for table_id, table_arr in base_labels.items()}
    known_indices = np.column_stack((
        np.concatenate([[table_id] * table_arr.sum() for table_id, table_arr in is_known.items()]),
        np.concatenate(list(known_indices_dict.values())),
        np.concatenate(list(known_labels_dict.values()))))
    unknown_indices = np.column_stack((
        np.concatenate([[table_id] * table_arr.sum() for table_id, table_arr in is_unknown.items()]),
        np.concatenate(list(unknown_indices_dict.values())),
        np.concatenate(list(unknown_labels_dict.values()))))
    return known_indices, unknown_indices

def load_data_from_pytables(tables, indices: np.ndarray, data_dtype: np.dtype) -> np.ndarray:
    sorted_indices = indices[indices[:, INDICES_TABLE_POS].argsort(kind="stable")]
    unique_tables, split_bounderies = np.unique(sorted_indices[:, INDICES_TABLE_POS], return_index=True)
    indices_per_table = np.split(sorted_indices, split_bounderies[1:])
    data = np.empty(len(indices), dtype=data_dtype)
    for table_id, table_indices in zip(unique_tables, indices_per_table):
        data[np.where(indices[:, INDICES_TABLE_POS] == table_id)[0]] = tables[table_id].read_coordinates(table_indices[:, INDICES_INDEX_POS])
    return data
