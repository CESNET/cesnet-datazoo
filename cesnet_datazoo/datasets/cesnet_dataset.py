
import calendar
import datetime
import os
import warnings
from functools import partial
from typing import Callable, Literal, Optional

import numpy as np
import pandas as pd
import tables as tb
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, Sampler, SequentialSampler
from typing_extensions import assert_never

from cesnet_datazoo.config import AppSelection, DataLoaderOrder, DatasetConfig, ValidationApproach
from cesnet_datazoo.constants import (APP_COLUMN, CATEGORY_COLUMN, DATASET_SIZES, INDICES_LABEL_POS,
                                      SERVICEMAP_FILE, UNKNOWN_STR_LABEL)
from cesnet_datazoo.datasets.loaders import collate_fn_simple, create_df_from_dataloader
from cesnet_datazoo.datasets.metadata.dataset_metadata import DatasetMetadata, load_metadata
from cesnet_datazoo.datasets.statistics import compute_dataset_statistics
from cesnet_datazoo.pytables_data.apps_split import is_background_app
from cesnet_datazoo.pytables_data.data_scalers import fit_scalers
from cesnet_datazoo.pytables_data.indices_setup import (IndicesTuple, compute_known_app_counts,
                                                        compute_unknown_app_counts,
                                                        date_weight_sample_train_indices,
                                                        init_or_load_test_indices,
                                                        init_or_load_train_indices,
                                                        init_or_load_val_indices,
                                                        subset_and_sort_indices)
from cesnet_datazoo.pytables_data.pytables_dataset import PyTablesDataset, worker_init_fn
from cesnet_datazoo.utils.class_info import ClassInfo, create_class_info
from cesnet_datazoo.utils.download import resumable_download, simple_download
from cesnet_datazoo.utils.random import RandomizedSection, get_fresh_random_generator

DATAFRAME_SAMPLES_WARNING_THRESHOLD = 20_000_000


class CesnetDataset():
    """
    The main class for accessing CESNET datasets. It handles downloading, train/validation/test splitting, and class selection. Access to data is provided through:

    - Iterable PyTorch DataLoader for batch processing. See [using dataloaders][using-dataloaders] for more details.
    - Pandas DataFrame for loading the entire train, validation, or test set at once.

    The dataset is stored in a [PyTables](https://www.pytables.org/) database. The internal `PyTablesDataset` class is used as a wrapper
    that implements the PyTorch [`Dataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) interface
    and is compatible with [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader),
    which provides efficient parallel loading of the data. The dataset configuration is done through the [`DatasetConfig`][config.DatasetConfig] class.

    **Intended usage:**

    1. Create an instance of the [dataset class][dataset-classes] with the desired size and data root. This will download the dataset if it has not already been downloaded.
    2. Create an instance of [`DatasetConfig`][config.DatasetConfig] and set it with [`set_dataset_config_and_initialize`][datasets.cesnet_dataset.CesnetDataset.set_dataset_config_and_initialize].
    This will initialize the dataset — select classes, split data into train/validation/test sets, and fit data scalers if needed. All is done according to the provided configuration and is cached for later use.
    3. Use [`get_train_dataloader`][datasets.cesnet_dataset.CesnetDataset.get_train_dataloader] or [`get_train_df`][datasets.cesnet_dataset.CesnetDataset.get_train_df] to get training data for a classification model.
    4. Validate the model and perform the hyperparameter optimalization on [`get_val_dataloader`][datasets.cesnet_dataset.CesnetDataset.get_val_dataloader] or [`get_val_df`][datasets.cesnet_dataset.CesnetDataset.get_val_df].
    5. Evaluate the model on [`get_test_dataloader`][datasets.cesnet_dataset.CesnetDataset.get_test_dataloader] or [`get_test_df`][datasets.cesnet_dataset.CesnetDataset.get_test_df].

    Parameters:
        data_root: Path to the folder where the dataset will be stored. Each dataset size has its own subfolder `data_root/size`
        size: Size of the dataset. Options are `XS`, `S`, `M`, `L`, `ORIG`.
        silent: Whether to suppress print and tqdm output.

    Attributes:
        name: Name of the dataset.
        database_filename: Name of the database file.
        database_path: Path to the database file.
        servicemap_path: Path to the servicemap file.
        statistics_path: Path to the dataset statistics folder.
        bucket_url: URL of the bucket where the database is stored.
        metadata: Additional [dataset metadata][metadata].
        available_classes: List of all available classes in the dataset.
        available_dates: List of all available dates in the dataset.
        time_periods: Predefined time periods. Each time period is a list of dates.
        default_train_period_name: Default time period for training.
        default_test_period_name: Default time period for testing.

    The following attributes are initialized when [`set_dataset_config_and_initialize`][datasets.cesnet_dataset.CesnetDataset.set_dataset_config_and_initialize] is called.

    Attributes:
        dataset_config: Configuration of the dataset.
        class_info: Structured information about the classes.
        dataset_indices: Named tuple containing `train_indices`, `val_known_indices`, `val_unknown_indices`, `test_known_indices`, `test_unknown_indices`. These are the indices into PyTables database that define train, validation, and test sets.
        train_dataset: Train set in the form of `PyTablesDataset` instance wrapping the PyTables database.
        val_dataset: Validation set in the form of `PyTablesDataset` instance wrapping the PyTables database.
        test_dataset: Test set in the form of `PyTablesDataset` instance wrapping the PyTables database.
        known_app_counts: Known application counts in the train, validation, and test sets.
        unknown_app_counts: Unknown application counts in the validation and test sets.
        train_dataloader: Iterable PyTorch [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) for training.
        train_dataloader_sampler: Sampler used for iterating the training dataloader. Either [`RandomSampler`](https://pytorch.org/docs/stable/data.html#torch.utils.data.RandomSampler) or [`SequentialSampler`](https://pytorch.org/docs/stable/data.html#torch.utils.data.SequentialSampler).
        train_dataloader_drop_last: Whether to drop the last incomplete batch when iterating the training dataloader.
        val_dataloader: Iterable PyTorch [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) for validation.
        test_dataloader: Iterable PyTorch [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) for testing.
    """
    data_root: str
    size: str
    silent: bool = False

    name: str
    database_filename: str
    database_path: str
    servicemap_path: str
    statistics_path: str
    bucket_url: str
    metadata: DatasetMetadata
    available_classes: list[str]
    available_dates: list[str]
    time_periods: dict[str, list[str]]
    default_train_period_name: str
    default_test_period_name: str

    dataset_config: Optional[DatasetConfig] = None
    class_info: Optional[ClassInfo] = None
    dataset_indices: Optional[IndicesTuple] = None
    train_dataset: Optional[PyTablesDataset] = None
    val_dataset: Optional[PyTablesDataset] = None
    test_dataset: Optional[PyTablesDataset] = None
    known_app_counts: Optional[pd.DataFrame] = None
    unknown_app_counts: Optional[pd.DataFrame] = None
    train_dataloader: Optional[DataLoader] = None
    train_dataloader_sampler: Optional[Sampler] = None
    train_dataloader_drop_last: bool = True
    val_dataloader: Optional[DataLoader] = None
    test_dataloader: Optional[DataLoader] = None

    _collate_fn: Optional[Callable] = None
    _tables_app_enum: dict[int, str]
    _tables_cat_enum: dict[int, str]

    def __init__(self, data_root: str, size: str = "S", database_checks_at_init: bool = False, silent: bool = False) -> None:
        self.silent = silent
        self.metadata = load_metadata(self.name)
        self.size = size
        if self.size != "ORIG":
            if size not in self.metadata.available_dataset_sizes:
                raise ValueError(f"Unknown dataset size {self.size}")
            self.name = f"{self.name}-{self.size}"
            filename, ext = os.path.splitext(self.database_filename)
            self.database_filename = f"{filename}-{self.size}{ext}"
        self.data_root = os.path.normpath(os.path.expanduser(os.path.join(data_root, self.size)))
        self.database_path = os.path.join(self.data_root, self.database_filename)
        self.servicemap_path = os.path.join(self.data_root, SERVICEMAP_FILE)
        self.statistics_path = os.path.join(self.data_root, "statistics")
        if not os.path.exists(self.data_root):
            os.makedirs(self.data_root)
        if not self._is_downloaded():
            self._download()
        if database_checks_at_init:
            with tb.open_file(self.database_path, mode="r") as database:
                tables_paths = list(map(lambda x: x._v_pathname, iter(database.get_node(f"/flows"))))
                num_samples = 0
                for p in tables_paths:
                    table = database.get_node(p)
                    assert isinstance(table, tb.Table)
                    if self._tables_app_enum != {v: k for k, v in dict(table.get_enum(APP_COLUMN)).items()}:
                        raise ValueError(f"Found mismatch between _tables_app_enum and the PyTables database enum in table {p}. Please report this issue.")
                    if self._tables_cat_enum != {v: k for k, v in dict(table.get_enum(CATEGORY_COLUMN)).items()}:
                        raise ValueError(f"Found mismatch between _tables_cat_enum and the PyTables database enum in table {p}. Please report this issue.")
                    num_samples += len(table)
                if self.size == "ORIG" and num_samples != self.metadata.available_samples:
                    raise ValueError(f"Expected {self.metadata.available_samples} samples, but got {num_samples} in the database. Please delete the data root folder, update cesnet-datazoo, and redownload the dataset.")
                if self.size != "ORIG" and num_samples != DATASET_SIZES[self.size]:
                    raise ValueError(f"Expected {DATASET_SIZES[self.size]} samples, but got {num_samples} in the database. Please delete the data root folder, update cesnet-datazoo, and redownload the dataset.")
                if self.available_dates != list(map(lambda x: x.removeprefix("/flows/D"), tables_paths)):
                    raise ValueError(f"Found mismatch between available_dates and the dates available in the PyTables database. Please report this issue.")
        # Add all available dates as single date time periods
        for d in self.available_dates:
            self.time_periods[d] = [d]
        available_applications = sorted([app for app in pd.read_csv(self.servicemap_path, index_col="Tag").index if not is_background_app(app)])
        if len(available_applications) != self.metadata.application_count:
            raise ValueError(f"Found {len(available_applications)} applications in the servicemap (omitting background traffic classes), but expected {self.metadata.application_count}. Please report this issue.")
        self.available_classes = available_applications + self.metadata.background_traffic_classes

    def set_dataset_config_and_initialize(self, dataset_config: DatasetConfig, disable_indices_cache: bool = False) -> None:
        """
        Initialize train, validation, and test sets. Data cannot be accessed before calling this method.

        Parameters:
            dataset_config: Desired configuration of the dataset.
            disable_indices_cache: Whether to disable caching of the dataset indices. This is useful when the dataset is used in many different configurations and you want to save disk space.
        """
        self.dataset_config = dataset_config
        self._clear()
        self._initialize_train_val_test(disable_indices_cache=disable_indices_cache)

    def get_train_dataloader(self) -> DataLoader:
        """
        Provides a PyTorch [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) for training. The dataloader is created on the first call and then cached.
        When the dataloader is iterated in random order, the last incomplete batch is dropped.
        The dataloader is configured with the following config attributes:

        | Dataset config               | Description                                                                                |
        | ---------------------------- | ------------------------------------------------------------------------------------------ |
        | `batch_size`                 | Number of samples per batch.                                                               |
        | `train_workers`              | Number of workers for loading train data.                                                  |
        | `train_dataloader_order`     | Whether to load train data in sequential or random order. See [config.DataLoaderOrder][].  |
        | `train_dataloader_seed`      | Seed for loading train data in random order.                                               |

        Returns:
            Train data as an iterable dataloader. See [using dataloaders][using-dataloaders] for more details.
        """
        if self.dataset_config is None:
            raise ValueError("Dataset is not initialized, use set_dataset_config_and_initialize() before getting train dataloader")
        if not self.dataset_config.need_train_set:
            raise ValueError("Train dataloader is not available when need_train_set is false")
        assert self.train_dataset
        if self.train_dataloader:
            return self.train_dataloader
        # Create sampler according to the selected order
        if self.dataset_config.train_dataloader_order == DataLoaderOrder.RANDOM:
            if self.dataset_config.train_dataloader_seed is not None:
                generator = torch.Generator()
                generator.manual_seed(self.dataset_config.train_dataloader_seed)
            else:
                generator = None
            self.train_dataloader_sampler = RandomSampler(self.train_dataset, generator=generator)
            self.train_dataloader_drop_last = True
        elif self.dataset_config.train_dataloader_order == DataLoaderOrder.SEQUENTIAL:
            self.train_dataloader_sampler = SequentialSampler(self.train_dataset)
            self.train_dataloader_drop_last = False
        else: assert_never(self.dataset_config.train_dataloader_order)
        # Create dataloader
        batch_sampler = BatchSampler(sampler=self.train_dataloader_sampler, batch_size=self.dataset_config.batch_size, drop_last=self.train_dataloader_drop_last)
        train_dataloader = DataLoader(
            self.train_dataset,
            num_workers=self.dataset_config.train_workers,
            worker_init_fn=worker_init_fn,
            collate_fn=self._collate_fn,
            persistent_workers=self.dataset_config.train_workers > 0,
            batch_size=None,
            sampler=batch_sampler,)
        if self.dataset_config.train_workers == 0:
            self.train_dataset.pytables_worker_init()
        self.train_dataloader = train_dataloader
        return train_dataloader

    def get_val_dataloader(self) -> DataLoader:
        """
        Provides a PyTorch [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) for validation.
        The dataloader is created on the first call and then cached.
        The dataloader is configured with the following config attributes:

        | Dataset config    | Description                                                       |
        | ------------------| ------------------------------------------------------------------|
        | `test_batch_size` | Number of samples per batch for loading validation and test data. |
        | `val_workers`     | Number of workers for loading validation data.                    |

        Returns:
            Validation data as an iterable dataloader. See [using dataloaders][using-dataloaders] for more details.
        """
        if self.dataset_config is None:
            raise ValueError("Dataset is not initialized, use set_dataset_config_and_initialize() before getting validaion dataloader")
        if not self.dataset_config.need_val_set:
            raise ValueError("Validation dataloader is not available when need_val_set is false")
        assert self.val_dataset is not None
        if self.val_dataloader:
            return self.val_dataloader
        batch_sampler = BatchSampler(sampler=SequentialSampler(self.val_dataset), batch_size=self.dataset_config.test_batch_size, drop_last=False)
        val_dataloader = DataLoader(
            self.val_dataset,
            num_workers=self.dataset_config.val_workers,
            worker_init_fn=worker_init_fn,
            collate_fn=self._collate_fn,
            persistent_workers=self.dataset_config.val_workers > 0,
            batch_size=None,
            sampler=batch_sampler,)
        if self.dataset_config.val_workers == 0:
            self.val_dataset.pytables_worker_init()
        self.val_dataloader = val_dataloader
        return val_dataloader

    def get_test_dataloader(self) -> DataLoader:
        """
        Provides a PyTorch [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) for testing.
        The dataloader is created on the first call and then cached.

        When the dataset is used in the open-world setting, and unknown classes are defined,
        the test dataloader returns `test_known_size` samples of known classes followed by `test_unknown_size` samples of unknown classes.

        The dataloader is configured with the following config attributes:

        | Dataset config    | Description                                                       |
        | ------------------| ------------------------------------------------------------------|
        | `test_batch_size` | Number of samples per batch for loading validation and test data. |
        | `test_workers`    | Number of workers for loading test data.                          |

        Returns:
            Test data as an iterable dataloader. See [using dataloaders][using-dataloaders] for more details.
        """
        if self.dataset_config is None:
            raise ValueError("Dataset is not initialized, use set_dataset_config_and_initialize() before getting test dataloader")
        if not self.dataset_config.need_test_set:
            raise ValueError("Test dataloader is not available when need_test_set is false")
        assert self.test_dataset is not None
        if self.test_dataloader:
            return self.test_dataloader
        batch_sampler = BatchSampler(sampler=SequentialSampler(self.test_dataset), batch_size=self.dataset_config.test_batch_size, drop_last=False)
        test_dataloader = DataLoader(
            self.test_dataset,
            num_workers=self.dataset_config.test_workers,
            worker_init_fn=worker_init_fn,
            collate_fn=self._collate_fn,
            persistent_workers=False,
            batch_size=None,
            sampler=batch_sampler,)
        if self.dataset_config.test_workers == 0:
            self.test_dataset.pytables_worker_init()
        self.test_dataloader = test_dataloader
        return test_dataloader

    def get_dataloaders(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Gets train, validation, and test dataloaders in one call."""
        if self.dataset_config is None:
            raise ValueError("Dataset is not initialized, use set_dataset_config_and_initialize() before getting dataloaders")
        train_dataloader = self.get_train_dataloader()
        val_dataloader = self.get_val_dataloader()
        test_dataloader = self.get_test_dataloader()
        return train_dataloader, val_dataloader, test_dataloader

    def get_train_df(self, flatten_ppi: bool = False) -> pd.DataFrame:
        """
        Creates a train Pandas [`DataFrame`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html). The dataframe is in sequential (datetime) order. Consider shuffling the dataframe if needed.

        !!! warning "Memory usage"

            The whole train set is loaded into memory. If the dataset size is larger than `'S'`, consider using `get_train_dataloader` instead.

        Parameters:
            flatten_ppi: Whether to flatten the PPI sequence into individual columns (named `IPT_X`, `DIR_X`, `SIZE_X`, `PUSH_X`, *X* being the index of the packet) or keep one `PPI` column with 2D data.

        Returns:
            Train data as a dataframe.
        """
        self._check_before_dataframe(check_train=True)
        assert self.dataset_config is not None and self.train_dataset is not None
        if len(self.train_dataset) > DATAFRAME_SAMPLES_WARNING_THRESHOLD:
            warnings.warn(f"Train set has ({len(self.train_dataset)} samples), consider using get_train_dataloader() instead")
        train_dataloader = self.get_train_dataloader()
        assert isinstance(train_dataloader.sampler, BatchSampler) and self.train_dataloader_sampler is not None
        # Read dataloader in sequential order
        train_dataloader.sampler.sampler = SequentialSampler(self.train_dataset)
        train_dataloader.sampler.drop_last = False
        feature_names = self.dataset_config.get_feature_names(flatten_ppi=flatten_ppi)
        df = create_df_from_dataloader(dataloader=train_dataloader,
                                       feature_names=feature_names,
                                       flatten_ppi=flatten_ppi,
                                       silent=self.silent)
        # Restore the original dataloader sampler and drop_last
        train_dataloader.sampler.sampler = self.train_dataloader_sampler
        train_dataloader.sampler.drop_last = self.train_dataloader_drop_last
        return df

    def get_val_df(self, flatten_ppi: bool = False) -> pd.DataFrame:
        """
        Creates validation Pandas [`DataFrame`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html). The dataframe is in sequential (datetime) order.

        !!! warning "Memory usage"

            The whole validation set is loaded into memory. If the dataset size is larger than `'S'`, consider using `get_val_dataloader` instead.

        Parameters:
            flatten_ppi: Whether to flatten the PPI sequence into individual columns (named `IPT_X`, `DIR_X`, `SIZE_X`, `PUSH_X`, *X* being the index of the packet) or keep one `PPI` column with 2D data.

        Returns:
            Validation data as a dataframe.
        """
        self._check_before_dataframe(check_val=True)
        assert self.dataset_config is not None and self.val_dataset is not None
        if len(self.val_dataset) > DATAFRAME_SAMPLES_WARNING_THRESHOLD:
            warnings.warn(f"Validation set has ({len(self.val_dataset)} samples), consider using get_val_dataloader() instead")
        feature_names = self.dataset_config.get_feature_names(flatten_ppi=flatten_ppi)
        return create_df_from_dataloader(dataloader=self.get_val_dataloader(),
                                         feature_names=feature_names,
                                         flatten_ppi=flatten_ppi,
                                         silent=self.silent)

    def get_test_df(self, flatten_ppi: bool = False) -> pd.DataFrame:
        """
        Creates test Pandas [`DataFrame`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html). The dataframe is in sequential (datetime) order.


        When the dataset is used in the open-world setting, and unknown classes are defined,
        the returned test dataframe is composed of `test_known_size` samples of known classes followed by `test_unknown_size` samples of unknown classes.


        !!! warning "Memory usage"

            The whole test set is loaded into memory. If the dataset size is larger than `'S'`, consider using `get_test_dataloader` instead.

        Parameters:
            flatten_ppi: Whether to flatten the PPI sequence into individual columns (named `IPT_X`, `DIR_X`, `SIZE_X`, `PUSH_X`, *X* being the index of the packet) or keep one `PPI` column with 2D data.

        Returns:
            Test data as a dataframe.
        """
        self._check_before_dataframe(check_test=True)
        assert self.dataset_config is not None and self.test_dataset is not None
        if len(self.test_dataset) > DATAFRAME_SAMPLES_WARNING_THRESHOLD:
            warnings.warn(f"Test set has ({len(self.test_dataset)} samples), consider using get_test_dataloader() instead")
        feature_names = self.dataset_config.get_feature_names(flatten_ppi=flatten_ppi)
        return create_df_from_dataloader(dataloader=self.get_test_dataloader(),
                                         feature_names=feature_names,
                                         flatten_ppi=flatten_ppi,
                                         silent=self.silent)

    def get_num_classes(self) -> int:
        """Returns the number of classes in the current configuration of the dataset."""
        if self.class_info is None:
            raise ValueError("Dataset is not initialized, use set_dataset_config_and_initialize() before getting the number of classes")
        return self.class_info.num_classes

    def get_known_apps(self) -> list[str]:
        """Returns the list of known applications in the current configuration of the dataset."""
        if self.class_info is None:
            raise ValueError("Dataset is not initialized, use set_dataset_config_and_initialize() before getting known apps")
        return self.class_info.known_apps

    def get_unknown_apps(self) -> list[str]:
        """Returns the list of unknown applications in the current configuration of the dataset."""
        if self.class_info is None:
            raise ValueError("Dataset is not initialized, use set_dataset_config_and_initialize() before getting unknown apps")
        return self.class_info.unknown_apps

    def compute_dataset_statistics(self, num_samples: int | Literal["all"] = 10_000_000, num_workers: int = 4, batch_size: int = 16384, disabled_apps: Optional[list[str]] = None) -> None:
        """
        Computes dataset statistics and saves them to the `statistics_path` folder.

        Parameters:
            num_samples: Number of samples to use for computing the statistics.
            num_workers: Number of workers for loading data.
            batch_size: Number of samples per batch for loading data.
            disabled_apps: List of applications to exclude from the statistics.
        """
        if disabled_apps:
            bad_disabled_apps = [a for a in disabled_apps if a not in self.available_classes]
            if len(bad_disabled_apps) > 0:
                raise ValueError(f"Bad applications in disabled_apps {bad_disabled_apps}. Use applications available in dataset.available_classes")
        if not os.path.exists(self.statistics_path):
            os.mkdir(self.statistics_path)
        compute_dataset_statistics(database_path=self.database_path,
                                   tables_app_enum=self._tables_app_enum,
                                   tables_cat_enum=self._tables_cat_enum,
                                   output_dir=self.statistics_path,
                                   packet_histograms=self.metadata.packet_histograms,
                                   flowstats_features_boolean=self.metadata.flowstats_features_boolean,
                                   protocol=self.metadata.protocol,
                                   extra_fields=not self.name.startswith("CESNET-TLS22"),
                                   disabled_apps=disabled_apps if disabled_apps is not None else [],
                                   num_samples=num_samples,
                                   num_workers=num_workers,
                                   batch_size=batch_size,
                                   silent=self.silent)

    def _generate_time_periods(self) -> None:
        time_periods = {}
        for period in self.time_periods:
            time_periods[period] = []
            if period.startswith("W"):
                split = period.split("-")
                collection_year, week = int(split[1]), int(split[2])
                for d in range(1, 8):
                    s = datetime.date.fromisocalendar(collection_year, week, d).strftime("%Y%m%d")
                    # last week of a year can span into the following year
                    if s not in self.metadata.missing_dates_in_collection_period and s.startswith(str(collection_year)):
                        time_periods[period].append(s)
            elif period.startswith("M"):
                split = period.split("-")
                collection_year, month = int(split[1]), int(split[2])
                for d in range(1, calendar.monthrange(collection_year, month)[1]):
                    s = datetime.date(collection_year, month, d).strftime("%Y%m%d")
                    if s not in self.metadata.missing_dates_in_collection_period:
                        time_periods[period].append(s)
        self.time_periods = time_periods

    def _is_downloaded(self) -> bool:
        """Servicemap is downloaded after the database; thus if it exists, the database is also downloaded"""
        return os.path.exists(self.servicemap_path) and os.path.exists(self.database_path)

    def _download(self) -> None:
        if not self.silent:
            print(f"Downloading {self.name} dataset")
        database_url = f"{self.bucket_url}&file={self.database_filename}"
        servicemap_url = f"{self.bucket_url}&file={SERVICEMAP_FILE}"
        resumable_download(url=database_url, file_path=self.database_path, silent=self.silent)
        simple_download(url=servicemap_url, file_path=self.servicemap_path)

    def _clear(self) -> None:
        self.class_info = None
        self.dataset_indices = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.known_app_counts = None
        self.unknown_app_counts = None
        self.train_dataloader = None
        self.train_dataloader_sampler = None
        self.train_dataloader_drop_last = True
        self.val_dataloader = None
        self.test_dataloader = None
        self._collate_fn = None

    def _check_before_dataframe(self, check_train: bool = False, check_val: bool = False, check_test: bool = False) -> None:
        if self.dataset_config is None:
            raise ValueError("Dataset is not initialized, use set_dataset_config_and_initialize() before getting a dataframe")
        if self.dataset_config.return_tensors:
            raise ValueError("Dataframes are not available when return_tensors is set. Use a dataloader instead.")
        if check_train and not self.dataset_config.need_train_set:
            raise ValueError("Train dataframe is not available when need_train_set is false")
        if check_val and not self.dataset_config.need_val_set:
            raise ValueError("Validation dataframe is not available when need_val_set is false")
        if check_test and not self.dataset_config.need_test_set:
            raise ValueError("Test dataframe is not available when need_test_set is false")

    def _initialize_train_val_test(self, disable_indices_cache: bool = False) -> None:
        assert self.dataset_config is not None
        dataset_config = self.dataset_config
        servicemap = pd.read_csv(dataset_config.servicemap_path, index_col="Tag")
        # Initialize train set
        if dataset_config.need_train_set:
            train_indices, train_unknown_indices, known_apps, unknown_apps = init_or_load_train_indices(dataset_config=dataset_config,
                                                                                                        tables_app_enum=self._tables_app_enum,
                                                                                                        servicemap=servicemap,
                                                                                                        disable_indices_cache=disable_indices_cache,)
            # Date weight sampling of train indices
            if dataset_config.train_dates_weigths is not None:
                assert dataset_config.train_size != "all"
                if dataset_config.val_approach == ValidationApproach.SPLIT_FROM_TRAIN:
                    # requested number of samples is train_size + val_known_size when using the split-from-train validation approach
                    assert dataset_config.val_known_size != "all"
                    num_samples = dataset_config.train_size + dataset_config.val_known_size
                else:
                    num_samples = dataset_config.train_size
                if num_samples > len(train_indices):
                    raise ValueError(f"Requested number of samples for weight sampling ({num_samples}) is larger than the number of available train samples ({len(train_indices)})")
                train_indices = date_weight_sample_train_indices(dataset_config=dataset_config, train_indices=train_indices, num_samples=num_samples)
        elif dataset_config.apps_selection == AppSelection.FIXED:
            known_apps = dataset_config.apps_selection_fixed_known
            unknown_apps = dataset_config.apps_selection_fixed_unknown
            train_indices = np.zeros((0,3), dtype=np.int64)
            train_unknown_indices = np.zeros((0,3), dtype=np.int64)
        else:
            raise ValueError("Either need train set or the fixed application selection")
        # Initialize validation set
        if dataset_config.need_val_set:
            if dataset_config.val_approach == ValidationApproach.VALIDATION_DATES:
                val_known_indices, val_unknown_indices, val_data_path = init_or_load_val_indices(dataset_config=dataset_config,
                                                                                                 known_apps=known_apps,
                                                                                                 unknown_apps=unknown_apps,
                                                                                                 tables_app_enum=self._tables_app_enum,
                                                                                                 disable_indices_cache=disable_indices_cache,)
            elif dataset_config.val_approach == ValidationApproach.SPLIT_FROM_TRAIN:
                train_val_rng = get_fresh_random_generator(dataset_config=dataset_config, section=RandomizedSection.TRAIN_VAL_SPLIT)
                val_data_path = dataset_config._get_train_data_path()
                val_unknown_indices = train_unknown_indices
                train_labels = train_indices[:, INDICES_LABEL_POS]
                if dataset_config.train_dates_weigths is not None:
                    assert dataset_config.val_known_size != "all"
                    # When weight sampling is used, val_known_size is kept but the resulting train size can be smaller due to no enough samples in some train dates
                    if dataset_config.val_known_size > len(train_indices):
                        raise ValueError(f"Requested validation size ({dataset_config.val_known_size}) is larger than the number of available train samples after weight sampling ({len(train_indices)})")
                    train_indices, val_known_indices = train_test_split(train_indices, test_size=dataset_config.val_known_size, stratify=train_labels, shuffle=True, random_state=train_val_rng)
                    dataset_config.train_size = len(train_indices)
                elif dataset_config.train_size == "all" and dataset_config.val_known_size == "all":
                    train_indices, val_known_indices = train_test_split(train_indices, test_size=dataset_config.train_val_split_fraction, stratify=train_labels, shuffle=True, random_state=train_val_rng)
                else:
                    if dataset_config.val_known_size != "all" and  dataset_config.train_size != "all" and dataset_config.train_size + dataset_config.val_known_size > len(train_indices):
                        raise ValueError(f"Requested train size + validation size ({dataset_config.train_size + dataset_config.val_known_size}) is larger than the number of available train samples ({len(train_indices)})")
                    if dataset_config.train_size != "all" and dataset_config.train_size > len(train_indices):
                        raise ValueError(f"Requested train size ({dataset_config.train_size}) is larger than the number of available train samples ({len(train_indices)})")
                    if dataset_config.val_known_size != "all" and dataset_config.val_known_size > len(train_indices):
                        raise ValueError(f"Requested validation size ({dataset_config.val_known_size}) is larger than the number of available train samples ({len(train_indices)})")
                    train_indices, val_known_indices = train_test_split(train_indices,
                                                                        train_size=dataset_config.train_size if dataset_config.train_size != "all" else None,
                                                                        test_size=dataset_config.val_known_size if dataset_config.val_known_size != "all" else None,
                                                                        stratify=train_labels, shuffle=True, random_state=train_val_rng)
        else:
            val_known_indices = np.zeros((0,3), dtype=np.int64)
            val_unknown_indices = np.zeros((0,3), dtype=np.int64)
            val_data_path = None
        # Initialize test set
        if dataset_config.need_test_set:
            test_known_indices, test_unknown_indices, test_data_path = init_or_load_test_indices(dataset_config=dataset_config,
                                                                                                 known_apps=known_apps,
                                                                                                 unknown_apps=unknown_apps,
                                                                                                 tables_app_enum=self._tables_app_enum,
                                                                                                 disable_indices_cache=disable_indices_cache,)
        else:
            test_known_indices = np.zeros((0,3), dtype=np.int64)
            test_unknown_indices = np.zeros((0,3), dtype=np.int64)
            test_data_path = None
        # Fit scalers if needed
        if (dataset_config.ppi_transform is not None and dataset_config.ppi_transform.needs_fitting or
            dataset_config.flowstats_transform is not None and dataset_config.flowstats_transform.needs_fitting):
            if not dataset_config.need_train_set:
                raise ValueError("Train set is needed to fit the scalers. Provide pre-fitted scalers.")
            fit_scalers(dataset_config=dataset_config, train_indices=train_indices)
        # Subset dataset indices based on the selected sizes and compute application counts
        dataset_indices = IndicesTuple(train_indices=train_indices, val_known_indices=val_known_indices, val_unknown_indices=val_unknown_indices, test_known_indices=test_known_indices, test_unknown_indices=test_unknown_indices)
        dataset_indices = subset_and_sort_indices(dataset_config=dataset_config, dataset_indices=dataset_indices)
        known_app_counts = compute_known_app_counts(dataset_indices=dataset_indices, tables_app_enum=self._tables_app_enum)
        unknown_app_counts = compute_unknown_app_counts(dataset_indices=dataset_indices, tables_app_enum=self._tables_app_enum)
        # Combine known and unknown test indicies to create a single dataloader
        assert isinstance(dataset_config.test_unknown_size, int)
        if dataset_config.test_unknown_size > 0 and len(unknown_apps) > 0:
            test_combined_indices = np.concatenate((dataset_indices.test_known_indices, dataset_indices.test_unknown_indices))
        else:
            test_combined_indices = dataset_indices.test_known_indices
        # Create encoder the class info structure
        encoder = LabelEncoder().fit(known_apps)
        encoder.classes_ = np.append(encoder.classes_, UNKNOWN_STR_LABEL)
        class_info = create_class_info(servicemap=servicemap, encoder=encoder, known_apps=known_apps, unknown_apps=unknown_apps)
        encode_labels_with_unknown_fn = partial(_encode_labels_with_unknown, encoder=encoder, class_info=class_info)
        # Create train, validation, and test datasets
        train_dataset = val_dataset = test_dataset = None
        if dataset_config.need_train_set:
            train_dataset = PyTablesDataset(
                database_path=dataset_config.database_path,
                tables_paths=dataset_config._get_train_tables_paths(),
                indices=dataset_indices.train_indices,
                tables_app_enum=self._tables_app_enum,
                tables_cat_enum=self._tables_cat_enum,
                flowstats_features=dataset_config.flowstats_features,
                flowstats_features_boolean=dataset_config.flowstats_features_boolean,
                flowstats_features_phist=dataset_config.flowstats_features_phist,
                other_fields=self.dataset_config.other_fields,
                ppi_channels=dataset_config.get_ppi_channels(),
                ppi_transform=dataset_config.ppi_transform,
                flowstats_transform=dataset_config.flowstats_transform,
                flowstats_phist_transform=dataset_config.flowstats_phist_transform,
                target_transform=encode_labels_with_unknown_fn,
                return_tensors=dataset_config.return_tensors,)
        if dataset_config.need_val_set:
            assert val_data_path is not None
            val_dataset = PyTablesDataset(
                database_path=dataset_config.database_path,
                tables_paths=dataset_config._get_train_tables_paths(),
                indices=dataset_indices.val_known_indices,
                tables_app_enum=self._tables_app_enum,
                tables_cat_enum=self._tables_cat_enum,
                flowstats_features=dataset_config.flowstats_features,
                flowstats_features_boolean=dataset_config.flowstats_features_boolean,
                flowstats_features_phist=dataset_config.flowstats_features_phist,
                other_fields=self.dataset_config.other_fields,
                ppi_channels=dataset_config.get_ppi_channels(),
                ppi_transform=dataset_config.ppi_transform,
                flowstats_transform=dataset_config.flowstats_transform,
                flowstats_phist_transform=dataset_config.flowstats_phist_transform,
                target_transform=encode_labels_with_unknown_fn,
                return_tensors=dataset_config.return_tensors,
                preload=dataset_config.preload_val,
                preload_blob=os.path.join(val_data_path, "preload", f"val_dataset-{dataset_config.val_known_size}.npz"),)
        if dataset_config.need_test_set:
            assert test_data_path is not None
            test_dataset = PyTablesDataset(
                database_path=dataset_config.database_path,
                tables_paths=dataset_config._get_test_tables_paths(),
                indices=test_combined_indices,
                tables_app_enum=self._tables_app_enum,
                tables_cat_enum=self._tables_cat_enum,
                flowstats_features=dataset_config.flowstats_features,
                flowstats_features_boolean=dataset_config.flowstats_features_boolean,
                flowstats_features_phist=dataset_config.flowstats_features_phist,
                other_fields=self.dataset_config.other_fields,
                ppi_channels=dataset_config.get_ppi_channels(),
                ppi_transform=dataset_config.ppi_transform,
                flowstats_transform=dataset_config.flowstats_transform,
                flowstats_phist_transform=dataset_config.flowstats_phist_transform,
                target_transform=encode_labels_with_unknown_fn,
                return_tensors=dataset_config.return_tensors,
                preload=dataset_config.preload_test,
                preload_blob=os.path.join(test_data_path, "preload", f"test_dataset-{dataset_config.test_known_size}-{dataset_config.test_unknown_size}.npz"),)
        self.class_info = class_info
        self.dataset_indices = dataset_indices
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.known_app_counts = known_app_counts
        self.unknown_app_counts = unknown_app_counts
        self._collate_fn = collate_fn_simple

def _encode_labels_with_unknown(labels, encoder: LabelEncoder, class_info: ClassInfo):
    return encoder.transform(np.where(np.isin(labels, class_info.known_apps), labels, UNKNOWN_STR_LABEL))
