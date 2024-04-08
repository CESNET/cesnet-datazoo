from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import warnings
from dataclasses import InitVar, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Callable, Literal, Optional

import yaml
from pydantic import model_validator
from pydantic.dataclasses import dataclass

from cesnet_datazoo.constants import (PHIST_BIN_COUNT, PPI_MAX_LEN, SELECTED_TCP_FLAGS,
                                      TCP_PPI_CHANNELS, UDP_PPI_CHANNELS)

if TYPE_CHECKING:
    from cesnet_datazoo.datasets.cesnet_dataset import CesnetDataset


class Protocol(Enum):
    TLS = "TLS"
    QUIC = "QUIC"
    def __str__(self): return self.value

class ValidationApproach(Enum):
    """The validation approach defines which samples should be used for creating a validation set."""
    SPLIT_FROM_TRAIN = "split-from-train"
    """Split train data into train and validation.
    Scikit-learn [`train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
    is used to create a random stratified validation set. The fraction of validation samples is defined in `train_val_split_fraction`."""
    VALIDATION_DATES = "validation-dates"
    """Use separate validation dates to create a validation set. Validation dates need to be specified in `val_dates`, and the name of the validation period in `val_period_name`."""
    def __str__(self): return self.value

class AppSelection(Enum):
    """
    Applications can be divided into *known* and *unknown* classes. To use a dataset in the standard closed-world setting, use `ALL_KNOWN ` to select all the applications as *known*.
    Use `TOPX_KNOWN` or `BACKGROUND_UNKNOWN` for the open-world setting and evaluation of out-of-distribution or open-set recognition methods.
    The `FIXED` is for manual selection of *known* and *unknown* applications.
    """
    ALL_KNOWN = "all-known"
    """Use all applications as *known*."""
    TOPX_KNOWN = "topx-known"
    """Use the first X (`apps_selection_topx`) most frequent (with the most samples) applications as *known*, and the rest as *unknown*.
    Applications with the same provider are never separated, i.e., all applications of a given provider are either *known* or *unknown*."""
    BACKGROUND_UNKNOWN = "background-unknown"
    """Use the list of background traffic classes (`apps_selection_background_unknown`) as *unknown*, and the rest as *known*."""
    FIXED = "fixed"
    """Manual application selection. Provide lists of *known* applications (`apps_selection_fixed_known`) and *unknown* applications (`apps_selection_fixed_unknown`)."""
    def __str__(self): return self.value

class MinTrainSamplesCheck(Enum):
    """
    Depending on the selected train dates, there might be applications with *not enough* samples for training (what is *not enough* will depend on the selected classification model).
    The threshold for the minimum number of samples can be set with `min_train_samples_per_app`, and its default value is 100.
    With the `DISABLE_APPS ` approach, these applications will be disabled and not used for training or testing.
    With the `WARN_AND_EXIT ` approach, the script will print a warning and exit if applications with not enough samples are encountered.
    To disable this check, set `min_train_samples_per_app` to 0.
    """
    WARN_AND_EXIT = "warn-and-exit"
    """Warn and exit if there are not enough training samples for some applications. It is up to the user to manually add these applications to `disabled_apps`."""
    DISABLE_APPS = "disable-apps"
    """Disable applications with not enough training samples."""
    def __str__(self): return self.value

class DataLoaderOrder(Enum):
    """
    Validation and test sets are always loaded in sequential order — sequential meaning in the order of dates and time.
    However, for the train set, it is sometimes required to iterate it in random order (for example, for training a neural network).
    Thus, use `RANDOM ` if your classification model requires it; `SEQUENTIAL` otherwise.
    This setting affects only [train_dataloader][datasets.cesnet_dataset.CesnetDataset.get_train_dataloader]. Dataframe [get_train_df][datasets.cesnet_dataset.CesnetDataset.get_train_df] is always created in sequential order.
    """
    RANDOM = "random"
    """Iterate train data in random order."""
    SEQUENTIAL = "sequential"
    """Iterate train data in sequential (datetime) order."""
    def __str__(self): return self.value

@dataclass(frozen=True)
class TrainDataParams():
    database_filename: str
    train_period_name: str
    train_tables_paths: list[str]
    apps_selection: AppSelection
    apps_selection_topx: int
    apps_selection_background_unknown: list[str]
    apps_selection_fixed_known: list[str]
    apps_selection_fixed_unknown: list[str]
    disabled_apps: list[str]
    min_train_samples_check: MinTrainSamplesCheck
    min_train_samples_per_app: int

@dataclass(frozen=True)
class TestDataParams():
    database_filename: str
    test_period_name: str
    test_tables_paths: list[str]
    known_apps: list[str]
    unknown_apps: list[str]

class C:
    arbitrary_types_allowed = True
    extra = "forbid"

@dataclass(config=C)
class DatasetConfig():
    """
    The main class for the configuration of:

    - Train, validation, test sets (dates, sizes, validation approach).
    - Application selection — either the standard closed-world setting (only *known* classes) or the open-world setting (*known* and *unknown* classes).
    - Data transformations. See the [transforms][transforms] page for more information.
    - Dataloader options like batch sizes, order of loading, or number of workers.

    When initializing this class, pass a [`CesnetDataset`][datasets.cesnet_dataset.CesnetDataset] instance to be configured and the desired configuration. Available options are [here][config.DatasetConfig--configuration-options].

    Attributes:
        dataset: The dataset instance to be configured.
        data_root: Taken from the dataset instance.
        database_filename: Taken from the dataset instance.
        database_path: Taken from the dataset instance.
        servicemap_path: Taken from the dataset instance.
        flowstats_features: Taken from `dataset.metadata.flowstats_features`.
        flowstats_features_boolean: Taken from `dataset.metadata.flowstats_features_boolean`.
        flowstats_features_phist: Taken from `dataset.metadata.packet_histograms` if `use_packet_histograms` is true, otherwise an empty list.
        other_fields: Taken from `dataset.metadata.other_fields` if `return_other_fields` is true, otherwise an empty list.

    # Configuration options

    Attributes:
        need_train_set: Use to disable the train set. `Default: True`
        need_val_set: Use to disable the validation set. `Default: True`
        need_test_set: Use to disable the test set. `Default: True`
        train_period_name: Name of the train period. See [instructions][config.DatasetConfig--how-to-configure-train-validation-and-test-sets].
        train_dates: Dates used for creating a train set.
        train_dates_weigths: To use a non-uniform distribution of samples across train dates.
        val_approach: How a validation set should be created. Either split train data into train and validation or have a separate validation period. `Default: SPLIT_FROM_TRAIN`
        train_val_split_fraction: The fraction of validation samples when splitting from the train set. `Default: 0.2`
        val_period_name: Name of the validation period. See [instructions][config.DatasetConfig--how-to-configure-train-validation-and-test-sets].
        val_dates: Dates used for creating a validation set.
        test_period_name: Name of the test period. See [instructions][config.DatasetConfig--how-to-configure-train-validation-and-test-sets].
        test_dates: Dates used for creating a test set.

        apps_selection: How to select application classes. `Default: ALL_KNOWN`
        apps_selection_topx: Take top X as known.
        apps_selection_background_unknown: Provide a list of background traffic classes to be used as unknown.
        apps_selection_fixed_known: Provide a list of manually selected known applications.
        apps_selection_fixed_unknown: Provide a list of manually selected unknown applications.
        disabled_apps: List of applications to be disabled and not used at all.
        min_train_samples_check: How to handle applications with *not enough* training samples. `Default: DISABLE_APPS`
        min_train_samples_per_app: Defines the threshold for *not enough*. `Default: 100`

        random_state: Fix all random processes performed during dataset initialization. `Default: 420`
        fold_id: To perform N-fold cross-validation, set this to `1..N`. Each fold will use the same configuration but a different random seed. `Default: 0`
        train_workers: Number of workers for loading train data. `0` means that the data will be loaded in the main process. `Default: 4`
        test_workers: Number of workers for loading test data. `0` means that the data will be loaded in the main process. `Default: 1`
        val_workers: Number of workers for loading validation data. `0` means that the data will be loaded in the main process. `Default: 1`
        batch_size: Number of samples per batch. `Default: 192`
        test_batch_size: Number of samples per batch for loading validation and test data. `Default: 2048`
        preload_val: Whether to dump the validation set with `numpy.savez_compressed` and preload it in future runs. Useful when running a lot of experiments with the same dataset configuration. `Default: False`
        preload_test: Whether to dump the test set with `numpy.savez_compressed` and preload it in future runs. `Default: False`
        train_size: Size of the train set. See [instructions][config.DatasetConfig--how-to-configure-train-validation-and-test-sets]. `Default: all`
        val_known_size: Size of the validation set. See [instructions][config.DatasetConfig--how-to-configure-train-validation-and-test-sets]. `Default: all`
        test_known_size: Size of the test set. See [instructions][config.DatasetConfig--how-to-configure-train-validation-and-test-sets]. `Default: all`
        val_unknown_size: Size of the unknown classes validation set. Use for evaluation in the open-world setting. `Default: 0`
        test_unknown_size: Size of the unknown classes test set. Use for evaluation in the open-world setting. `Default: 0`
        train_dataloader_order: Whether to load train data in sequential or random order. `Default: RANDOM`
        train_dataloader_seed: Seed for loading train data in random order. `Default: None`

        return_other_fields: Whether to return [auxiliary fields][other-fields], such as communicating hosts, flow times, and more fields extracted from the ClientHello message. `Default: False`
        return_tensors: Use for returning `torch.Tensor` from dataloaders. Dataframes are not available when this option is used. `Default: False`
        use_packet_histograms: Whether to use packet histogram features, if available in the dataset. `Default: True`
        use_tcp_features: Whether to use TCP features, if available in the dataset. `Default: True`
        use_push_flags: Whether to use push flags in packet sequences, if available in the dataset. `Default: False`
        fit_scalers_samples: Used when scaling transformation is configured and requires fitting. Fraction of train samples used for fitting, if float. The absolute number of samples otherwise. `Default: 0.25`
        ppi_transform: Transform function for PPI sequences. See the [transforms][transforms] page for more information. `Default: None`
        flowstats_transform: Transform function for flow statistics. See the [transforms][transforms] page for more information. `Default: None`
        flowstats_phist_transform: Transform function for packet histograms. See the [transforms][transforms] page for more information. `Default: None`

    # How to configure train, validation, and test sets
    There are three options for how to define train/validation/test dates.

    1. Choose a predefined time period (`train_period_name`, `val_period_name`, or `test_period_name`) available in `dataset.time_periods` and leave the list of dates (`train_dates`, `val_dates`, or `test_dates`) empty.
    2. Provide a list of dates and a name for the time period. The dates are checked against `dataset.available_dates`.
    3. Do not specify anything and use the dataset's defaults `dataset.default_train_period_name` and `dataset.default_test_period_name`.

    There are two options for configuring sizes of train/validation/test sets.

    1. Select an appropriate dataset size (default is `S`) when creating the [`CesnetDataset`][datasets.cesnet_dataset.CesnetDataset] instance and leave `train_size`, `val_known_size`, and `test_known_size` with their default `all` value.
    This will create train/validation/test sets with all samples available in the selected dataset size (of course, depending on the selected dates and validation approach).
    2. Provide exact sizes in `train_size`, `val_known_size`, and `test_known_size`. This will create train/validation/test sets of the given sizes by doing a random subset.
    This is especially useful when using the `ORIG` dataset size and want to control the size of experiments.

    !!! tip Validation set
        The default approach for creating a validation set is to randomly split the train data into train and validation. The second approach is to define separate validation dates. See [ValidationApproach][config.ValidationApproach].

    """
    dataset: InitVar[CesnetDataset]
    data_root: str = field(init=False)
    database_filename: str =  field(init=False)
    database_path: str =  field(init=False)
    servicemap_path: str = field(init=False)
    flowstats_features: list[str] = field(init=False)
    flowstats_features_boolean: list[str] = field(init=False)
    flowstats_features_phist: list[str] = field(init=False)
    other_fields: list[str] = field(init=False)

    need_train_set: bool = True
    need_val_set: bool = True
    need_test_set: bool = True
    train_period_name: str = ""
    train_dates: list[str] = field(default_factory=list)
    train_dates_weigths: Optional[list[int]] = None
    val_approach: ValidationApproach = ValidationApproach.SPLIT_FROM_TRAIN
    train_val_split_fraction: float = 0.2
    val_period_name: str = ""
    val_dates: list[str] = field(default_factory=list)
    test_period_name: str = ""
    test_dates: list[str] = field(default_factory=list)

    apps_selection: AppSelection = AppSelection.ALL_KNOWN
    apps_selection_topx: int = 0
    apps_selection_background_unknown: list[str] = field(default_factory=list)
    apps_selection_fixed_known: list[str] = field(default_factory=list)
    apps_selection_fixed_unknown: list[str] = field(default_factory=list)
    disabled_apps: list[str] = field(default_factory=list)
    min_train_samples_check: MinTrainSamplesCheck = MinTrainSamplesCheck.DISABLE_APPS
    min_train_samples_per_app: int = 100

    random_state: int = 420
    fold_id: int = 0
    train_workers: int = 4
    test_workers: int = 1
    val_workers: int = 1
    batch_size: int = 192
    test_batch_size: int = 2048
    preload_val: bool = False
    preload_test: bool = False
    train_size: int | Literal["all"] = "all"
    val_known_size: int | Literal["all"] = "all"
    test_known_size: int | Literal["all"] = "all"
    val_unknown_size: int | Literal["all"] = 0
    test_unknown_size: int | Literal["all"] = 0
    train_dataloader_order: DataLoaderOrder = DataLoaderOrder.RANDOM
    train_dataloader_seed: Optional[int] = None

    return_other_fields: bool = False
    return_tensors: bool = False
    use_packet_histograms: bool = False
    use_tcp_features: bool = False
    use_push_flags: bool = False
    fit_scalers_samples: int | float = 0.25
    ppi_transform: Optional[Callable] = None
    flowstats_transform: Optional[Callable] = None
    flowstats_phist_transform: Optional[Callable] = None

    def __post_init__(self, dataset: CesnetDataset):
        """
        Ensures valid configuration. Catches all incompatible options and raise exceptions as soon as possible.
        """
        self.data_root = dataset.data_root
        self.servicemap_path = dataset.servicemap_path
        self.database_filename = dataset.database_filename
        self.database_path = dataset.database_path

        if not self.need_train_set:
            if self.apps_selection != AppSelection.FIXED:
                raise ValueError("Application selection has to be fixed when need_train_set is false")
            if (len(self.train_dates) > 0 or self.train_period_name != ""):
                raise ValueError("train_dates and train_period_name cannot be specified when need_train_set is false")
        else:
            # Configure train dates
            if len(self.train_dates) > 0 and self.train_period_name == "":
                raise ValueError("train_period_name has to be specified when train_dates are set")
            if len(self.train_dates) == 0 and self.train_period_name != "":
                if self.train_period_name not in dataset.time_periods:
                    raise ValueError(f"Unknown train_period_name {self.train_period_name}. Use time period available in dataset.time_periods")
                self.train_dates = dataset.time_periods[self.train_period_name]
            if len(self.train_dates) == 0 and self.train_period_name == "":
                self.train_period_name = dataset.default_train_period_name
                self.train_dates = dataset.time_periods[dataset.default_train_period_name]
        # Configure test dates
        if not self.need_test_set:
            if (len(self.test_dates) > 0 or self.test_period_name != ""):
                raise ValueError("test_dates and test_period_name cannot be specified when need_test_set is false")
        else:
            if len(self.test_dates) > 0 and self.test_period_name == "":
                raise ValueError("test_period_name has to be specified when test_dates are set")
            if len(self.test_dates) == 0 and self.test_period_name != "":
                if self.test_period_name not in dataset.time_periods:
                    raise ValueError(f"Unknown test_period_name {self.test_period_name}. Use time period available in dataset.time_periods")
                self.test_dates = dataset.time_periods[self.test_period_name]
            if len(self.test_dates) == 0 and self.test_period_name == "":
                self.test_period_name = dataset.default_test_period_name
                self.test_dates = dataset.time_periods[dataset.default_test_period_name]
        # Configure val dates
        if not self.need_val_set:
            if len(self.val_dates) > 0 or self.val_period_name != "" or self.val_approach != ValidationApproach.SPLIT_FROM_TRAIN:
                raise ValueError("val_dates, val_period_name, and val_approach cannot be specified when need_val_set is false")
        else:
            if self.val_approach == ValidationApproach.SPLIT_FROM_TRAIN:
                if len(self.val_dates) > 0 or self.val_period_name != "":
                    raise ValueError("val_dates and val_period_name cannot be specified when the validation approach is split-from-train")
                if not self.need_train_set:
                    raise ValueError("Cannot use the split-from-train validation approach when need_train_set is false. Either use the validation-dates approach or set need_val_set to false.")
            elif self.val_approach == ValidationApproach.VALIDATION_DATES:
                if len(self.val_dates) > 0 and self.val_period_name == "":
                    raise ValueError("val_period_name has to be specified when val_dates are set")
                if len(self.val_dates) == 0 and self.val_period_name != "":
                    if self.val_period_name not in dataset.time_periods:
                        raise ValueError(f"Unknown val_period_name {self.val_period_name}. Use time period available in dataset.time_periods")
                    self.val_dates = dataset.time_periods[self.val_period_name]
                if len(self.val_dates) == 0 and self.val_period_name == "":
                    raise ValueError("val_period_name and val_dates (or val_period_name from dataset.time_periods) have to be specified when the validation approach is validation-dates")
        # Check if train, val, and test dates are available in the dataset
        bad_train_dates = [t for t in self.train_dates if t not in dataset.available_dates]
        bad_val_dates = [t for t in self.val_dates if t not in dataset.available_dates]
        bad_test_dates = [t for t in self.test_dates if t not in dataset.available_dates]
        if len(bad_train_dates) > 0:
            raise ValueError(f"Bad train dates {bad_train_dates}. Use dates available in dataset.available_dates (collection period {dataset.metadata.collection_period})" \
                            + (f". These dates are missing from the dataset collection period {dataset.metadata.missing_dates_in_collection_period}" if dataset.metadata.missing_dates_in_collection_period else ""))
        if len(bad_val_dates) > 0:
            raise ValueError(f"Bad validation dates {bad_val_dates}. Use dates available in dataset.available_dates (collection period {dataset.metadata.collection_period})" \
                            + (f". These dates are missing from the dataset collection period {dataset.metadata.missing_dates_in_collection_period}" if dataset.metadata.missing_dates_in_collection_period else ""))
        if len(bad_test_dates) > 0:
            raise ValueError(f"Bad test dates {bad_test_dates}. Use dates available in dataset.available_dates (collection period {dataset.metadata.collection_period})" \
                            + (f". These dates are missing from the dataset collection period {dataset.metadata.missing_dates_in_collection_period}" if dataset.metadata.missing_dates_in_collection_period else ""))
        # Check time order of train, val, and test periods
        train_dates = [datetime.strptime(date_str, "%Y%m%d").date() for date_str in self.train_dates]
        test_dates = [datetime.strptime(date_str, "%Y%m%d").date() for date_str in self.test_dates]
        if len(train_dates) > 0 and len(test_dates) > 0 and min(test_dates) <= max(train_dates):
            warnings.warn(f"Some test dates ({min(test_dates).strftime('%Y%m%d')}) are before or equal to the last train date ({max(train_dates).strftime('%Y%m%d')}). This might lead to improper evaluation and should be avoided.")
        if self.val_approach == ValidationApproach.VALIDATION_DATES:
            val_dates = [datetime.strptime(date_str, "%Y%m%d").date() for date_str in self.val_dates]
            if len(train_dates) > 0 and min(val_dates) <= max(train_dates):
                warnings.warn(f"Some validation dates ({min(val_dates).strftime('%Y%m%d')}) are before or equal to the last train date ({max(train_dates).strftime('%Y%m%d')}). This might lead to improper evaluation and should be avoided.")
            if len(test_dates) > 0 and min(test_dates) <= max(val_dates):
                warnings.warn(f"Some test dates ({min(test_dates).strftime('%Y%m%d')}) are before or equal to the last validation date ({max(val_dates).strftime('%Y%m%d')}). This might lead to improper evaluation and should be avoided.")
        # Configure features
        self.flowstats_features = dataset.metadata.flowstats_features
        self.flowstats_features_boolean = dataset.metadata.flowstats_features_boolean
        self.other_fields = dataset.metadata.other_fields if self.return_other_fields else []
        if self.use_packet_histograms:
            if len(dataset.metadata.packet_histograms) == 0:
                raise ValueError("This dataset does not support use_packet_histograms")
            self.flowstats_features_phist = dataset.metadata.packet_histograms
        else:
            self.flowstats_features_phist = []
            if self.flowstats_phist_transform is not None:
                raise ValueError("flowstats_phist_transform cannot be specified when use_packet_histograms is false")
        if dataset.metadata.protocol == Protocol.TLS:
            if self.use_tcp_features:
                self.flowstats_features_boolean = self.flowstats_features_boolean + SELECTED_TCP_FLAGS
            if self.use_push_flags and "PUSH_FLAG" not in dataset.metadata.ppi_features:
                raise ValueError("This TLS dataset does not support use_push_flags")
        if dataset.metadata.protocol == Protocol.QUIC:
            if self.use_tcp_features:
                raise ValueError("QUIC datasets do not support use_tcp_features")
            if self.use_push_flags:
                raise ValueError("QUIC datasets do not support use_push_flags")
        # When train_dates_weigths are used, train_size and val_known_size have to be specified
        if self.train_dates_weigths is not None:
            if not self.need_train_set:
                raise ValueError("train_dates_weigths cannot be specified when need_train_set is false")
            if len(self.train_dates_weigths) != len(self.train_dates):
                raise ValueError("train_dates_weigths has to have the same length as train_dates")
            if self.train_size == "all":
                raise ValueError("train_size cannot be 'all' when train_dates_weigths are speficied")
            if self.val_approach == ValidationApproach.SPLIT_FROM_TRAIN and self.val_known_size == "all":
                raise ValueError("val_known_size cannot be 'all' when train_dates_weigths are speficied and validation_approach is split-from-train")
        # App selection
        if self.apps_selection == AppSelection.ALL_KNOWN:
            self.val_unknown_size = 0
            self.test_unknown_size = 0
            if self.apps_selection_topx != 0 or len(self.apps_selection_background_unknown) > 0 or len(self.apps_selection_fixed_known) > 0 or len(self.apps_selection_fixed_unknown) > 0:
                raise ValueError("apps_selection_topx, apps_selection_background_unknown, apps_selection_fixed_known, and apps_selection_fixed_unknown cannot be specified when application selection is all-known")
        if self.apps_selection == AppSelection.TOPX_KNOWN:
            if self.apps_selection_topx == 0:
                raise ValueError("apps_selection_topx has to be greater than 0 when application selection is top-x-known")
            if len(self.apps_selection_background_unknown) > 0 or len(self.apps_selection_fixed_known) > 0 or len(self.apps_selection_fixed_unknown) > 0:
                raise ValueError("apps_selection_background_unknown, apps_selection_fixed_known, and apps_selection_fixed_unknown cannot be specified when application selection is top-x-known")
        if self.apps_selection == AppSelection.BACKGROUND_UNKNOWN:
            if len(self.apps_selection_background_unknown) == 0:
                raise ValueError("apps_selection_background_unknown has to be specified when application selection is background-unknown")
            bad_apps = [a for a in self.apps_selection_background_unknown if a not in dataset.available_classes]
            if len(bad_apps) > 0:
                raise ValueError(f"Bad applications in apps_selection_background_unknown {bad_apps}. Use applications available in dataset.available_classes")
            if self.apps_selection_topx != 0 or len(self.apps_selection_fixed_known) > 0 or len(self.apps_selection_fixed_unknown) > 0:
                raise ValueError("apps_selection_topx, apps_selection_fixed_known, and apps_selection_fixed_unknown cannot be specified when application selection is background-unknown")
        if self.apps_selection == AppSelection.FIXED:
            if len(self.apps_selection_fixed_known) == 0:
                raise ValueError("apps_selection_fixed_known has to be specified when application selection is fixed")
            bad_apps = [a for a in self.apps_selection_fixed_known + self.apps_selection_fixed_unknown if a not in dataset.available_classes]
            if len(bad_apps) > 0:
                raise ValueError(f"Bad applications in apps_selection_fixed_known or apps_selection_fixed_unknown {bad_apps}. Use applications available in dataset.available_classes")
            if len(self.disabled_apps) > 0:
                raise ValueError("disabled_apps cannot be specified when application selection is fixed")
            if self.min_train_samples_per_app != 0 and self.min_train_samples_per_app != 100:
                warnings.warn("min_train_samples_per_app is not used when application selection is fixed")
            if self.apps_selection_topx != 0 or len(self.apps_selection_background_unknown) > 0:
                raise ValueError("apps_selection_topx and apps_selection_background_unknown cannot be specified when application selection is fixed")
        # More asserts
        bad_disabled_apps = [a for a in self.disabled_apps if a not in dataset.available_classes]
        if len(bad_disabled_apps) > 0:
            raise ValueError(f"Bad applications in disabled_apps {bad_disabled_apps}. Use applications available in dataset.available_classes")
        if isinstance(self.fit_scalers_samples, float) and (self.fit_scalers_samples <= 0 or self.fit_scalers_samples > 1):
            raise ValueError("fit_scalers_samples has to be either float between 0 and 1 (giving the fraction of training samples used for fitting scalers) or an integer")

    def get_flowstats_features_len(self) -> int:
        """Gets the number of flow statistics features."""
        return len(self.flowstats_features) + len(self.flowstats_features_boolean) + PHIST_BIN_COUNT * len(self.flowstats_features_phist)

    def get_flowstats_feature_names_expanded(self, shorter_names: bool = False) -> list[str]:
        """Gets names of flow statistics features. Packet histograms are expanded into bin features."""
        phist_mapping = {
            "PHIST_SRC_SIZES": [f"PSIZE_BIN{i}" for i in range(1, PHIST_BIN_COUNT + 1)],
            "PHIST_DST_SIZES": [f"PSIZE_BIN{i}_REV" for i in range(1, PHIST_BIN_COUNT + 1)],
            "PHIST_SRC_IPT": [f"IPT_BIN{i}" for i in range(1, PHIST_BIN_COUNT + 1)],
            "PHIST_DST_IPT": [f"IPT_BIN{i}_REV" for i in range(1, PHIST_BIN_COUNT + 1)],
        }
        short_names_mapping = {
            "FLOW_ENDREASON_IDLE": "FEND_IDLE",
            "FLOW_ENDREASON_ACTIVE": "FEND_ACTIVE",
            "FLOW_ENDREASON_END": "FEND_END",
            "FLOW_ENDREASON_OTHER": "FEND_OTHER",
            "FLAG_CWR": "F_CWR",
            "FLAG_CWR_REV": "F_CWR_REV",
            "FLAG_ECE": "F_ECE",
            "FLAG_ECE_REV": "F_ECE_REV",
            "FLAG_PSH_REV": "F_PSH_REV",
            "FLAG_RST": "F_RST",
            "FLAG_RST_REV": "F_RST_REV",
            "FLAG_FIN": "F_FIN",
            "FLAG_FIN_REV": "F_FIN_REV",
        }
        feature_names = self.flowstats_features[:]
        for f in self.flowstats_features_boolean:
            if shorter_names and f in short_names_mapping:
                feature_names.append(short_names_mapping[f])
            else:
                feature_names.append(f)
        for f in self.flowstats_features_phist:
            feature_names.extend(phist_mapping[f])
        assert len(feature_names) == self.get_flowstats_features_len()
        return feature_names

    def get_ppi_feature_names(self) -> list[str]:
        """Gets the names of flattened PPI features."""
        ppi_feature_names = [f"IPT_{i}" for i in range(1, PPI_MAX_LEN + 1)] + \
                               [f"DIR_{i}" for i in range(1, PPI_MAX_LEN + 1)] + \
                               [f"SIZE_{i}" for i in range(1, PPI_MAX_LEN + 1)]
        if self.use_push_flags:
            ppi_feature_names += [f"PUSH_{i}" for i in range(1, PPI_MAX_LEN + 1)]
        return ppi_feature_names

    def get_ppi_channels(self) -> list[int]:
        """Gets the available features (channels) in PPI sequences."""
        if self.use_push_flags:
            return TCP_PPI_CHANNELS
        else:
            return UDP_PPI_CHANNELS

    def get_feature_names(self, flatten_ppi: bool = False, shorter_names: bool = False) -> list[str]:
        """
        Gets feature names.

        Parameters:
            flatten_ppi: Whether to flatten PPI into individual feature names or keep one `PPI` column.
        """
        feature_names = self.get_ppi_feature_names() if flatten_ppi else ["PPI"]
        feature_names += self.get_flowstats_feature_names_expanded(shorter_names=shorter_names)
        return feature_names

    def _get_train_tables_paths(self) -> list[str]:
        return list(map(lambda t: f"/flows/D{t}", self.train_dates))

    def _get_val_tables_paths(self) -> list[str]:
        if self.val_approach == ValidationApproach.SPLIT_FROM_TRAIN:
            return self._get_train_tables_paths()
        return list(map(lambda t: f"/flows/D{t}", self.val_dates))

    def _get_test_tables_paths(self) -> list[str]:
        return list(map(lambda t: f"/flows/D{t}", self.test_dates))

    def _get_train_data_hash(self) -> str:
        train_data_params = self._get_train_data_params()
        params_hash = hashlib.sha256(json.dumps(dataclasses.asdict(train_data_params), sort_keys=True, default=str).encode()).hexdigest()
        params_hash = params_hash[:10]
        return params_hash

    def _get_train_data_path(self) -> str:
        if self.need_train_set:
            params_hash = self._get_train_data_hash()
            return os.path.join(self.data_root, "train-data", f"{params_hash}_{self.random_state}", f"fold_{self.fold_id}")
        else:
            return os.path.join(self.data_root, "train-data", "default")

    def _get_train_data_params(self) -> TrainDataParams:
        return TrainDataParams(
            database_filename=self.database_filename,
            train_period_name=self.train_period_name,
            train_tables_paths=self._get_train_tables_paths(),
            apps_selection=self.apps_selection,
            apps_selection_topx=self.apps_selection_topx,
            apps_selection_background_unknown=self.apps_selection_background_unknown,
            apps_selection_fixed_known=self.apps_selection_fixed_known,
            apps_selection_fixed_unknown=self.apps_selection_fixed_unknown,
            disabled_apps=self.disabled_apps,
            min_train_samples_per_app=self.min_train_samples_per_app,
            min_train_samples_check=self.min_train_samples_check,)

    def _get_val_data_params_and_path(self, known_apps: list[str], unknown_apps: list[str]) -> tuple[TestDataParams, str]:
        assert self.val_approach == ValidationApproach.VALIDATION_DATES
        val_data_params = TestDataParams(
            database_filename=self.database_filename,
            test_period_name=self.val_period_name,
            test_tables_paths=self._get_val_tables_paths(),
            known_apps=known_apps,
            unknown_apps=unknown_apps,)
        params_hash = hashlib.sha256(json.dumps(dataclasses.asdict(val_data_params), sort_keys=True).encode()).hexdigest()
        params_hash = params_hash[:10]
        val_data_path = os.path.join(self.data_root, "val-data", f"{params_hash}_{self.random_state}")
        return val_data_params, val_data_path

    def _get_test_data_params_and_path(self, known_apps: list[str], unknown_apps: list[str]) -> tuple[TestDataParams, str]:
        test_data_params = TestDataParams(
            database_filename=self.database_filename,
            test_period_name=self.test_period_name,
            test_tables_paths=self._get_test_tables_paths(),
            known_apps=known_apps,
            unknown_apps=unknown_apps,)
        params_hash = hashlib.sha256(json.dumps(dataclasses.asdict(test_data_params), sort_keys=True).encode()).hexdigest()
        params_hash = params_hash[:10]
        test_data_path = os.path.join(self.data_root, "test-data", f"{params_hash}_{self.random_state}")
        return test_data_params, test_data_path

    @model_validator(mode="before") # type: ignore
    @classmethod
    def check_deprecated_args(cls, values):
        kwargs = values.kwargs
        if "train_period" in kwargs:
            warnings.warn("train_period is deprecated. Use train_period_name instead.")
            kwargs["train_period_name"] = kwargs["train_period"]
        if "val_period" in kwargs:
            warnings.warn("val_period is deprecated. Use val_period_name instead.")
            kwargs["val_period_name"] = kwargs["val_period"]
        if "test_period" in kwargs:
            warnings.warn("test_period is deprecated. Use test_period_name instead.")
            kwargs["test_period_name"] = kwargs["test_period"]
        return values

    def __str__(self):
        _process_tag = yaml.emitter.Emitter.process_tag
        _ignore_aliases = yaml.Dumper.ignore_aliases
        yaml.emitter.Emitter.process_tag = lambda self, *args, **kw: None
        yaml.Dumper.ignore_aliases = lambda self, *args, **kw: True
        s = yaml.dump(dataclasses.asdict(self), sort_keys=False)
        yaml.emitter.Emitter.process_tag = _process_tag
        yaml.Dumper.ignore_aliases = _ignore_aliases
        return s
