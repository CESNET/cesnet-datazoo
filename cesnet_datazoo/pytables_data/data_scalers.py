from __future__ import annotations

import logging
import os
import time
import warnings
from typing import TYPE_CHECKING

import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from typing_extensions import assert_never

from cesnet_datazoo.config import DatasetConfig, Scaler, ScalerEnum
from cesnet_datazoo.constants import (DIR_POS, FLOWSTATS_NO_CLIP, FLOWSTATS_TO_SCALE, IPT_POS,
                                      PPI_COLUMN, SIZE_POS)
from cesnet_datazoo.pytables_data.pytables_dataset import load_data_from_tables, load_database
from cesnet_datazoo.utils.fileutils import pickle_dump, pickle_load
from cesnet_datazoo.utils.random import RandomizedSection, get_fresh_random_generator

if TYPE_CHECKING:
    from cesnet_datazoo.datasets.cesnet_dataset import CesnetDataset

log = logging.getLogger(__name__)

def get_scaler_attrs(scaler: Scaler) -> dict[str, list[float]]:
    assert Scaler is not None
    if isinstance(scaler, StandardScaler):
        assert hasattr(scaler, "mean_") and scaler.mean_ is not None and hasattr(scaler, "scale_") and scaler.scale_ is not None
        scaler_attrs = {"mean_": scaler.mean_.tolist(), "scale_": scaler.scale_.tolist()}
    elif isinstance(scaler, RobustScaler):
        assert hasattr(scaler, "center_") and hasattr(scaler, "scale_")
        scaler_attrs = {"center_": scaler.center_.tolist(), "scale_": scaler.scale_.tolist()}
    elif isinstance(scaler, MinMaxScaler):
        assert hasattr(scaler, "min_") and hasattr(scaler, "scale_")
        scaler_attrs = {"min_": scaler.min_.tolist(), "scale_": scaler.scale_.tolist()}
    return scaler_attrs

def set_scaler_attrs(scaler: Scaler, scaler_attrs: dict[str, list[float]]):
    assert Scaler is not None
    if isinstance(scaler, StandardScaler):
        assert "mean_" in scaler_attrs and "scale_" in scaler_attrs
        scaler.mean_ = np.array(scaler_attrs["mean_"])
        scaler.scale_ = np.array(scaler_attrs["scale_"])
    elif isinstance(scaler, RobustScaler):
        assert "center_" in scaler_attrs and "scale_" in scaler_attrs
        scaler.center_ = np.array(scaler_attrs["center_"])
        scaler.scale_ = np.array(scaler_attrs["scale_"])
    elif isinstance(scaler, MinMaxScaler):
        assert "min_" in scaler_attrs and "scale_" in scaler_attrs
        scaler.min_ = np.array(scaler_attrs["min_"])
        scaler.scale_ = np.array(scaler_attrs["scale_"])

def save_scalers_attrs_as_dict(dataset: CesnetDataset) -> dict:
    assert dataset.flowstats_scaler is not None or dataset.psizes_scaler is not None or dataset.ipt_scaler is not None
    scalers_dict = {}
    if dataset.flowstats_scaler is not None:
        scalers_dict["flowstats_scaler_attrs"] = get_scaler_attrs(dataset.flowstats_scaler)
    if dataset.psizes_scaler is not None:
        scalers_dict["psizes_scaler_attrs"] = get_scaler_attrs(dataset.psizes_scaler)
    if dataset.ipt_scaler is not None:
        scalers_dict["ipt_scaler_attrs"] = get_scaler_attrs(dataset.ipt_scaler)
    assert dataset.flowstats_quantiles is not None
    scalers_dict["flowstats_quantiles"] = dataset.flowstats_quantiles.tolist()
    return scalers_dict

def fit_or_load_scalers(dataset_config: DatasetConfig, train_indices: np.ndarray) -> tuple[Scaler, Scaler, Scaler, np.ndarray]:
    # Load the scalers from pickled files if scalers_attrs are not provided
    if dataset_config.scalers_attrs is None:
        train_data_path = dataset_config._get_train_data_path()
        flowstats_scaler_path = os.path.join(train_data_path, "stand", f"flowstats_scaler-{dataset_config.flowstats_scaler}-q{dataset_config.flowstats_clip}.pickle")
        psizes_sizes_scaler_path = os.path.join(train_data_path, "stand", f"psizes_scaler-{dataset_config.psizes_scaler}-psizes_max{dataset_config.psizes_max}.pickle")
        ipt_scaler_path = os.path.join(train_data_path, "stand", f"ipt_scaler-{dataset_config.ipt_scaler}-ipt_min{dataset_config.ipt_min}-ipt_max{dataset_config.ipt_max}.pickle")
        flowstats_quantiles_path = os.path.join(train_data_path, "stand", f"flowstats_quantiles-q{dataset_config.flowstats_clip}.pickle")
        if os.path.isfile(flowstats_scaler_path) and os.path.isfile(flowstats_quantiles_path) and os.path.isfile(ipt_scaler_path) and os.path.isfile(psizes_sizes_scaler_path):
            flowstats_scaler = pickle_load(flowstats_scaler_path)
            psizes_scaler = pickle_load(psizes_sizes_scaler_path)
            ipt_scaler = pickle_load(ipt_scaler_path)
            flowstats_quantiles = pickle_load(flowstats_quantiles_path)
            return flowstats_scaler, psizes_scaler, ipt_scaler, flowstats_quantiles
    # Initialize the scalers classes based on the config
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
    # Load scalers learned attributes from config if provided
    if dataset_config.scalers_attrs is not None:
        if "flowstats_scaler_attrs" in dataset_config.scalers_attrs:
            if flowstats_scaler is not None:
                set_scaler_attrs(flowstats_scaler, dataset_config.scalers_attrs["flowstats_scaler_attrs"])
            else:
                warnings.warn("Ignoring flowstats_scaler_attrs because flowstats_scaler is None")
        if "psizes_scaler_attrs" in dataset_config.scalers_attrs:
            if psizes_scaler is not None:
                set_scaler_attrs(psizes_scaler, dataset_config.scalers_attrs["psizes_scaler_attrs"])
            else:
                warnings.warn("Ignoring psizes_scaler_attrs because psizes_scaler is None")
        if "ipt_scaler_attrs" in dataset_config.scalers_attrs:
            if ipt_scaler is not None:
                set_scaler_attrs(ipt_scaler, dataset_config.scalers_attrs["ipt_scaler_attrs"])
            else:
                warnings.warn("Ignoring ipt_scaler_attrs because ipt_scaler is None")
        assert "flowstats_quantiles" in dataset_config.scalers_attrs
        flowstats_quantiles = np.array(dataset_config.scalers_attrs["flowstats_quantiles"])
        return flowstats_scaler, psizes_scaler, ipt_scaler, flowstats_quantiles
    # If the scalers are not loaded at this point, fit them
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
        psizes_scaler=psizes_scaler,
        ipt_scaler=ipt_scaler,
        flowstats_quantile_clip=dataset_config.flowstats_clip,
        ipt_min=dataset_config.ipt_min,
        ipt_max=dataset_config.ipt_max,
        psizes_max=dataset_config.psizes_max)
    pickle_dump(flowstats_scaler, flowstats_scaler_path)
    pickle_dump(psizes_scaler, psizes_sizes_scaler_path)
    pickle_dump(ipt_scaler, ipt_scaler_path)
    pickle_dump(flowstats_quantiles, flowstats_quantiles_path)
    return flowstats_scaler, psizes_scaler, ipt_scaler, flowstats_quantiles

def fit_scalers(database_path: str,
                train_tables_paths: list[str],
                fit_scalers_indices: np.ndarray,
                flowstats_scaler: Scaler,
                psizes_scaler: Scaler,
                ipt_scaler: Scaler,
                flowstats_quantile_clip: float,
                ipt_min: int,
                ipt_max: int,
                psizes_max: int) -> np.ndarray:
    start_time = time.time()
    database, tables = load_database(database_path, tables_paths=train_tables_paths)
    data = load_data_from_tables(tables=tables, indices=fit_scalers_indices, data_dtype=tables[0].dtype)
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
    # Flow statistics
    train_flowstats = structured_to_unstructured(data[FLOWSTATS_TO_SCALE])
    flowstats_quantiles = np.quantile(train_flowstats, q=flowstats_quantile_clip, axis=0)
    flowstats_quantiles[-len(FLOWSTATS_NO_CLIP):] = np.inf # disable clipping for features with "fixed" range
    if flowstats_scaler:
        train_flowstats = train_flowstats.clip(max=flowstats_quantiles)
        flowstats_scaler.fit(train_flowstats)
    log.info(f"Reading data and fitting scalers took {time.time() - start_time:.2f} seconds")
    return flowstats_quantiles