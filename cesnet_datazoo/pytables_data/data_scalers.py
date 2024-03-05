import json
import logging
import os
import time
import warnings

import numpy as np
from cesnet_models.transforms import ClipAndScaleFlowstats, ClipAndScalePPI
from numpy.lib.recfunctions import structured_to_unstructured
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from cesnet_datazoo.config import DatasetConfig
from cesnet_datazoo.constants import DIR_POS, FLOWSTATS_NO_CLIP, IPT_POS, PPI_COLUMN, SIZE_POS
from cesnet_datazoo.pytables_data.pytables_dataset import load_data_from_tables, load_database
from cesnet_datazoo.utils.random import RandomizedSection, get_fresh_random_generator

log = logging.getLogger(__name__)


def get_scaler_attrs(scaler: StandardScaler | RobustScaler | MinMaxScaler) -> dict[str, list[float]]:
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

def fit_scalers(dataset_config: DatasetConfig, train_indices: np.ndarray) -> None:
    # Define indices for fitting scalers
    if isinstance(dataset_config.fit_scalers_samples, int) and dataset_config.fit_scalers_samples > len(train_indices):
        warnings.warn(f"The number of samples for fitting scalers ({dataset_config.fit_scalers_samples}) is larger than the number of train samples ({len(train_indices)}), using the number of train samples instead")
        dataset_config.fit_scalers_samples = len(train_indices)
    fit_scalers_rng = get_fresh_random_generator(dataset_config=dataset_config, section=RandomizedSection.FIT_SCALERS_SAMPLE)
    if isinstance(dataset_config.fit_scalers_samples, float):
        num_samples = int(dataset_config.fit_scalers_samples * len(train_indices))
    else:
        num_samples = dataset_config.fit_scalers_samples
    fit_scalers_indices = train_indices[fit_scalers_rng.choice(len(train_indices), size=num_samples, replace=False)]
    # Load data
    start_time = time.time()
    database, tables = load_database(dataset_config.database_path, tables_paths=dataset_config._get_train_tables_paths())
    data = load_data_from_tables(tables=tables, indices=fit_scalers_indices, data_dtype=tables[0].dtype)
    database.close()

    clip_and_scale_ppi_transform = dataset_config.ppi_transform # TODO Fix after transforms composing is implemented
    clip_and_scale_flowstats_transform = dataset_config.flowstats_transform

    # Fit the ClipAndScalePPI transform
    if clip_and_scale_ppi_transform is not None and clip_and_scale_ppi_transform.needs_fitting:
        assert isinstance(clip_and_scale_ppi_transform, ClipAndScalePPI)
        data_ppi = data[PPI_COLUMN].astype("float32")
        ppi_channels = data_ppi.shape[1]
        data_ppi = data_ppi.transpose(0, 2, 1).reshape(-1, ppi_channels)
        padding_mask = data_ppi[:, DIR_POS] == 0 # Mask of padded packets
        # Fit IPT scaler
        train_ipt = data_ppi[:, IPT_POS].clip(max=clip_and_scale_ppi_transform.ipt_max, min=clip_and_scale_ppi_transform.ipt_min)
        train_ipt[padding_mask] = np.nan # NaNs are ignored in sklearn scalers
        if isinstance(clip_and_scale_ppi_transform.ipt_scaler, MinMaxScaler):
            # Let zero be the minimum for minmax scaling
            train_ipt = np.concatenate((train_ipt, [0]))
        clip_and_scale_ppi_transform.ipt_scaler.fit(train_ipt.reshape(-1, 1))
        # Fit packet sizes scaler
        train_psizes = data_ppi[:, SIZE_POS].clip(max=clip_and_scale_ppi_transform.psizes_max, min=clip_and_scale_ppi_transform.pszies_min)
        train_psizes[padding_mask] = np.nan
        if isinstance(clip_and_scale_ppi_transform.psizes_scaler, MinMaxScaler):
            train_psizes = np.concatenate((train_psizes, [0]))
        clip_and_scale_ppi_transform.psizes_scaler.fit(train_psizes.reshape(-1, 1))
        clip_and_scale_ppi_transform.needs_fitting = False

    # Fit the ClipAndScaleFlowstats transform
    if clip_and_scale_flowstats_transform is not None and clip_and_scale_flowstats_transform.needs_fitting:
        assert isinstance(clip_and_scale_flowstats_transform, ClipAndScaleFlowstats)
        train_flowstats = structured_to_unstructured(data[dataset_config.flowstats_features])
        flowstats_quantiles = np.quantile(train_flowstats, q=clip_and_scale_flowstats_transform.quantile_clip, axis=0)
        idx_no_clip = [dataset_config.flowstats_features.index(f) for f in FLOWSTATS_NO_CLIP]
        flowstats_quantiles[idx_no_clip] = np.inf # Disable clipping for features with "fixed" range
        train_flowstats = train_flowstats.clip(max=flowstats_quantiles)
        clip_and_scale_flowstats_transform.flowstats_scaler.fit(train_flowstats)
        clip_and_scale_flowstats_transform.flowstats_quantiles = flowstats_quantiles.tolist()
        clip_and_scale_flowstats_transform.needs_fitting = False

    log.info(f"Reading data and fitting scalers took {time.time() - start_time:.2f} seconds")
    train_data_path = dataset_config._get_train_data_path()
    if clip_and_scale_ppi_transform is not None:
        ppi_transform_path = os.path.join(train_data_path, "transforms", "ppi-transform.json")
        ppi_transform_dict = {
            "psizes_scaler_enum": str(clip_and_scale_ppi_transform._psizes_scaler_enum),
            "psizes_scaler_attrs": get_scaler_attrs(clip_and_scale_ppi_transform.psizes_scaler),
            "pszies_min": clip_and_scale_ppi_transform.pszies_min,
            "psizes_max": clip_and_scale_ppi_transform.psizes_max,
            "ipt_scaler_enum": str(clip_and_scale_ppi_transform._ipt_scaler_enum),
            "ipt_scaler_attrs": get_scaler_attrs(clip_and_scale_ppi_transform.ipt_scaler),
            "ipt_min": clip_and_scale_ppi_transform.ipt_min,
            "ipt_max": clip_and_scale_ppi_transform.ipt_max,
        }
        json.dump(ppi_transform_dict, open(ppi_transform_path, "w"), indent=4)
    if clip_and_scale_flowstats_transform is not None:
        assert clip_and_scale_flowstats_transform.flowstats_quantiles is not None
        flowstats_transform_path = os.path.join(train_data_path, "transforms", "flowstats-transform.json")
        flowstats_transform_dict = {
            "flowstats_scaler_enum": str(clip_and_scale_flowstats_transform._flowstats_scaler_enum),
            "flowstats_scaler_attrs": get_scaler_attrs(clip_and_scale_flowstats_transform.flowstats_scaler),
            "flowstats_quantiles": clip_and_scale_flowstats_transform.flowstats_quantiles,
            "quantile_clip": clip_and_scale_flowstats_transform.quantile_clip,
        }
        json.dump(flowstats_transform_dict, open(flowstats_transform_path, "w"), indent=4)
