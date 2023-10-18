
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from cesnet_datazoo.constants import APP_COLUMN


def load_from_dataloader(dataloader: DataLoader, silent: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_ppi = []
    data_flowstats = []
    labels = []
    if not silent:
        print("Loading data from dataloader")
    for batch_ppi, batch_flowstats, batch_labels in tqdm(dataloader, total=len(dataloader), disable=silent):
        data_ppi.append(batch_ppi)
        data_flowstats.append(batch_flowstats)
        labels.append(batch_labels)
    data_ppi = np.concatenate(data_ppi)
    data_flowstats = np.concatenate(data_flowstats)
    labels = np.concatenate(labels)
    return data_ppi, data_flowstats, labels

def create_df_from_dataloader(dataloader: DataLoader, feature_names: list[str], flatten_ppi: bool = False, silent: bool = False) -> pd.DataFrame:
    data_ppi, data_flowstats, labels = load_from_dataloader(dataloader, silent=silent)
    if flatten_ppi:
        data_ppi = data_ppi.reshape(data_ppi.shape[0], -1)
        data = np.column_stack((data_ppi, data_flowstats))
        df = pd.DataFrame(data=data, columns=feature_names)
    else:
        ppi_column, *feature_names = feature_names
        df = pd.DataFrame(data=data_flowstats, columns=feature_names)
        df.insert(0, column=ppi_column, value=list(data_ppi))
    df[APP_COLUMN] = labels
    return df
