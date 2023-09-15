
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from cesnet_datazoo.constants import APP_COLUMN


def load_from_dataloader(dataloader: DataLoader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_ppi = []
    data_flowstats = []
    labels = []
    print("Loading data from dataloader")
    for *x_batch, y_batch in tqdm(dataloader, total=len(dataloader)):
        ppi, flowstats  = x_batch
        data_ppi.append(ppi)
        data_flowstats.append(flowstats)
        labels.append(y_batch)
    data_ppi = np.concatenate(data_ppi)
    data_flowstats = np.concatenate(data_flowstats)
    labels = np.concatenate(labels)
    return data_ppi, data_flowstats, labels

def create_df_from_dataloader(dataloader: DataLoader, feature_names: list[str], flatten_ppi: bool = False) -> pd.DataFrame:
    data_ppi, data_flowstats, labels = load_from_dataloader(dataloader)

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