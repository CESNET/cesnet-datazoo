# Getting started

## Jupyter notebooks
Example Jupyter notebooks are provided at [https://github.com/CESNET/cesnet-tcexamples](https://github.com/CESNET/cesnet-tcexamples). Start with:

* Initialize the CESNET-QUIC22 dataset and explore its data features - [explore_data.ipynb](https://nbviewer.org/github/CESNET/cesnet-tcexamples/blob/main/notebooks/explore_data.ipynb)
* Training of a LightGBM classifier and its evaluation on a per-week and per-day basis - [example_evaluation.ipynb](https://nbviewer.org/github/CESNET/cesnet-tcexamples/blob/main/notebooks/example_evaluation.ipynb)
* Training of a neural network from the `cesnet-models` package with data transformations - [example_train_nn.ipynb](https://nbviewer.org/github/CESNET/cesnet-tcexamples/blob/main/notebooks/example_train_nn.ipynb)

## Code snippets

### Download a dataset and compute statistics
```python
from cesnet_datazoo.datasets import CESNET_QUIC22
dataset = CESNET_QUIC22("/datasets/CESNET-QUIC22/", size="XS")
dataset.compute_dataset_statistics(num_samples=100_000, num_workers=0)
```
This code will download the dataset, compute dataset statistics, and save them into `/datasets/CESNET-QUIC22/statistics`.

### Enable logging and set the spawn method on Windows
```python
import logging
import multiprocessing as mp

mp.set_start_method("spawn") 
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
```
For running on Windows, we recommend using the `spawn` method for creating dataloader worker processes. Set up logging to get more information from the package.

### Initialize dataset to create train, validation, and test dataframes

```python
from cesnet_datazoo.datasets import CESNET_QUIC22
from cesnet_datazoo.config import DatasetConfig, AppSelection

dataset = CESNET_QUIC22("/datasets/CESNET-QUIC22/", size="XS")
dataset_config = DatasetConfig(
    dataset=dataset,
    apps_selection=AppSelection.ALL_KNOWN,
    train_period_name="W-2022-44",
    test_period_name="W-2022-45",
)
dataset.set_dataset_config_and_initialize(dataset_config)
train_dataframe = dataset.get_train_df()
val_dataframe = dataset.get_val_df()
test_dataframe = dataset.get_test_df()
```

The [`DatasetConfig`][config.DatasetConfig] class handles the configuration of datasets, and calling `set_dataset_config_and_initialize` initializes train, validation, and test sets with the desired configuration.
Data can be read into Pandas DataFrames as shown here or via PyTorch DataLoaders. See [`CesnetDataset`][datasets.cesnet_dataset.CesnetDataset] reference.