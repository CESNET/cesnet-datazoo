<p align="center">
    <img src="https://raw.githubusercontent.com/CESNET/cesnet-datazoo/main/docs/images/datazoo.svg" width="450">
</p>

[![](https://img.shields.io/badge/license-BSD-blue.svg)](https://github.com/CESNET/cesnet-datazoo/blob/main/LICENCE)
[![](https://img.shields.io/badge/docs-cesnet--datazoo-blue.svg)](https://cesnet.github.io/cesnet-datazoo/)
[![](https://img.shields.io/badge/python->=3.10-blue.svg)](https://pypi.org/project/cesnet-datazoo/)
[![](https://img.shields.io/pypi/v/cesnet-datazoo)](https://pypi.org/project/cesnet-datazoo/)
[![](https://img.shields.io/uptimerobot/status/m801936469-e8219ca3245b73b08cf33ef4?label=storage%20status)](https://stats.uptimerobot.com/6a75HRSoRU)


The goal of this project is to provide tools for working with large network traffic datasets and to facilitate research in the traffic classification area. The core functions of the `cesnet-datazoo` package are:

- A common API for downloading, configuring, and loading of four public datasets of encrypted network traffic.
- Extensive configuration options for:
    - Selection of train, validation, and test periods.
    - Selection of application classes and splitting classes between *known* and *unknown*.
    - Data transformations, such as feature scaling.
- Built on suitable data structures for experiments with large datasets. There are several caching mechanisms to make repeated runs faster, for example, when searching for the best model configuration.
- Datasets are offered in multiple sizes to give users an option to start the experiments at a smaller scale (also faster dataset download, disk space, etc.). The default is the `S` size containing 25 million samples.

:brain: :brain: See a related project [CESNET Models](https://github.com/CESNET/cesnet-models) providing pretrained neural networks for traffic classification. :brain: :brain:

:notebook: :notebook: Example Jupyter notebooks are included in a separate [Traffic Classification Examples](https://github.com/CESNET/cesnet-tcexamples) repository. :notebook: :notebook:

:rocket: :rocket: [Transfer Learning Codebase](https://github.com/CESNET/tc-transfer/) for reproducing experiments from our paper — covering ten downstream traffic classification tasks with three transfer approaches (k-NN, linear probing, and full model fine-tuning). :rocket: :rocket:

## Datasets
The `cesnet-datazoo` package currently provides four datasets with details in the following table (you might need to scroll the table horizontally to see all datasets).

1. CESNET-TLS22
2. CESNET-QUIC22
3. CESNET-TLS-Year22
4. CESNET-QUICEXT-25

| Name                                 | CESNET-TLS22                                                                   | CESNET-QUIC22                                                                         | CESNET-TLS-Year22                                                         | CESNET-QUICEXT-25                                                   |
|--------------------------------------|--------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|---------------------------------------------------------------------------|---------------------------------------------------------------------|
| _Protocol_                           | TLS                                                                            | QUIC                                                                                  | TLS                                                                       | QUIC                                                                |
| _Published in_                       | 2022                                                                           | 2023                                                                                  | 2024                                                                      | 2025                                                                |
| _Collection duration_                | 2 weeks                                                                        | 4 weeks                                                                               | 1 year                                                                    | 1 year                                                              |
| _Collection period_                  | 4.10.2021 - 17.10.2021                                                         | 31.10.2022 - 27.11.2022                                                               | 1.1.2022 - 31.12.2022                                                     | 1.6.2024 - 31.5.2025                                                |
| _Application count_                  | 191                                                                            | 102                                                                                   | 180                                                                       | 50                                                                  |
| _Available samples_                  | 141392195                                                                      | 153226273                                                                             | 507739073                                                                 | 194296462                                                           |
| _Available dataset sizes_            | XS, S, M, L                                                                    | XS, S, M, L                                                                           | XS, S, M, L                                                               | XS, S, M, L                                                         |
| _Cite_                               | [10.1016/j.comnet.2022.109467](https://doi.org/10.1016/j.comnet.2022.109467)   | [10.1016/j.dib.2023.108888](https://doi.org/10.1016/j.dib.2023.108888)                | [10.1038/s41597-024-03927-4](https://doi.org/10.1038/s41597-024-03927-4)  | *In preparation*                                                    |
| _Zenodo URL_                         | [zenodo.org/record/7965515](https://zenodo.org/record/7965515)                 | [zenodo.org/record/7963302](https://zenodo.org/record/7963302)                        | [zenodo.org/records/10608607](https://zenodo.org/records/10608607)        | [zenodo.org/records/17249078](https://zenodo.org/records/17249078)  |
| _Related papers_                     |                                                                                | [10.23919/TMA58422.2023.10199052](https://doi.org/10.23919/TMA58422.2023.10199052)    |                                                                           | [10.1145/3768988](https://doi.org/10.1145/3768988)                  |

## Installation

Install the package from pip with:

```bash
pip install cesnet-datazoo
```

or for editable install with:

```bash
pip install -e git+https://github.com/CESNET/cesnet-datazoo
```

## Examples
#### Initialize dataset to create train, validation, and test dataframes

```py
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

The [`DatasetConfig`](https://cesnet.github.io/cesnet-datazoo/reference_dataset_config/) class handles the configuration of datasets, and calling `set_dataset_config_and_initialize` initializes train, validation, and test sets with the desired configuration.
Data can be read into Pandas DataFrames as shown here or via PyTorch DataLoaders. See [`CesnetDataset`](https://cesnet.github.io/cesnet-datazoo/reference_cesnet_dataset/) reference.

See more examples in the [documentation](https://cesnet.github.io/cesnet-datazoo/getting_started/).

## Papers

* [DataZoo: Streamlining Traffic Classification Experiments](https://doi.org/10.1145/3630050.3630176) <br>
Jan Luxemburk and Karel Hynek <br>
CoNEXT Workshop on Explainable and Safety Bounded, Fidelitous, Machine Learning for Networking (SAFE), 2023

* [CESNET-TLS-Year22: A year-spanning TLS network traffic dataset from backbone lines](https://doi.org/10.1038/s41597-024-03927-4) <br>
Karel Hynek, Jan Luxemburk, Jaroslav Pešek, Tomáš Čejka, and Pavel Šiška  <br>
Scientific Data (Nature Portfolio), 2024

* [CESNET-QUIC22: A large one-month QUIC network traffic dataset from backbone lines](https://doi.org/10.1016/j.dib.2023.108888) <br>
Jan Luxemburk, Karel Hynek, Tomáš Čejka, Andrej Lukačovič, and Pavel Šiška  <br>
Data in Brief, 2023

## Acknowledgments

This project was supported by the Ministry of the Interior of the Czech Republic, grant No. VJ02010024: Flow-Based Encrypted Traffic Analysis.