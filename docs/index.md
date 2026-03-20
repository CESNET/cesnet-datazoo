# CESNET DataZoo

This is documentation of the [CESNET DataZoo](https://github.com/CESNET/cesnet-datazoo) project. 

The goal of this project is to provide tools for working with large network traffic datasets and to facilitate research in the traffic classification area. The core functions of the `cesnet-datazoo` package are:

- A common API for downloading, configuring, and loading of four public datasets of encrypted network traffic — CESNET-TLS22, CESNET-QUIC22, CESNET-TLS-Year22, and CESNET-QUICEXT-25. Details about the available datasets are on the [dataset overview][overview-of-datasets] page.
- Provides standard features used for traffic classification, such as sizes, directions, and inter-packet times of the first 30 packets of each flow. More details on the [data features][features] page.
- Extensive configuration options for:
    - Selection of train, validation, and test periods. The datasets span from two weeks to one year; therefore, it is possible to evaluate classification methods in a time-based fashion that is closer to practical deployment.
    - Selection of application classes and splitting classes between *known* and *unknown*. This enables research in the open-world setting, in which classification models need to handle new classes that were not seen during the training process.
    - Data transformations, such as feature scaling. Transforms are implemented in a separate package [CESNET Models](https://github.com/CESNET/cesnet-models). See `cesnet_models.transforms` [documentation](https://cesnet.github.io/cesnet-models/reference_transforms/) for details.
- Built on suitable data structures for experiments with large datasets. There are several caching mechanisms to make repeated runs faster, for example, when searching for the best model configuration.
- Datasets are offered in multiple sizes to give users an option to start experiments at a smaller scale (also faster dataset download, disk space, etc.). The default is the `S` size containing 25 million samples. 

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