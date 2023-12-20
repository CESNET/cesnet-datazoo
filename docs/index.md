# CESNET DataZoo

This is the documentation of the [CESNET DataZoo](https://github.com/CESNET/cesnet-datazoo) project. 

The goal of this project is to provide tools for working with large network traffic datasets and to facilitate research in the traffic classification area. The core functions of the `cesnet-datazoo` package are:

- A common API for downloading, configuring, and loading of three public datasets of encrypted network traffic — CESNET-TLS22, CESNET-QUIC22, and CESNET-TLS-Year22. Details about the available datasets are on the [dataset overview][overview-of-datasets] page.
- Provides standard features used for traffic classification, such as sizes, directions, and inter-packet times of the first 30 packets of each flow. More details on the [data features][features] page.
- Extensive configuration options for:
    - Selection of train, validation, and test periods. The datasets span from two weeks to one year; therefore, it is possible to evaluate classification methods in a time-based fashion that is closer to practical deployment.
    - Selection of application classes and splitting classes between *known* and *unknown*. This enables research in the open-world setting, in which classification models need to handle new classes that were not seen during the training process.
    - Feature scaling.
- Built on suitable data structures for experiments with large datasets. There are several caching mechanisms to make repeated runs faster, for example, when searching for the best model configuration.
- Datasets are offered in multiple sizes to give users an option to start experiments at a smaller scale (also faster dataset download, disk space, etc.). The default is the `S` size containing 25 million samples. 

## Papers

* [DataZoo: Streamlining Traffic Classification Experiments](https://doi.org/10.1145/3630050.3630176) <br>
Jan Luxemburk and Karel Hynek <br>
CoNEXT Workshop on Explainable and Safety Bounded, Fidelitous, Machine Learning for Networking (SAFE), 2023