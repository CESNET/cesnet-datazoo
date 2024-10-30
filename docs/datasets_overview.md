# Overview of datasets

## CESNET-TLS22

!!! info inline end "CESNET-TLS22"
    - TLS protocol
    - Collected in 2021
    - Spans two weeks
    - Contains 141 million samples
    - Has 191 application classes

This dataset was published in *"Fine-grained TLS services classification with reject option"* ([DOI](https://doi.org/10.1016/j.comnet.2022.109467), [arXiv](https://arxiv.org/abs/2202.11984)). It was built from live traffic collected using high-speed monitoring probes at the perimeter of the CESNET2 network.

For detailed information about the dataset, please refer to the linked paper and the [dataset metadata][metadata] page.

## CESNET-QUIC22

!!! info inline end "CESNET-QUIC22"
    - QUIC protocol
    - Collected in 2022
    - Spans four weeks
    - Contains 153 million samples
    - Has 102 application classes and three background traffic classes

This dataset was published in *"CESNET-QUIC22: A large one-month QUIC network traffic dataset from backbone lines"* ([DOI](https://doi.org/10.1016/j.dib.2023.108888)).
The QUIC protocol has the potential to replace TLS over TLS as the standard protocol for reliable and secure Internet communication.
Due to its design that makes the inspection of connection handshakes challenging and its usage in HTTP/3, there is an increasing demand for QUIC traffic classification methods.

For detailed information about the dataset, please refer to the linked paper and the [dataset metadata][metadata] page. Experiments based on this dataset were published in *"Encrypted traffic classification: the QUIC case"* ([DOI](https://ieeexplore.ieee.org/abstractdocument/10199052/)).

## CESNET-TLS-Year22

!!! info inline end "CESNET-TLS-Year22"
    - TLS protocol
    - Collected in 2022
    - Spans one year
    - Contains 507 million samples
    - Has 180 application classes

Our newest dataset was published in the Scientific Data journal - *"CESNET-TLS-Year22: A year-spanning TLS network traffic dataset from backbone lines"* ([DOI](https://doi.org/10.1038/s41597-024-03927-4)).
This dataset is similar to CESNET-TLS22, containing almost the same classes (180 of the 191 application classes from CESNET-TLS22), while also providing additional features and covering the entire year of 2022. It is suitable for a comprehensive evaluation of traffic classification models and an assessment of their robustness in the ever-evolving environment of production networks.

For detailed information about the dataset, please refer to the linked paper and the [dataset metadata][metadata] page.
