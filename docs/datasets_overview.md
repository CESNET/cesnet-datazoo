# Overview of datasets

## CESNET-TLS22

!!! info inline end "CESNET-TLS22"
    - TLS protocol
    - Collected in 2021
    - Spans two weeks
    - Contains 141 million samples
    - Has 191 application classes

This dataset was published in *"Fine-grained TLS services classification with reject option"* ([DOI](https://doi.org/10.1016/j.comnet.2022.109467), [arXiv](https://arxiv.org/abs/2202.11984)). It was built from live traffic collected using high-speed monitoring probes at the perimeter of the CESNET2 network.

For detailed information about the dataset, see the linked paper and the [dataset metadata][metadata] page.

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

For detailed information about the dataset, see the linked paper and the [dataset metadata][metadata] page. Experiments based on this dataset were published in *"Encrypted traffic classification: the QUIC case"* ([DOI](https://ieeexplore.ieee.org/abstractdocument/10199052/)).

## CESNET-TLS-Year22

!!! info inline end "CESNET-TLS-Year22"
    - TLS protocol
    - Collected in 2022
    - Spans one year
    - Contains 507 million samples
    - Has 182 application classes

This dataset is similar to CESNET-TLS22; however, it spans the *entire year 2022*. It will be published in the near future.
