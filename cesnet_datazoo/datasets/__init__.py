from typing import Type

from .datasets import CESNET_QUIC22, CESNET_TLS22, CESNET_TLS_Year22, CesnetDataset

AVAILABLE_DATASETS: dict[str, Type[CesnetDataset]] = {
    "CESNET-TLS22": CESNET_TLS22,
    "CESNET-QUIC22": CESNET_QUIC22,
    "CESNET-TLS-Year22": CESNET_TLS_Year22
}
