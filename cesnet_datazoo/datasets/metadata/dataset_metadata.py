import os

import pandas as pd
from pydantic import ValidationInfo, field_validator
from pydantic.dataclasses import dataclass

from cesnet_datazoo.config import Protocol


@dataclass()
class DatasetMetadata():
    protocol: Protocol
    published_in: int
    collected_in: int
    collection_duration: str
    available_samples: int
    available_dataset_sizes: list[str]
    collection_period: str
    missing_dates_in_collection_period: list[str]
    application_count: int
    background_traffic_classes: list[str]
    ppi_features: list[str]
    flowstats_features: list[str]
    flowstats_features_boolean: list[str]
    packet_histograms: list[str]
    tcp_features: list[str]
    other_fields: list[str]
    cite: str
    zenodo_url: str
    related_papers: list[str]

    @field_validator("available_dataset_sizes", "missing_dates_in_collection_period", "background_traffic_classes", "ppi_features",
                     "flowstats_features", "flowstats_features_boolean", "packet_histograms", "tcp_features", "other_fields", "related_papers", mode="before")
    @classmethod
    def parse_string_to_list(cls, v: str, info: ValidationInfo) -> list[str]:
        l = list(map(str.strip, v.split(","))) if v else []
        return l

metadata_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "metadata.csv"), index_col="Name", keep_default_na=False)
def load_metadata(dataset_name: str) -> DatasetMetadata:
    d = metadata_df.loc[dataset_name].to_dict()
    d = {k.replace(" ", "_").lower(): v for k, v in d.items()} # type: ignore
    return DatasetMetadata(**d)
