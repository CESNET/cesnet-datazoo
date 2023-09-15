import os
from dataclasses import fields

import pandas as pd
from pydantic.config import Extra
from pydantic.dataclasses import dataclass

from cesnet_datazoo.config import Protocol


class C:
    extra = Extra.forbid

@dataclass(config=C)
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
    background_traffic: list[str]
    features_in_packet_sequences: list[str]
    packet_histogram_features: list[str]
    flowstats_features: list[str]
    tcp_features: list[str]
    other_fields: list[str]
    cite: str
    zenodo_url: str
    related_papers: list[str]

    def __post_init__(self):
        for f in fields(DatasetMetadata):
            if f.type == list[str]:
                value =  getattr(self, f.name)
                setattr(self, f.name, list(map(str.strip, value.split(","))) if value else [])

metadata_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "metadata.csv"), index_col="Name", keep_default_na=False)
def load_metadata(dataset_name: str) -> DatasetMetadata:
    d = metadata_df.loc[dataset_name].to_dict()
    d = {k.replace(" ", "_").lower(): v for k, v in d.items()}
    return DatasetMetadata(**d)
