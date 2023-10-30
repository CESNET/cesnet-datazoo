from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from cesnet_datazoo.constants import SERVICEMAP_CATEGORY_COLUMN, SERVICEMAP_PROVIDER_COLUMN


@dataclass()
class ClassInfo:
    target_names: list[str]
    num_classes: int
    known_apps: list[str]
    unknown_apps: list[str]
    unknown_class_label: int
    group_matrix: np.ndarray
    has_provider: dict[str, bool]
    provider_mapping: dict[str, str]
    provider_members: dict[str, list[str]]
    categories_mapping: dict[str, Optional[str]]

def create_class_info(servicemap: Any, encoder: LabelEncoder, known_apps_database_enum: dict[int, str], unknown_apps_database_enum: dict[int, str]) -> ClassInfo:
    known_apps = sorted(known_apps_database_enum.values())
    unknown_apps = sorted(unknown_apps_database_enum.values())
    target_names_arr = encoder.classes_
    assert known_apps == list(target_names_arr[:-1])
    group_matrix = np.array([[a == b or
                (a in servicemap.index and b in servicemap.index and
                not pd.isnull(servicemap.loc[a, SERVICEMAP_PROVIDER_COLUMN]) and not pd.isnull(servicemap.loc[b, SERVICEMAP_PROVIDER_COLUMN]) and
                servicemap.loc[a, SERVICEMAP_PROVIDER_COLUMN] == servicemap.loc[b, SERVICEMAP_PROVIDER_COLUMN])
                for a in target_names_arr] for b in target_names_arr])
    has_provider = {app: app in servicemap.index and not pd.isnull(servicemap.loc[app, SERVICEMAP_PROVIDER_COLUMN]) for app in target_names_arr}
    provider_mapping = {app: servicemap.loc[app, SERVICEMAP_PROVIDER_COLUMN] if has_provider[app] else app for app in target_names_arr}
    providers = sorted({provider_mapping[app] for app in target_names_arr if has_provider[app]})
    provider_members = {p: [app for app in target_names_arr if provider_mapping[app] == p] for p in providers}
    categories_mapping = {app: servicemap.loc[app, SERVICEMAP_CATEGORY_COLUMN] if app in servicemap.index else None for app in target_names_arr}
    return ClassInfo(
            target_names=list(target_names_arr),
            num_classes=len(known_apps),
            known_apps=known_apps,
            unknown_apps=unknown_apps,
            unknown_class_label=len(known_apps),
            group_matrix=group_matrix,
            has_provider=has_provider,
            provider_mapping=provider_mapping,
            provider_members=provider_members,
            categories_mapping=categories_mapping,
    )
