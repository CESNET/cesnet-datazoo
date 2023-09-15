from dataclasses import dataclass

import numpy as np
import pandas as pd

from cesnet_datazoo.constants import SERVICEMAP_CATEGORY_COLUMN, SERVICEMAP_PROVIDER_COLUMN


@dataclass()
class ClassInfo:
    target_names: list[str]
    known_apps: list[str]
    group_matrix: np.ndarray
    superclass_members: dict[str, list[str]]
    has_superclass: dict[str, bool]
    superclass_mapping: dict[str, str]
    superclass_mapping_arr: np.ndarray
    categories_mapping: dict[str, str]

    def get_num_classes(self):
        return len(self.known_apps)

def create_superclass_structures(servicemap: pd.DataFrame, target_names: list[str]) -> ClassInfo:
    known_apps = target_names[:-1]
    target_names_arr = np.array(target_names)
    group_matrix = np.array([[
                a in servicemap.index and b in servicemap.index and
                not pd.isnull(servicemap.loc[a, SERVICEMAP_PROVIDER_COLUMN]) and not pd.isnull(servicemap.loc[b, SERVICEMAP_PROVIDER_COLUMN]) and
                servicemap.loc[a, SERVICEMAP_PROVIDER_COLUMN] == servicemap.loc[b, SERVICEMAP_PROVIDER_COLUMN]
                for a in target_names_arr] for b in target_names_arr])
    has_superclass = {app: app in servicemap.index and not pd.isnull(servicemap.loc[app, SERVICEMAP_PROVIDER_COLUMN]) for app in target_names_arr}
    superclass_mapping: dict[str, str] = {app: servicemap.loc[app, SERVICEMAP_PROVIDER_COLUMN] if has_superclass[app] else app for app in target_names_arr} # type: ignore
    superclass_mapping_arr = np.array(list(superclass_mapping.values()))
    superclass_members = {superclass: servicemap.loc[servicemap[SERVICEMAP_PROVIDER_COLUMN] == superclass].index.to_list()
                    for superclass in servicemap.loc[:, SERVICEMAP_PROVIDER_COLUMN].dropna().unique()}
    categories_mapping: dict[str, str] = {app: servicemap.loc[app, SERVICEMAP_CATEGORY_COLUMN] if app in servicemap.index else None for app in target_names_arr} # type: ignore
    return ClassInfo(
            target_names=target_names,
            known_apps=known_apps,
            group_matrix=group_matrix,
            superclass_members=superclass_members,
            has_superclass=has_superclass,
            superclass_mapping=superclass_mapping,
            superclass_mapping_arr=superclass_mapping_arr,
            categories_mapping=categories_mapping,
    )
