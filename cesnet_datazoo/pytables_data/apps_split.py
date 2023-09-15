import pandas as pd

from cesnet_datazoo.constants import SERVICEMAP_PROVIDER_COLUMN


def is_background_app(app: str) -> bool:
    return app.endswith("-background")

def create_provider_groups(apps: list[str], servicemap: pd.DataFrame) -> dict[str, list[str]]:
    provider_groups_dict = {
        app: [] if pd.isnull(servicemap.loc[app, SERVICEMAP_PROVIDER_COLUMN])
                else [x for x in servicemap[servicemap[SERVICEMAP_PROVIDER_COLUMN]==servicemap.loc[app, SERVICEMAP_PROVIDER_COLUMN]].index if x in apps]
            for app in servicemap.index}
    return provider_groups_dict

def split_apps_topx_with_provider_groups(sorted_apps: list[str], known_count: int, servicemap: pd.DataFrame) -> tuple[list[str], list[str]]:
    known_apps = []
    provider_groups_dict = create_provider_groups(apps=sorted_apps, servicemap=servicemap)
    for app in sorted_apps:
        if len(known_apps) == known_count:
            break
        if is_background_app(app):
            continue
        if app in known_apps:
            continue
        provider_group = provider_groups_dict[app]
        if not provider_group:
            known_apps.append(app)
        elif len(known_apps) + len(provider_group) <= known_count:
            known_apps.extend(provider_group)
    unknown_apps  = [x for x in sorted_apps if x not in known_apps]
    return known_apps, unknown_apps
