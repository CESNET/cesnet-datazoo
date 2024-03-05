import os
from collections import Counter
from typing import Any, Literal

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from numpy.lib.recfunctions import structured_to_unstructured
from torch.utils.data import BatchSampler, DataLoader, SequentialSampler
from tqdm import tqdm

from cesnet_datazoo.config import Protocol
from cesnet_datazoo.constants import (APP_COLUMN, CATEGORY_COLUMN, IPT_POS, PPI_COLUMN, SIZE_POS,
                                      UDP_PPI_CHANNELS)
from cesnet_datazoo.pytables_data.indices_setup import sort_indices
from cesnet_datazoo.pytables_data.pytables_dataset import (PyTablesDataset, list_all_tables,
                                                           load_database, worker_init_fn)


def pick_quic_fields(batch):
    return (
        batch["QUIC_SNI"],
        batch["QUIC_USERAGENT"],
        batch["QUIC_VERSION"],
    )

def pick_stats_fields(batch):
    return (
        batch[PPI_COLUMN],
        batch["DURATION"],
        batch["PACKETS"] + batch["PACKETS_REV"],
        batch["BYTES"] + batch["BYTES_REV"],
        batch[APP_COLUMN],
        batch[CATEGORY_COLUMN],
    )

def pick_extra_fields(batch, packet_histograms: list[str], flow_endreason_features: list[str]):
    return (
        batch["DST_ASN"],
        batch[packet_histograms],
        batch[flow_endreason_features],
    )

def compute_dataset_statistics(database_path: str,
                               output_dir: str,
                               packet_histograms: list[str],
                               flowstats_features_boolean: list[str],
                               protocol: Protocol, extra_fields: bool,
                               disabled_apps: list[str],
                               num_samples: int | Literal["all"] = 10_000_000,
                               num_workers: int = 4,
                               batch_size: int = 4096,
                               silent: bool = False):
    stats_pdf_path = os.path.join(output_dir, "dataset-statistics.pdf")
    stats_csv_path = os.path.join(output_dir, "dataset-statistics.csv")
    categories_csv_path = os.path.join(output_dir, "categories.csv")
    categories_figure_path = os.path.join(output_dir, "categories-tikzpicture.txt")
    app_path = os.path.join(output_dir, "app.csv")
    asn_path = os.path.join(output_dir, "asn.csv")
    phist_path = os.path.join(output_dir, "phists.csv")
    packet_sizes_path = os.path.join(output_dir, "ppi-packet-sizes.csv")
    ipt_path = os.path.join(output_dir, "ppi-ipt.csv")
    flow_endreason_path = os.path.join(output_dir, "flow-endreason.csv")
    quic_sni_path = os.path.join(output_dir, "quic-sni.csv")
    quic_ua_path = os.path.join(output_dir, "quic-ua.csv")
    quic_version_path = os.path.join(output_dir, "quic-version.csv")

    df_categories = pd.DataFrame()
    df_phist = pd.DataFrame()
    app_series = pd.Series(dtype="int64")
    quic_sni_series = pd.Series(dtype="int64")
    quic_ua_series = pd.Series(dtype="int64")
    quic_version_series = pd.Series(dtype="int64")
    asn_series = pd.Series(dtype="int64")
    flow_endreason_series = pd.Series(dtype="int64")
    feature_duration = []
    feature_packets_total = []
    feature_bytes_total = []
    packet_sizes_counter = Counter()
    ipt_counter = Counter()
    flow_endreason_features = [f for f in flowstats_features_boolean if f.startswith("FLOW_ENDREASON")]
    if not silent:
        print(f"Reading data from {database_path} for statistics")
    table_paths = list_all_tables(database_path)
    stats_dataset = PyTablesDataset(database_path=database_path,
                                    tables_paths=table_paths,
                                    indices=None,
                                    disabled_apps=disabled_apps,
                                    return_all_fields=True,
                                    flowstats_features=[],
                                    flowstats_features_boolean=[],
                                    flowstats_features_phist=[],
                                    other_fields=[],
                                    ppi_channels=UDP_PPI_CHANNELS,)
    if num_samples != "all":
        subset_indices = np.random.randint(low=0, high=len(stats_dataset.indices), size=num_samples)
        stats_dataset.indices = sort_indices(stats_dataset.indices[subset_indices])
    stats_batch_sampler = BatchSampler(sampler=SequentialSampler(stats_dataset), batch_size=batch_size, drop_last=False)
    stats_dloader = DataLoader(
        stats_dataset,
        pin_memory=False,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        collate_fn=_collate_fn_simple,
        persistent_workers=False,
        batch_size=None,
        sampler=stats_batch_sampler)
    if num_workers == 0:
        stats_dataset.pytables_worker_init()

    for batch, batch_idx in tqdm(stats_dloader, total=len(stats_dloader), disable=silent):
        ppi, duration, packets_total, bytes_total, app, cat = pick_stats_fields(batch)
        # Saving feature values for distribution plots
        feature_duration.append(duration)
        feature_packets_total.append(packets_total)
        feature_bytes_total.append(bytes_total)
        packet_sizes_counter.update(ppi[:, SIZE_POS, :].flatten().astype(int))
        ipt_counter.update(ppi[:, IPT_POS, :].flatten().astype(int))
        # Aggregating features for value_counts
        app_series = app_series.add(pd.Series(app).value_counts(), fill_value=0)
        # Grouping features per categories
        df1 = pd.DataFrame(data={"cat": cat, "BYTES_TOTAL": bytes_total})
        flow_counts = df1["cat"].value_counts().rename("FLOW_COUNT")
        byte_volumes = df1.groupby("cat")["BYTES_TOTAL"].sum()
        df_categories = df_categories.add(pd.concat((flow_counts, byte_volumes), axis=1), fill_value=0)
        # QUIC features
        if protocol == Protocol.QUIC:
            sni, user_agent, quic_version = pick_quic_fields(batch)
            quic_sni_series = quic_sni_series.add(pd.Series(sni).str.decode("utf-8").value_counts(), fill_value=0)
            quic_ua_series = quic_ua_series.add(pd.Series(user_agent).str.decode("utf-8").value_counts(), fill_value=0)
            quic_version_series = quic_version_series.add(pd.Series(quic_version).value_counts(), fill_value=0)
        if extra_fields:
            asn, phist, flowend_reason = pick_extra_fields(batch, packet_histograms=packet_histograms, flow_endreason_features=flow_endreason_features)
            asn_series = asn_series.add(pd.Series(asn).value_counts(), fill_value=0)
            flow_endreason_series = flow_endreason_series.add(pd.Series(structured_to_unstructured(flowend_reason).sum(axis=0)), fill_value=0)
            df2 = pd.DataFrame(data=zip(*np.split(structured_to_unstructured(phist).sum(axis=0), 4)), columns=packet_histograms)
            df_phist = df_phist.add(df2, fill_value=0)
    feature_duration = np.concatenate(feature_duration)
    feature_packets_total = np.concatenate(feature_packets_total)
    feature_bytes_total = np.concatenate(feature_bytes_total)
    packet_sizes_counter.pop(0) # remove zero packets as those are just padding
    assert df_categories["BYTES_TOTAL"].sum() == feature_bytes_total.sum()
    assert df_categories["FLOW_COUNT"].sum() == app_series.sum()

    # Flow statistics distribution output
    df_flowstats = pd.DataFrame(data={"FLOW DURATION": feature_duration, "FLOW BYTE VOLUME": feature_bytes_total, "FLOW LENGTH": feature_packets_total}).describe()
    df_flowstats.to_csv(stats_csv_path)
    # Categories tikzpicture and csv output; first, get the categories and applications enum
    temp_database, temp_tables = load_database(database_path=database_path, tables_paths=table_paths[:1])
    cat_enum = temp_tables[0].get_enum(CATEGORY_COLUMN)
    app_enum = temp_tables[0].get_enum(APP_COLUMN)
    temp_database.close()
    df_categories.index = df_categories.index.map(cat_enum)
    df_categories = df_categories.drop("default", errors="ignore")
    df_categories["FLOW_PERC"] = df_categories["FLOW_COUNT"] / sum(df_categories["FLOW_COUNT"]) * 100
    df_categories["BYTES_PERC"] = df_categories["BYTES_TOTAL"] / sum(df_categories["BYTES_TOTAL"]) * 100
    df_categories = df_categories.round(3)
    df_categories = df_categories.sort_values("FLOW_PERC", ascending=False)
    with open(categories_figure_path, "w") as f:
        print(f"Categories:\n{df_categories.index.tolist()}", file=f)
        print(f"Flow Percentage:\n{list(df_categories['FLOW_PERC'].items())}", file=f)
        print(f"Byte Volume Percentage:\n{list(df_categories['BYTES_PERC'].items())}", file=f)
    df_categories.rename({"FLOW_COUNT": "FLOW COUNT", "BYTES_TOTAL": "BYTE VOLUME", "FLOW_PERC": "FLOW PERC", "BYTES_PERC": "BYTE VOLUME PERC"}, axis=1).to_csv(categories_csv_path, index_label="CATEGORIES")
    # Application distribution output
    app_df = pd.DataFrame({"COUNT": app_series.sort_values(ascending=False).astype("int64")})
    app_df["PERC"] = (app_df["COUNT"] / app_df["COUNT"].sum() * 100).round(3)
    app_df.index = app_df.index.map(app_enum)
    app_df.index.name = "LABEL"
    app_df.to_csv(app_path)
    # Packet sizes histogram output
    packet_sizes_df = pd.DataFrame({"COUNT": pd.Series(packet_sizes_counter)}).sort_index()
    packet_sizes_df["PERC"] = (packet_sizes_df["COUNT"] / packet_sizes_df["COUNT"].sum() * 100).round(3)
    packet_sizes_df.index.name = "PACKET SIZE"
    packet_sizes_df.to_csv(packet_sizes_path)
    # IPT histogram output
    ipt_df = pd.DataFrame({"COUNT": pd.Series(ipt_counter)}).sort_index()
    ipt_df["PERC"] = (ipt_df["COUNT"] / ipt_df["COUNT"].sum() * 100).round(3)
    ipt_df.index.name = "INTER-PACKET TIME"
    ipt_df.to_csv(ipt_path)
    # QUIC features output
    if protocol == Protocol.QUIC:
        quic_ua_series = quic_ua_series.rename(index={"": "No User Agent"})
        quic_ua_series.sort_values(ascending=False).astype("int64").rename("COUNT").to_csv(quic_ua_path, index_label="QUIC USER AGENT")
        quic_sni_series.sort_values(ascending=False).astype("int64").rename("COUNT").to_csv(quic_sni_path, index_label="QUIC SNI")
        quic_version_df = pd.DataFrame({"COUNT": quic_version_series.sort_values(ascending=False).astype("int64")})
        quic_version_df["PERC"] = (quic_version_df["COUNT"] / quic_version_df["COUNT"].sum() * 100).round(3)
        quic_version_df.index = quic_version_df.index.map(hex)
        quic_version_df.index.name = "QUIC VERSION"
        quic_version_df.to_csv(quic_version_path)
    if extra_fields:
        # ASN distribution output
        asn_df = pd.DataFrame({"COUNT": asn_series.sort_values(ascending=False).astype("int64")})
        asn_df["PERC"] = (asn_df["COUNT"] / asn_df["COUNT"].sum() * 100).round(3)
        asn_df.index.name = "DESTINATION ASN"
        asn_df.to_csv(asn_path)
        # Flow end reason output
        flow_endreason_df = pd.DataFrame({"COUNT": flow_endreason_series.astype("int64")})
        flow_endreason_df["PERC"] = (flow_endreason_df["COUNT"] / flow_endreason_df["COUNT"].sum() * 100).round(3)
        flow_endreason_df.index.name = "FLOW ENDREASON"
        flow_endreason_df.index = pd.Index(flow_endreason_features)
        flow_endreason_df.to_csv(flow_endreason_path)
        # PHIST output
        df_phist.index.name = "BINS"
        df_phist.columns = list(map(lambda x: x.upper().replace("_", " "), packet_histograms))
        df_phist = df_phist.astype("int64")
        for i, column in zip((1, 3, 5, 7), df_phist.columns):
            df_phist.insert(i, column + " PERC", (df_phist[column] / df_phist[column].sum() * 100).round(3))
        df_phist.to_csv(phist_path)

    # Dataset stats figure
    axes: Any
    fig, axes = plt.subplots(2, 2, figsize=(7, 7))
    ax1 = axes[0][0]
    ax2 = axes[0][1]
    ax3 = axes[1][0]
    ax4 = axes[1][1]
    x_log_text_margin = 1.3
    x_text_margin = 1.7
    y_prob_text_margin = 0.03
    x_psizes_margin = 0.5

    sns.ecdfplot(feature_duration, ax=ax1, log_scale=True)
    cdf = ax1.get_lines()[0].get_data()
    ax1.set_title("Flow Duration")
    ax1.set_xlabel("Seconds")
    prob_at_1s = cdf[1][np.searchsorted(cdf[0], 1, side="right")]
    ax1.axvline(1, color="grey", linestyle="--")
    ax1.text(1 * x_log_text_margin, y_prob_text_margin, f"{prob_at_1s*100:.1f}% < 1s", rotation=90)
    prob_at_30s = cdf[1][np.searchsorted(cdf[0], 30, side="right")]
    ax1.axvline(30, color="grey", linestyle="--")
    ax1.text(30 * x_log_text_margin, y_prob_text_margin, f"{prob_at_30s*100:.1f}% < 30s", rotation=90)

    sns.ecdfplot(feature_bytes_total, ax=ax2, log_scale=True)
    cdf = ax2.get_lines()[0].get_data()
    ax2.set_title("Flow Byte Volume")
    ax2.set_xlabel("Bytes")
    prob_at_10kb = cdf[1][np.searchsorted(cdf[0], 10_000, side="right")]
    ax2.axvline(10_000, color="grey", linestyle="--")
    ax2.text(10_000 * x_log_text_margin, y_prob_text_margin, f"{prob_at_10kb*100:.1f}% < 10KB", rotation=90)
    prob_at_1mb = cdf[1][np.searchsorted(cdf[0], 1_000_000, side="right")]
    ax2.axvline(1_000_000, color="grey", linestyle="--")
    ax2.text(1_000_000 * x_log_text_margin, y_prob_text_margin, f"{prob_at_1mb*100:.1f}% < 1MB", rotation=90)
    mb_at_999perc = cdf[0][np.searchsorted(cdf[1], 0.999, side="right")]
    ax2.axvline(mb_at_999perc, color="grey", linestyle="--")
    ax2.text(mb_at_999perc * x_log_text_margin, y_prob_text_margin, f"99.9% < {mb_at_999perc / 1_000_000:.1f}1MB", rotation=90)

    sns.ecdfplot(feature_packets_total, ax=ax3)
    cdf = ax3.get_lines()[0].get_data()
    ax3.set_title("Flow Length")
    ax3.set_xlabel("Packets")
    ax3.set_xlim(0, 110)
    prob_at_30packets = cdf[1][np.searchsorted(cdf[0], 30, side="left")]
    prob_at_100packets = cdf[1][np.searchsorted(cdf[0], 100, side="left")]
    ax3.axvline(30, color="grey", linestyle="--")
    ax3.text(30 + x_text_margin, y_prob_text_margin, f"{prob_at_30packets*100:.1f}% < 30 packets", rotation=90)
    ax3.axvline(100, color="grey", linestyle="--")
    ax3.text(100 + x_text_margin, y_prob_text_margin, f"{prob_at_100packets*100:.1f}% < 100 packets", rotation=90)

    packet_sizes_hist = list(zip(*sorted(dict(packet_sizes_counter).items(), key=lambda x: x[0])))
    sns.histplot(x=packet_sizes_hist[0], weights=packet_sizes_hist[1], ax=ax4, binwidth=50, binrange=(0, 1500), stat="proportion")
    ax4.set_title("Packet Sizes")
    ax4.set_xlabel("Bytes")

    plt.tight_layout()
    fig.savefig(stats_pdf_path, bbox_inches="tight")

def _collate_fn_simple(batch):
    return batch
