import pandas as pd
import networkx as nx
import numpy as np
import os
import os.path as osp
from sklearn.manifold import TSNE
from tqdm import tqdm
import matplotlib.pyplot as plt
from joblib import dump, load
import torch
from torch_geometric.data import InMemoryDataset, download_url
import torch
import json
import sys
import math
from misc import get_corpus_data, GraphConstruct, find_most_recent_fts
from collections import defaultdict

# Output
# 1. tr2data_map (for training)
# {trace_id: {entry_id: <id>, run_time_id: <id>, timestamp: <timestamp>, x: <x>, y: <y>}
# 2. entry2runtimes_map (for test)
# entry2runtimes_map = {
#   <entry_id>: {
#     ”<run_time_ids>”: <occurences>,
#   }
# }
# 3. runtime2graph (for test)
# {run_time_id: edge_index: <edge_index>, edge_attr: <edge_attr>}}


def get_file(filename):
    print(f"Getting {filename}...")
    if os.path.isfile(filename):
        print(f"File: {filename} found!")
        return load(filename)
    else:
        return None


def get_tr2ts_map(df):
    print("Getting tr2ts_map.joblib...")
    tr2ts_map = get_file("processed/tr2ts_map.joblib")
    if tr2ts_map is not None:
        return tr2ts_map
    print("tr2ts_map.joblib not found, generating...")
    tr2ts_map = df.groupby("traceid").timestamp.min() // 30000 * 30000
    dump(tr2ts_map, "processed/tr2ts_map.joblib")
    return tr2ts_map


def get_tr2tsrange_map(df):
    # get (timestamp.max() - timestamp.min()) for each trace
    # return {trace_id: tsrange}
    # return df.groupby("traceid").agg({"timestamp": np.ptp}).to_dict()["timestamp"]
    return df.groupby("traceid").agg({"timestamp": np.ptp}).to_dict()["timestamp"]


def update_tr2data_map(
    traceids,
    kmeans_labels,
    tr2ts_map,
    tr2data_map,
    # method="graph_union",
    # graph_type="dag",
):
    # {trace_id: {cluster_id: <id>, timestamp: <timestamp>}}
    # cluster_id is used with the cluster2graph_map to get the graph (edge_index)
    # timestamp is used with the resource table to get the cpu/ram
    for trace_id, kmeans_label in zip(traceids, kmeans_labels):
        tr2data_map[trace_id] = {
            "cluster_id": kmeans_label,
            "timestamp": tr2ts_map[trace_id],
        }
    return tr2data_map


def combine_tr2data_with_delay(tr2data_map, tr2delay_map, tr2tsrange_map):
    for trace_id, delay in tr2delay_map.items():
        tr2data_map[trace_id]["delay"] = delay

    for trace_id, tsrange in tr2tsrange_map.items():
        tr2data_map[trace_id]["tsrange"] = tsrange

    return tr2data_map


def graph_dtw():
    return




def map_consecutive_ids(df, cols):
    # map the original ids to consecutive ids
    # return {original_id: consecutive_id}
    print(f"Mapping {cols} to consecutive ids...")
    # unique = np.unique(np.concatenate([np.unique(df[col].values.astype(str)) for col in cols]))
    # mappings = dict(zip(unique, range(len(unique))))
    # json.dump(mappings, open(f"processed/{'_'.join(cols)}_mappings.json", "w"))
    # for col in cols:
    # df[col] = df[col].map(mappings)
    if cols is not None:
        stacked = df[cols].stack()
        codes, uniques = stacked.factorize()
        df.loc[:, cols] = pd.Series(codes, index=stacked.index).unstack()
    else:
        codes, uniques = df.factorize()
        df[:] = codes
    return df, uniques


def filter_traces_with_missing_entry_and_get_delay(df):
    # filter out traces with missing entry
    print("Filtering traces with missing entry and get delay...")
    print(f"Total number of traces (Without filtering): {len(set(df.traceid.values))}")
    # Filter out all trace without the entry
    # related_traceids = df[df.rpcid == "0"].traceid.unique()  # How to identify the entry?

    related_traceids = []
    tr2entry = dict()
    num_traces_without_entry_candidate = 0
    num_traces_with_more_than_one_entry_candidate = 0
    for traceid, group in tqdm(df.groupby("traceid")):
        entry_candidate = group[
            (group.rpctype == "http")
            & (group.timestamp == group.timestamp.min())
            & (group.rt.abs() == group.rt.abs().max())
        ]
        entry = None

        if entry_candidate.shape[0] == 1:
            entry = entry_candidate.iloc[0]
        elif entry_candidate.shape[0] > 1:
            entry_candidate = entry_candidate[entry_candidate.um == "(?)"]
            if entry_candidate.shape[0] == 1:
                entry = entry_candidate.iloc[0]
            else:
                print(
                    f"Trace {traceid} has more than one entry candidate: {entry_candidate}"
                )
                num_traces_with_more_than_one_entry_candidate += 1
                continue
        else:
            num_traces_without_entry_candidate += 1
            continue

        # if there is only one entry candidate:
        entry_str = entry.dm + "_" + str(entry.interface)
        related_traceids.append(traceid)
        tr2entry[traceid] = entry_str
    df["entryid"] = df["traceid"].map(tr2entry)

    df = df[df.traceid.isin(related_traceids)]
    print(
        "==>> num_traces_without_entry_candidate: ", num_traces_without_entry_candidate
    )
    print(
        "==>> num_traces_with_more_than_one_entry_candidate: ",
        num_traces_with_more_than_one_entry_candidate,
    )
    print(f"Total number of traces (After filtering): {len(set(df.traceid.values))}")
    return df


# group.groupby(['traceid'])['rt'].apply(lambda x: x.abs().max())


def filter_traces_with_missing_features(df, resource_df):
    ms_with_resources = set(resource_df.msname.values)
    # Filter out all traces that have too many missing features
    # Get the list of traceids that have missing features
    print("Filtering traces with missing features...")
    num_traces_without_filtering = len(set(df.traceid.values))
    print(f"Total number of traces (Without filtering): {num_traces_without_filtering}")

    def intersect_percent(x):
        ms_in_traces = set(x.values.ravel())
        return len(ms_in_traces.intersection(ms_with_resources)) / len(ms_in_traces)

    intersect_percent_df = df.groupby("traceid")[["um", "dm"]].apply(
        lambda x: intersect_percent(x)
    )
    filtered_traceids = intersect_percent_df[intersect_percent_df >= 0.6].index

    print()
    df = df[df.traceid.isin(filtered_traceids)]
    print(
        f"Total number of traces where >60% MS have node features: {len(set(df.traceid.values))}"
    )
    return df


def filter_traces_by_occurence_per_entry(df, min_occurence=100):
    # Filter out all traces where the entry occurs less than min_occurence times
    print("Filtering traces by occurence per entry...")
    print(f"Total number of traces (Without filtering): {len(set(df.traceid.values))}")
    entry_occurence = df.groupby(["entryid"]).agg({"traceid": "nunique"})
    entry_occurence = entry_occurence[entry_occurence.traceid > min_occurence]
    df = df[df.entryid.isin(entry_occurence.index)]
    print(f"Total number of traces (After filtering): {len(set(df.traceid.values))}")
    return df


def get_df():
    if os.path.isfile("processed/processed_df.csv") and os.path.isfile(
        "processed/processed_resource_df.csv"
    ):
        # if None:
        print("File: preprocessed_df.csv and processed_resource_df found!")
        df = pd.read_csv("processed/processed_df.csv", engine="pyarrow")
        resource_df = pd.read_csv(
            "processed/processed_resource_df.csv", engine="pyarrow"
        )
    else:
        print("Getting call_graph df...")
        df = pd.concat(
            (
                pd.read_csv(f"./data/MSCallGraph/{f}", index_col=0, engine="pyarrow")
                # .drop("Unamed: 0", axis=1)
                .replace(np.nan, "nan")
                for f in tqdm(os.listdir("./data/MSCallGraph"))
                if f.endswith(".csv")
            ),
            ignore_index=True,
        ).drop_duplicates()
        # .drop(columns=["Unnamed: 0"])
        df = df.sort_values(by=["timestamp"])
        print(f"{df.shape}")

        df, _ = map_consecutive_ids(df, ["traceid"])
        df, _ = map_consecutive_ids(df, ["interface"])
        df = filter_traces_with_missing_entry_and_get_delay(df)
        df, _ = map_consecutive_ids(df, ["entryid"])
        df, _ = map_consecutive_ids(df, ["rpcid"])
        df, _ = map_consecutive_ids(df, ["rpctype"])

        df.to_csv("processed/processed_df.csv", index=False)
        ##############################################
        ##### Get resource df ########################
        ##############################################
        print("Getting resource df...")
        resource_df = pd.concat(
            pd.read_csv(f"./data/MSResource/{f}", engine="pyarrow")
            # .drop("Unnamed: 0", axis=1)
            .loc[
                :,
                ["timestamp", "msname", "instance_cpu_usage", "instance_memory_usage"],
            ]
            for f in tqdm(os.listdir("./data/MSResource"))
            if f.endswith(".csv")
        )
        resource_df = resource_df.groupby(["timestamp", "msname"]).agg(
            ["max", "min", "mean", "median"]
        )
        resource_df.columns = ["_".join(c) for c in resource_df.columns]

        resource_df = resource_df.reset_index()
        resource_df.to_csv("processed/processed_resource_df.csv", index=False)

        ##############################################
        df = filter_traces_with_missing_features(df, resource_df)
        df = filter_traces_by_occurence_per_entry(df, min_occurence=100)

        unique_ms = list(
            set(df.um.values).union(set(df.dm.values)).union(resource_df.msname.values)
        )
        ms2int = dict(zip(unique_ms, range(len(unique_ms))))
        df["um"] = df.um.map(ms2int)
        df["dm"] = df.dm.map(ms2int)
        resource_df["msname"] = resource_df.msname.map(ms2int)

        # resource_df["msname"] = resource_df["msname"].map(ms2int).astype(int)
        # ms_with_resources = set(resource_df.msname.astype(int).unique())
        df.to_csv("processed/processed_df.csv", index=False)
        resource_df.to_csv("processed/processed_resource_df.csv", index=False)

    df["endTimestamp"] = df["timestamp"] + df["rt"].abs()
    resource_df["msname"] = resource_df["msname"].astype(int)

    return df, resource_df


def main():
    print("Loading Data")
    # groupby_entry = get_file("groupby_entry.joblib")
    df, resource_df = get_df()
    resource_df = resource_df.set_index(["timestamp", "msname"])
    ms_with_resources = resource_df.index.get_level_values("msname").unique()
    # resource_df = resource_df.set_index("timestamp")
    tr2ts_map = get_tr2ts_map(df)
    tr2data_map = dict()
    entry2runtimes_map = dict()
    runtime2spangraph_map = dict()
    runtime2pertgraph_map = dict()
    # pd.merge_asof(df, node, by="nodeid", left_on="timestamp", right_on="timestamp")
    df["um_dm_interface"] = (
        df["um"].astype(str)
        + "_"
        + df["dm"].astype(str)
        + "_"
        + df["interface"].astype(str)
    )
    corpus_df = df.groupby("traceid").apply(
        lambda x: " ".join(x["um_dm_interface"].values)
    )
    tr2delay = df.groupby(["traceid"])["rt"].apply(lambda x: x.abs().max()).to_dict()
    traceid2runtime_id, _ = map_consecutive_ids(corpus_df, None)

    for entry, entry_group in tqdm(df.groupby(["entryid"])):
        for traceid, trace_df in entry_group.groupby(["traceid"]):
            runtime_id = traceid2runtime_id[traceid]
            trace_timestamp = tr2ts_map[traceid]
            trace_resource_df = find_most_recent_fts(resource_df, trace_timestamp)

            tr2data_map[traceid] = {
                "entry_id": entry,
                "runtime_id": runtime_id,
                "timestamp": trace_timestamp,
                "y": torch.tensor(tr2delay[traceid]),
            }
            if entry not in entry2runtimes_map:
                entry2runtimes_map[entry] = {runtime_id: 1}
            else:
                if runtime_id not in entry2runtimes_map[entry]:
                    entry2runtimes_map[entry][runtime_id] = 1
                else:
                    entry2runtimes_map[entry][runtime_id] += 1
            if runtime_id not in runtime2spangraph_map:
                g = GraphConstruct(trace_df, trace_resource_df, ms_with_resources)

                (
                    span_edge_index,
                    span_X,
                    span_node_depth,
                    span_edge_attr, 
                    span_edge_orderings, 
                    span_edge_durations, 
                    sorted_unique_ms
                ) = g.get_span_edge_index()

                ms_id = torch.tensor(
                    [[ms] for ms in sorted_unique_ms], dtype=torch.long
                )
                num_nodes = span_edge_index.max().item() + 1
                runtime2spangraph_map[runtime_id] = {
                    "edge_index": span_edge_index,
                    "ms_id": ms_id,
                    "occurences": 1,
                    "num_nodes": num_nodes,
                    "node_depth": span_node_depth,
                    "edge_attr": span_edge_attr,
                    "edge_orderings": span_edge_orderings
                }
            else:
                runtime2spangraph_map[runtime_id]["occurences"] += 1
            if runtime_id not in runtime2pertgraph_map:
                g = GraphConstruct(trace_df, trace_resource_df, ms_with_resources)

                (
                    pert_edge_index,
                    pert_edge_attr,
                    pert_X,
                    pert_node_depth,
                    sorted_unique_ms
                ) = g.get_pert_edge_index()

                ms_id = torch.tensor(
                    [[ms] for ms in sorted_unique_ms], dtype=torch.long
                )
                num_nodes = pert_edge_index.max().item() + 1
                runtime2pertgraph_map[runtime_id] = {
                    "edge_index": pert_edge_index,
                    "ms_id": ms_id,
                    "occurences": 1,
                    "num_nodes": num_nodes,
                    "node_depth": pert_node_depth,
                    "edge_attr": pert_edge_attr,
                }
            else:
                runtime2pertgraph_map[runtime_id]["occurences"] += 1

        print(f"Num traces: {len(entry_group.traceid.unique())}")

    # Normalize occurences
    for entry, entry_data in entry2runtimes_map.items():
        total = sum(entry_data.values())
        for runtime_id, occurences in entry_data.items():
            entry_data[runtime_id] = occurences / total

    # Save the tr2data_map, entry2runtimes_map and entry2runtimes_map to disk
    torch.save(runtime2spangraph_map, "processed/runtime2spangraph_map.pt")
    torch.save(runtime2pertgraph_map, "processed/runtime2pertgraph_map.pt")
    torch.save(tr2data_map, "processed/tr2data.pt")
    dump(entry2runtimes_map, "processed/entry2runtimes.joblib")


if __name__ == "__main__":
    main()

    # edge_index = torch.tensor(
    #     weighted_edge_list_df.loc[:, ["um", "dm"]].values.T, dtype=torch.long
    # ).contiguous()

