import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from joblib import dump, load
import pandas as pd
from tqdm import tqdm
import os
import numpy as np
from misc import get_file
import sys


print("Loading tr2data_map")
# {trace_id: {cluster_id: <id>, timestamp: <timestamp>}}
tr2data_map = load("processed/tr2data.joblib")

print("Loading cluster2graph")
# {cluster_id: {"edge_index": edge_index, "edge_attr": edge_attr}
cluster2graph = torch.load("processed/cluster2graph.pt")


print("Loading df")
if os.path.isfile("processed_df.csv"):
    print("File: processed_df.csv found!!")
    df = pd.read_csv("processed/processed_df.csv", engine='pyarrow')
else:
    print("File: processed_df.csv not found!! Run preprocess.py first!!!")
    sys.exit()


print("Loading Resource df")
if os.path.isfile("processed_resource_df.csv"):
    print("File: processed_resource_df.csv found!!")
    resource_df = pd.read_csv("processed/processed_resource_df.csv", engine='pyarrow')
else:
    print("File: processed_resource_df.csv not found!! Run preprocess.py first!!!")
    sys.exit()

tr2delay = get_file("processed/tr2delay.joblib")
if tr2delay is None:
    # tr2delay = {}
    # for traceid, group in tqdm(df.groupby("trace_id")):
    #     entry_candidate = group[(group.rpctype=="http") & (group.timestamp==group.timestamp.min()) & (group.rt.abs()==group.rt.abs().max())]
    #     if entry_candidate.shape[0] == 1:
    #         tr2delay[traceid] = entry_candidate.rt.abs().iloc[0]
    #     elif entry_candidate.shape[0] > 1:  # if there are multiple candidate, we further filter out the one where um == '(?)'
    #         entry_candidate = entry_candidate[entry_candidate.um == 0]
    #         if entry_candidate.shape[0] == 1:
    #             tr2delay[traceid] = entry_candidate.rt.abs().iloc[0]
    tr2delay = df.groupby(['traceid'])['rt'].apply(lambda x: x.abs().max()).to_dict()
    dump(tr2delay, "processed/tr2delay.joblib")
    

resource_df['msname'] = resource_df['msname'].astype(int)
ms_with_resources = resource_df['msname'].unique()
resource_df = resource_df.set_index(["timestamp", "msname"])
num_features = resource_df.shape[1]
# ms_with_resources = np.array(resource_df.loc[0].index).astype(
    # int
# )  # indexing the first timestamp and get all msnames

data_list = []

for trace_id, trace_data in tqdm(tr2data_map.items()):
    cluster_id = trace_data["cluster_id"]
    timestamp = trace_data["timestamp"]
    edge_index = cluster2graph[cluster_id]["edge_index"]
    edge_attr = cluster2graph[cluster_id]["edge_attr"]
    sorted_unique_ms, edge_index = torch.unique(
        edge_index, sorted=True, return_inverse=True
    )  # map to consecutive integer starting from zero
    sorted_unique_ms = sorted_unique_ms.tolist()
    num_nodes = len(sorted_unique_ms)
    # edge_index is now start from zero

    # create the features
    # For nodes with missing values, the featuers will be [0,0,0,...,1]
    # For nodes with features, it will be [cpu_mean, ram_mean, cpu_max, ram_max, ..., 0]
    x = torch.tensor(
        [[0 for _ in range(num_features + 1)] for _ in range(num_nodes)],
        dtype=torch.float,
    )

    related_ms = [ms for ms in sorted_unique_ms if ms in ms_with_resources]
    # print("==>> ms_with_resources: ", ms_with_resources)
    # print("==>> sorted_unique_ms: ", sorted_unique_ms)
    # print("==>> related_ms: ", related_ms)
    num_related_ms = len(related_ms)
    if num_related_ms > 0:
        ms2nid = {ms: nid for ms, nid in zip(sorted_unique_ms, range(num_nodes))}
        related_nids = [ms2nid[ms] for ms in related_ms]
        non_missing_x = torch.tensor(
            resource_df.loc[[(timestamp, ms) for ms in related_ms]].values, dtype=torch.float
        )
        non_missing_indicator = torch.tensor([[0] for _ in range(num_related_ms)], dtype=torch.float)
        non_missing_x = torch.hstack((non_missing_x, non_missing_indicator))
        x[related_nids, :] = non_missing_x
        # print("==>> num_features: ", num_features)
        # print("==>> x.shape: ", x.shape)
        # print("==>> non_missing_x.shape: ", non_missing_x.shape)
        # print("==>> non_missing_x: ", non_missing_x)
        # print("==>> x.shape: ", x.shape)
        # print("==>> x: ", x)
    else:
        print("All nodes are having missing node features. Please rerun preprocess.py!")
        sys.exit()

    edge_attr = cluster2graph[cluster_id]["edge_attr"]
    y = torch.tensor(tr2delay[trace_id])
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data_list.append(data)

torch.save(data_list, "processed/data_list.pt")
