import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from joblib import load
import pandas as pd
from tqdm import tqdm
import os
import numpy as np
import argparse
import itertools
from functools import lru_cache
from model import SAGEDeterministic


parser = argparse.ArgumentParser(description="Alibaba traces")
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--log_steps", type=int, default=1)
parser.add_argument("--use_sage", action="store_true")
parser.add_argument("--num_layers", type=int, default=1)
parser.add_argument("--hidden_channels", type=int, default=32)
parser.add_argument("--dropout", type=float, default=0)
parser.add_argument("--lr", type=float, default=0.0003)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--runs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=170)
parser.add_argument("--graph_type", type=str, default="span", help="span or pert")
args = parser.parse_args()
print(args)

device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
device = torch.device(device)


tr2data = torch.load("processed/tr2data.pt")
# sample 100000 traces
tr2data = {tr: tr2data[tr] for tr in list(tr2data.keys())[:100000]}
if args.graph_type == "span":
    runtime2graph = torch.load("processed/runtime2spangraph_map.pt")
elif args.graph_type == "pert":
    runtime2graph = torch.load("processed/runtime2pertgraph_map.pt")
entry2runtimes = load("processed/entry2runtimes.joblib")
resource_df = pd.read_csv("processed/processed_resource_df.csv", engine="pyarrow")
resource_df["msname"] = resource_df["msname"].astype(int)
resource_df = resource_df.set_index(["timestamp", "msname"])

unique_ms_ids = np.unique(
    np.array(
        [
            ms_id.item()
            for runtime_id in runtime2graph.keys()
            for ms_id in runtime2graph[runtime_id]["ms_id"]
        ]
    ).ravel()
)
print("==>> num unique_ms_ids: ", len(unique_ms_ids))


@lru_cache(maxsize=None)
def get_x(timestamp, sorted_unique_ms, ms_with_resources):
    num_features = resource_df.shape[1]
    num_nodes = len(sorted_unique_ms)
    x = torch.hstack(
        (
            torch.tensor(
                [[0 for _ in range(num_features)] for _ in range(num_nodes)],
                dtype=torch.float,
            ),
            torch.tensor([[1] for _ in range(num_nodes)], dtype=torch.float),
        )  # missing indicator
    )
    related_ms = [ms for ms in sorted_unique_ms if ms in ms_with_resources]
    num_related_ms = len(related_ms)
    if num_related_ms > 0:
        ms2nid = {ms: nid for ms, nid in zip(sorted_unique_ms, range(num_nodes))}
        related_nids = [ms2nid[ms] for ms in related_ms]
        non_missing_x = torch.tensor(
            resource_df.loc[[(timestamp, ms) for ms in related_ms]].values,
            dtype=torch.float,
        )
        non_missing_indicator = torch.tensor(
            [[0] for _ in range(num_related_ms)], dtype=torch.float
        )
        non_missing_x = torch.hstack((non_missing_x, non_missing_indicator))
        x[related_nids, :] = non_missing_x
    return x


def get_all_runtimes_id_probs(entry_id, entry2runtimes):
    return (
        list(entry2runtimes[entry_id].keys()),
        list(entry2runtimes[entry_id].values()),
    )

@lru_cache(maxsize=None)
def get_edge_attr(rt_ids):
    return torch.cat(
        [
            runtime2graph[rt_id]["edge_attr"]
            for rt_id in rt_ids
        ],
        dim=0,
    )

@lru_cache(maxsize=None)
def get_edge_orderings(rt_ids):
    return torch.cat(
        [
            runtime2graph[rt_id]["edge_orderings"]
            for rt_id in rt_ids
        ],
        dim=0,
    )

@lru_cache(maxsize=None)
def get_pattern_num_nodes(rt_ids):
    return torch.tensor(
        [
            [runtime2graph[rt_id]["num_nodes"]]
            for rt_id in rt_ids
            for _ in range(runtime2graph[rt_id]["num_nodes"])
        ],
        dtype=torch.float,
    )

@lru_cache(maxsize=None)
def get_cat_X(rt_ids):
    return torch.cat([runtime2graph[rt_id]["ms_id"] for rt_id in rt_ids], dim=0)

@lru_cache(maxsize=None)
def get_node_depth(rt_ids):
    return torch.cat([runtime2graph[rt_id]["node_depth"] for rt_id in rt_ids], dim=0)

@lru_cache(maxsize=None)
def get_edge_index(rt_ids):
    cum_num_nodes = np.append(
        0, np.cumsum([runtime2graph[rt_id]["num_nodes"] for rt_id in rt_ids])[:-1],
    )
    return torch.cat(
        [
            runtime2graph[rt_id]["edge_index"] + cum_num_node
            for rt_id, cum_num_node in zip(rt_ids, cum_num_nodes)
        ],
        dim=1,
    )

@lru_cache(maxsize=None)
def transform_pattern_probs(rt_probs, rt_ids):
    return torch.tensor(
        [
            [rt_prob]
            for (rt_id, rt_prob) in zip(rt_ids, rt_probs)
            for _ in range(runtime2graph[rt_id]["num_nodes"])
        ]
        , dtype=torch.float)

def get_entry_data(entry_id, timestamp, entry2runtimes, runtime2graph, y):
    ms_with_resources = np.unique(resource_df.index.get_level_values("msname").values)
    rt_ids, rt_probs = get_all_runtimes_id_probs(entry_id, entry2runtimes)
    rt_probs = np.array(rt_probs)[np.newaxis].T
    X = torch.cat(
        [
            get_x(
                timestamp,
                tuple(
                    itertools.chain.from_iterable(
                        runtime2graph[rt_id]["ms_id"].tolist()
                    )
                ),
                tuple(ms_with_resources),
            )
            for rt_id in rt_ids
        ],
        dim=0,
    )
    rt_ids = tuple(rt_ids)
    cat_X = get_cat_X(rt_ids)
    node_depth = get_node_depth(rt_ids)

    edge_index = get_edge_index(rt_ids)

    edge_attr = get_edge_attr(rt_ids)

    pattern_num_nodes = get_pattern_num_nodes(rt_ids)

    return Data(
        x=X,
        edge_index=edge_index,
        edge_attr=edge_attr,
        cat_X=cat_X,
        node_depth=node_depth,
        pattern_num_nodes=pattern_num_nodes,
        pattern_probs=torch.tensor(rt_probs, dtype=torch.float),
        entry_id=torch.tensor([entry_id], dtype=torch.long),
        y=y,
    )


def get_data_list(tr2data, entry2runtimes, runtime2graph):
    # Return the data list for one epoch of training
    data_list = []

    for data in tqdm(tr2data.values()):
        entry_id = data["entry_id"]
        timestamp = data["timestamp"]
        g = get_entry_data(
            entry_id, timestamp, entry2runtimes, runtime2graph, data["y"]
        )
        data_list.append(g)

    return data_list


def torch_quantile_loss(y_test, y_hat, tau):
    e = y_test - y_hat
    return torch.mean(torch.maximum(tau * e, (tau - 1) * e))


def get_data_loader(data_list):
    # Return the data loader for one epoch of training
    train_data_list = data_list[: int(len(data_list) * 0.6)]
    valid_data_list = data_list[int(len(data_list) * 0.6) : int(len(data_list) * 0.8)]
    test_data_list = data_list[int(len(data_list) * 0.8) :]
    train_data_loader = DataLoader(
        train_data_list, batch_size=args.batch_size, shuffle=True
    )
    valid_data_loader = DataLoader(valid_data_list, batch_size=args.batch_size, shuffle=False)
    test_data_loader = DataLoader(
        test_data_list, batch_size=args.batch_size, shuffle=False
    )
    return train_data_loader, valid_data_loader, test_data_loader


def train(train_loader):
    model.train()

    total_loss = 0
    mape = 0
    # Create train loader
    for data in (pbar := tqdm(train_loader)) :
        rt_ids_rt_probs = [get_all_runtimes_id_probs(entry_id.item(), entry2runtimes) for entry_id in data.entry_id]
        rt_probs = torch.cat([transform_pattern_probs(tuple(rt_probs), tuple(rt_ids)).to(device) for (rt_ids, rt_probs) in rt_ids_rt_probs], dim=0)
        data = data.to(device)
        optimizer.zero_grad()
        global_pred, local_pred = model(
            data.x,
            data.cat_X,
            data.edge_index,
            data.edge_attr,
            data.edge_orderings,
            data.pattern_num_nodes,
            # data.pattern_probs,
            rt_probs,
            data.entry_id,
            data.batch,
        )
        # quantile loss
        loss = torch_quantile_loss(data.y.float(), global_pred.flatten(), 0.95)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
        mape += ((global_pred.flatten() - data.y).abs() / data.y).sum()
        pbar.set_description(f"Batch loss: {loss}")
    return total_loss / len(train_loader.dataset), mape / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()

    # mse = 0
    mae = 0
    mape = 0
    q95loss = 0
    for data in loader:
        rt_ids_rt_probs = [get_all_runtimes_id_probs(entry_id.item(), entry2runtimes) for entry_id in data.entry_id]
        rt_probs = torch.cat([transform_pattern_probs(tuple(rt_probs), tuple(rt_ids)).to(device) for (rt_ids, rt_probs) in rt_ids_rt_probs], dim=0)
        data = data.to(device)
        global_pred, local_pred = model(
            data.x,
            data.cat_X,
            data.edge_index,
            data.edge_attr,
            data.edge_orderings,
            data.pattern_num_nodes,
            # data.pattern_probs,
            rt_probs,
            data.entry_id,
            data.batch,
        )
        mae += (global_pred.flatten() - data.y).abs().sum()
        mape += ((global_pred.flatten() - data.y).abs() / data.y).sum()
        q95loss += torch_quantile_loss(data.y.float(), global_pred.flatten(), 0.95) * data.y.shape[0]
    return mae / len(loader.dataset), mape / len(loader.dataset), q95loss / len(loader.dataset)


if os.path.exists("processed/full_span_data_list.pt"):
    data_list = torch.load("processed/full_span_data_list.pt")
else:
    print("Getting data list...")
    data_list = get_data_list(tr2data, entry2runtimes, runtime2graph)
    torch.save(data_list, "processed/full_span_data_list.pt")
train_loader, valid_loader, test_loader = get_data_loader(data_list)
num_features = resource_df.shape[1]
entry_ids = [data.entry_id.item() for data in data_list]
entry_id_max = max(entry_ids)
interface_id_max = max([data.edge_attr[:, 0].max().item() for data in data_list])
rpctype_id_max = max([data.edge_attr[:, 1].max().item() for data in data_list])

# num_features + 1 because of the missing indicator


model = SAGEDeterministic(
    num_features + 1,
    [unique_ms_ids.max() + 1],
    entry_id_max,
    interface_id_max, 
    rpctype_id_max,
    args.hidden_channels,
    args.num_layers,
    args.dropout,
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
for epoch in (pbar := tqdm(range(1, args.epochs + 1))) :
    train_mae, train_mape = train(train_loader)
    valid_mae, valid_mape, valid_q95_loss = test(valid_loader)
    test_mae, test_mape, test_q95_loss = test(test_loader)
    print(
        f"Epoch: {epoch}, Train: {train_mae}, Test mae: {test_mae}, Train mape: {train_mape}, Test mape: {test_mape}, Test q95 loss: {test_q95_loss}"
    )

