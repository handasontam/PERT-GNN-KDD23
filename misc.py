import os
from joblib import load
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    HashingVectorizer,
    TfidfTransformer,
)
from sklearn.pipeline import make_pipeline
from kneed import KneeLocator
from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np
import torch
import sys
from collections import defaultdict


def get_file(filename):
    print(f"Getting {filename}...")
    if os.path.isfile(filename):
        print(f"File: {filename} found!")
        return load(filename)
    else:
        return None


###################################
# Trace processing #
###################################


def get_corpus_data(entry_df):
    corpus = []
    sorted_traceids = []
    # for traceid, group in df[df.traceid.isin(traceids)].groupby("traceid"):
    for traceid, group in entry_df.groupby("traceid"):
        # Get the list of ms sorted in timestamp
        span_sequence = group["um_dm_interface"].values

        # Transform the list of ms into document text (a string)
        trace_text = " ".join(span_sequence)
        corpus.append(trace_text)
        sorted_traceids.append(traceid)
    return corpus, sorted_traceids


def get_corpus_from_trace(trace_df):
    corpus = []


def tfidf_microservices(ms_corpus):
    # ms_corpus is a list of string, each element (string) contains the microservices id seperated by space
    # return: a document-term matrix, shape = (number of string in the corpus, number of unique microservices)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X = vectorizer.fit_transform(ms_corpus)
    # dump(X, "doc_term_mat.joblib")
    return X


def hash_vec_microservices(ms_corpus):
    # ms_corpus is a list of string, each element (string) contains the microservices id seperated by space
    # return: a document-term matrix
    hasher = HashingVectorizer(
        ngram_range=(1, 2), n_features=1000, alternate_sign=False
    )
    vectorizer = make_pipeline(hasher, TfidfTransformer)
    # dump(vectorizer, "hash_vectorizer.joblib")
    X = vectorizer.fit_transform(ms_corpus)
    return X


# TODO: Use spark kmeans to cluster all MSs


def find_optimal_clusters(data, max_k):

    # f, ax = plt.subplots(1, 1)
    # ax.plot(iters, sse, marker="o")
    # ax.set_xlabel("Cluster Centers")
    # ax.set_xticks(iters)
    # ax.set_xticklabels(iters)
    # ax.set_ylabel("SSE")
    # ax.set_title("SSE by Cluster Center Plot")
    # plt.savefig("./SSE_Kmeans_plot.png")
    iters = range(2, min(data.shape[0], max_k + 1), 2)

    k_list = []
    sse = []
    for k in iters:
        k_list.append(k)
        # inertia = (
        #     KMeans(n_clusters=k, init="k-means++", random_state=20, verbose=0)
        #     .fit(data)
        #     .inertia_
        # )

        inertia = (
            MiniBatchKMeans(
                n_clusters=k,
                init="k-means++",
                batch_size=256 * 64 + 1,
                random_state=20,
                compute_labels=False,  # to use approximation of the inertia based on an exponentially weighted average of the batch inertiae
                verbose=0,
            )
            .fit(data)
            .inertia_
        )
        sse.append(inertia)
        if inertia == 0:
            break
        # print("Fit {} clusters".format(k))

    if len(k_list) > 1:
        kneedle = KneeLocator(
            x=k_list,
            y=sse,
            curve="convex",
            direction="decreasing",
            interp_method="polynomial",
        )
        knee = kneedle.knee
        if knee is None:
            kneedle = KneeLocator(
                x=k_list,
                y=sse,
                curve="convex",
                direction="decreasing",
                interp_method="interp1d",
            )
            knee = kneedle.knee
        if knee is None:
            knee = 1

    else:
        knee = 1

    print(f"sse={sse}")
    print(f"knee={knee}")
    # plt.plot(kneedle.x_difference, kneedle.y_difference)

    kmeans = KMeans(n_clusters=knee, init="k-means++", random_state=20).fit(data)
    return kmeans


###################################
# Trace clustering #
###################################


def get_pert_from_trace_cluster(entry_df, merge_method="graph_union"):
    # TODO
    pass


def update_cluster2graph_map(
    kmeans_labels, trace_ids, graph_type, merge_method, entry_group, cluster2graph_map
):
    # {cluster_id: {"edge_index": edge_index, "edge_attr": edge_attr}
    if graph_type == "dag":
        get_g_from_cluster = get_dag_prototype_from_trace_cluster
    elif graph_type == "pert":
        get_g_from_cluster = get_pert_from_trace_cluster
        print("PERT not yet implemented!")
        sys.exit()

    else:
        print(f"Graph type {graph_type} is not supported.")

    for label in np.unique(kmeans_labels):
        clustered_trace_ids = [
            trace_id
            for trace_id, kmeans_label in zip(trace_ids, kmeans_labels)
            if kmeans_label == label
        ]
        clusterd_trace_df = entry_group[entry_group.traceid.isin(clustered_trace_ids)]
        prototype_edge_index, prototype_edge_attr = get_g_from_cluster(
            clusterd_trace_df, merge_method=merge_method
        )
        cluster2graph_map[label] = {
            "edge_index": prototype_edge_index,
            "edge_attr": prototype_edge_attr,
        }

    return cluster2graph_map


def get_dag_prototype_from_trace_cluster(trace_cluster_df, merge_method="graph_union"):
    if merge_method == "graph_union":
        weighted_edge_list_df = (
            trace_cluster_df[["um", "dm"]].value_counts().reset_index()
        )

        # weighted_edge_list_df = trace_cluster_df.groupby(["um", "dm"]).count()
        # weighted_edge_list_df = weighted_edge_list_df.reset_index()
        edge_index = torch.tensor(
            weighted_edge_list_df.loc[:, ["um", "dm"]].values.T, dtype=torch.long
        ).contiguous()
        edge_attr = torch.tensor(
            weighted_edge_list_df.loc[:, 0].values, dtype=torch.float
        )

    elif merge_method == "graph_dtw":
        print(f"Method graph_dtw is not supported.")
        sys.exit()
    else:
        print(f"Method {merge_method} is not supported.")
        sys.exit()

    return edge_index, edge_attr


def update_max_kmeans_label(kmeans_labels, max_kmeans_label):
    return max(kmeans_labels) + max_kmeans_label + 1


class DFS:
    def __init__(self, num_nodes, adj_list):
        self.adj_list = adj_list
        # self.node_depth = [0] * num_nodes
        self.max_node_depth = [-1] * num_nodes
        self.min_node_depth = [float("infinity")] * num_nodes

    def dfs_min_node_depth(self, v, depth):
        if self.min_node_depth[v] > depth:
            self.min_node_depth[v] = depth
            for neighbor in self.adj_list[v]:
                self.dfs_min_node_depth(neighbor, depth + 1)

    def dfs_max_node_depth(self, v, depth):
        if self.max_node_depth[v] < depth:
            self.max_node_depth[v] = depth
            for neighbor in self.adj_list[v]:
                self.dfs_max_node_depth(neighbor, depth + 1)


class GraphConstruct:
    def __init__(self, trace_span_df, resource_df, ms_with_resources):
        self.root_span = self.get_root_spanID(trace_span_df)
        self.trace_span_df_no_duplicates = self.drop_wrong_edges(trace_span_df)
        # print(
        #     "==>> self.trace_span_df_no_duplicates.loc ",
        #     self.trace_span_df_no_duplicates.loc[
        #         :, ["um", "dm", "interface", "rpctype"]
        #     ],
        # )
        self.resource_df = resource_df
        # print("==>> self.resource_df: ", self.resource_df)
        self.ms_with_resource = ms_with_resources
        self.n_features = self.resource_df.shape[1]

    def drop_wrong_edges(self, trace_span_df):

        # remove self-loop
        trace_span_df = trace_span_df[trace_span_df.um != trace_span_df.dm]

        # Remove duplicates
        trace_span_df = trace_span_df.drop_duplicates(subset="rpcid", keep="first")

        # remove the last call, which causes a cycle
        trace_span_df = trace_span_df[(trace_span_df.dm != self.root_span)]
        # TODO: nodes should be ms+interface instead of just ms
        trace_span_df = trace_span_df.drop_duplicates(subset=["um", "dm"], keep="last")

        # TODO: Remove cycles in another way
        trace_span_df["ordered-umdm"] = trace_span_df.apply(
            lambda x: "-".join(sorted([str(x["um"]), str(x["dm"])])), axis=1
        )
        trace_span_df = trace_span_df.drop_duplicates(["ordered-umdm"])
        trace_span_df.drop(["ordered-umdm"], axis=1, inplace=True)
        return trace_span_df

    def build_adj_list(self, edge_index):
        adj_list = defaultdict(list)  # to store graph
        for edge in edge_index.T.detach().cpu().numpy():
            adj_list[edge[0]].append(edge[1])
        return adj_list

    def get_node_depth(self, root, num_nodes, adj_list):
        dfs = DFS(num_nodes, adj_list)

        depth = 0

        dfs.dfs_min_node_depth(root, depth)
        # TODO: Since it causes cycle
        # try:
        #     dfs.dfs_max_node_depth(root, depth)
        # except RecursionError:
        #     print("Recursion Error:")
        #     print("==>> adj_list: ", adj_list)
        #     print(
        #         "==>> self.trace_span_df_no_duplicates.loc ",
        #         self.trace_span_df_no_duplicates.loc[
        #             :, ["um", "dm", "interface", "rpctype", "rpcid", "timestamp"]
        #         ],
        #     )
        #     print("==>> self.root_span: ", self.root_span)
        #     import sys

        #     sys.exit()

        return dfs.min_node_depth

    def get_root_spanID(self, trace_span_df):
        return trace_span_df[
            (trace_span_df.rt.abs() == trace_span_df.rt.abs().max())
            & (trace_span_df.timestamp == trace_span_df.timestamp.min())
        ].um.iloc[0]

    def get_node_features(self, sorted_ms_id, edge_index, root_nid, num_nodes):
        N = len(sorted_ms_id)
        X = torch.tensor(np.zeros((N, self.n_features + 1)), dtype=torch.float)
        ms_with_features = sorted_ms_id[np.isin(sorted_ms_id, self.ms_with_resource)]
        ms2nid = {ms: nid for ms, nid in zip(sorted_ms_id, range(N))}
        X[np.isin(sorted_ms_id, self.ms_with_resource), :-1] = torch.tensor(
            self.resource_df.loc[ms_with_features].values, dtype=torch.float
        )
        # 1 for ms with features, 0 for ms without features
        X[np.isin(sorted_ms_id, self.ms_with_resource), -1] = 1

        #################### Node depth ##############################
        span_adj_list = self.build_adj_list(edge_index)

        min_node_depth = self.get_node_depth(
            root_nid, num_nodes, span_adj_list
        )
        min_node_depth = np.array(min_node_depth)
        min_node_depth[np.isinf(min_node_depth)] = 0
        # print("==>> min_node_depth: ", min_node_depth)
        # max_node_depth = np.array(max_node_depth)
        # print("==>> max_node_depth: ", max_node_depth)
        # max_normalization_constant = max(max_node_depth) if max(max_node_depth) > 0 else 1
        # print("==>> max_normalization_constant: ", max_normalization_constant)
        min_normalization_constant = max(min_node_depth) if max(min_node_depth) > 0 else 1
        # print("==>> min_normalization_constant: ", min_normalization_constant)
        # node_depth = np.array(
        #     [min_node_depth / min_normalization_constant, max_node_depth / max_normalization_constant]
        # ).T  # N*2
        node_depth = np.array(
            [min_node_depth / min_normalization_constant]
        ).T  # N*1

        return X, node_depth

    def get_span_edge_features(self):
        span_edge_attr = torch.tensor(
            self.trace_span_df_no_duplicates.loc[:, ["interface", "rpctype"]].values,
            dtype=torch.long,
        ).contiguous()

        # The orderings of the edges
        span_edge_orderings = torch.tensor(
            self.trace_span_df_no_duplicates.loc[:, ["timestamp"]]
            .rank(method="average")
            .values
            / len(self.trace_span_df_no_duplicates),
            dtype=torch.float,
        ).contiguous()

        span_edge_durations = torch.tensor(
            self.trace_span_df_no_duplicates.loc[:, ["rt"]].abs().values,
            dtype=torch.long,
        ).contiguous()

        return span_edge_attr, span_edge_orderings, span_edge_durations

    def get_span_edge_index(self):
        span_edge_index = torch.tensor(
            self.trace_span_df_no_duplicates.loc[:, ["um", "dm"]].values.T,
            dtype=torch.long,
        ).contiguous()

        sorted_unique_ms, span_edge_index = torch.unique(
            span_edge_index, sorted=True, return_inverse=True
        )  # map to consecutive integer starting from zero
        sorted_unique_ms = sorted_unique_ms.cpu().numpy()

        num_nodes = len(sorted_unique_ms)
        span2nid = {span: nid for span, nid in zip(sorted_unique_ms, range(num_nodes))}
        # root_span = self.get_root_spanID(self.trace_span_df_no_duplicates)
        root_nid = span2nid[self.root_span]

        X, node_depth = self.get_node_features(
            sorted_unique_ms, span_edge_index, root_nid, num_nodes
        )

        edge_attr, edge_orderings, edge_durations = self.get_span_edge_features()

        return (
            span_edge_index,
            X,
            torch.tensor(node_depth, dtype=torch.long),
            edge_attr,
            edge_orderings,
            edge_durations,
            sorted_unique_ms,
        )

    def get_pert_edge_index(self):
        # trace_span_df_no_entry = self.trace_span_df[
        # self.trace_span_df.parentSpanID != self.nan_span_int
        # ]  # remove the entry edge because the root is nan, which do not contain cpu/ram features.

        # for neighbor in self.span_root_id:
        #     # order the neighbor by time
        #############################
        # Create all nodes
        # stages[spanID] = list of node id
        stages = dict()
        num_nodes = 0
        edge_index = []
        # interface, rpctype, call indicator, same ms indicator
        # Call indicator: 1 for call, 0 for return
        # Same ms indicator: 1 for same ms (different stages), 0 for different ms
        edge_attr = []  # N_edges * 4

        sorted_span_id = []  # for extracting node features
        for um, count in (
            self.trace_span_df_no_duplicates["um"].value_counts().iteritems()
        ):
            n_nbrs = count
            # For each node in the span graph, there will be (#neighbors * 2 + 1) copy of nodes in the PERT graph
            n_stages = n_nbrs * 2 + 1
            stages[um] = np.arange(n_stages) + num_nodes
            for previous, current in zip(stages[um], stages[um][1:]):
                edge_index.append([previous, current])
                edge_attr.append([0, 0, 1, 1])

            num_nodes = num_nodes + n_stages
            sorted_span_id.extend([um] * n_stages)
        leave_nodes = set(self.trace_span_df_no_duplicates["dm"]).difference(
            set(self.trace_span_df_no_duplicates["um"])
        )
        for leave_node in leave_nodes:
            stages[leave_node] = [num_nodes]
            sorted_span_id.extend([leave_node])
            num_nodes = num_nodes + 1

        # # Create spant2eventtime for calculating durations
        # # e.g. for span A, the event time is [1,3,5,8]
        # # Then the duration for A.1, A.2 and A.3 are (3-1), (5-3), and (8-5) respectively
        # span2eventtime = defaultdict(list)
        # # Add start and endtime first
        # for row_index, row in self.trace_span_df_no_duplicates.iterrows():
        #     span2eventtime[row["dm"]].extend(
        #         [row["timestamp"], row["endTimestamp"]]
        #     )
        # print("Before")
        # print("==>> span2eventtime: ", span2eventtime)

        # Add edges and event time
        for um, group in self.trace_span_df_no_duplicates.groupby("um"):
            # order the neighbor (request and response) by time
            # time2span["timestamp"]["spanid"] = spanid
            # time2span["timestamp"]["mode"] = start/end
            time_mode_span_ft = (
                []
            )  # list of tuple of (time, mode, spanID), mode canbe "start"/"end"
            for row_index, row in group.iterrows():
                time_mode_span_ft.append(
                    (
                        row["timestamp"],
                        "start",
                        row["dm"],
                        row["interface"],
                        row["rpctype"],
                    )
                )
                time_mode_span_ft.append((row["endTimestamp"], "end", row["dm"], 0, 0))
            for i, (time, mode, dm, interface, rpctype) in enumerate(
                sorted(time_mode_span_ft, key=lambda tup: tup[0])
            ):
                if mode == "start":
                    edge_from = stages[um][i]
                    edge_to = stages[dm][0]
                    call_indicator = 1
                elif mode == "end":
                    edge_from = stages[dm][-1]
                    edge_to = stages[um][i + 1]
                    call_indicator = 0
                edge_index.append([edge_from, edge_to])
                edge_attr.append([interface, rpctype, call_indicator, 0])
                # span2eventtime[um].insert(-1, time)
        # print("==>> span2eventtime: ", span2eventtime)

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.long).contiguous()

        ######################### Node features #######################
        # root_span = self.get_root_spanID(self.trace_span_df_no_duplicates)
        root_nid = stages[self.root_span][0]

        sorted_span_id = np.array(sorted_span_id)
        # print("==>> stages: ", stages)
        # print("==>> edge_index: ", edge_index)

        X, node_depth = self.get_node_features(
            sorted_span_id, edge_index, root_nid, num_nodes
        )

        # ###################### Durations ########################
        # # calculate durations from span2eventtime
        # span2durations = dict()
        # for span, eventtime in span2eventtime.items():
        #     eventtime = np.array(eventtime)
        #     span2durations[span] = eventtime[1:] - eventtime[:-1]
        # print("==>> span2durations: ", span2durations)

        # durations = np.array([])
        # previous_span = -1
        # for span_id in sorted_span_id:
        #     if previous_span == span_id:
        #         continue
        #     else:
        #         durations = np.concatenate((durations, span2durations[span_id]))
        #         previous_span = span_id
        # print("==>> durations: ", durations)
        # durations[durations < 0] = 0
        # durations = durations[np.newaxis].T
        # print("==>> X: ", X)
        # print("==>> node_depth: ", node_depth)
        # print("==>> span2eventtime: ", span2eventtime)
        # print("==>> span2durations: ", span2durations)
        # print("==>> durations: ", durations)

        # if X.shape[0] != durations.shape[0]:
        #     print("==>> self.trace_span_df: ", self.trace_span_df)
        #     print("==>> span2eventtime: ", span2eventtime)
        #     print("==>> stages: ", stages)
        #     for span in stages.keys():
        #         print("span: ", span)
        #         print("len of span2eventtime: ", len(span2eventtime[span]))
        #         print("len of stages: ", len(stages[span]))
        #     print("==>> X: ", X)
        #     print("==>> X.shape: ", X.shape)
        #     print("==>> durations: ", durations)
        #     print("==>> durations.shape: ", durations.shape)
        #     print("==>> edge_index: ", edge_index)
        #     import sys

        #     sys.exit()

        return (
            edge_index,
            edge_attr,
            X,
            # torch.tensor(durations, dtype=torch.float),
            torch.tensor(node_depth, dtype=torch.long),
            sorted_span_id,
        )


def find_most_recent_fts(resource_df, ts):
    return resource_df.loc[ts]
    # result = resource_df.iloc[resource_df.index.get_indexer([ts], method="pad")[0]]
    # return result if type(result) == np.int64 else result


# max_kmeans_label = 0
# doc_term_mat = tfidf_microservices(corpus)
# doc_term_mat = hash_vec_microservices(corpus)

# c = time.time()

# kmeans = find_optimal_clusters(doc_term_mat, max_k=200)

# d = time.time()

# kmeans_labels = np.array(kmeans.labels_) + max_kmeans_label + 1
# max_kmeans_label = max(kmeans_labels)


# corpus, sorted_traceids = get_corpus_data(entry_group)

# tr2data_map = update_tr2data_map(
# sorted_traceids, kmeans_labels, tr2ts_map, tr2data_map
# )
