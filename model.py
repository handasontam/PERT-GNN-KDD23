import torch
from torch_geometric.nn import (
    global_add_pool,
    TransformerConv,
)
import torch.nn.functional as F
from torch_geometric.nn import Linear


class SAGEDeterministic(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        cat_dims,
        entry_id_max,
        interface_id_max,
        rpctype_id_max,
        hidden_channels,
        num_layers,
        dropout,
    ):
        super(SAGEDeterministic, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            TransformerConv(
                in_channels=in_channels + hidden_channels,
                out_channels=hidden_channels,
                heads=1,
                edge_dim=hidden_channels * 2,
            )
        )
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                TransformerConv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    heads=1,
                    edge_dim=hidden_channels * 2,
                )
            )
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(
            TransformerConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                heads=1,
                edge_dim=hidden_channels * 2,
            )
        )
        self.local_linear = Linear(hidden_channels, 1)
        self.global_linear1 = Linear(hidden_channels * 2, hidden_channels)
        self.global_linear2 = Linear(hidden_channels, 1)
        self.cat_embedding = torch.nn.ModuleList()
        for num_categories in cat_dims:
            self.cat_embedding.append(
                torch.nn.Embedding(num_categories, hidden_channels)
            )

        self.dropout = dropout
        self.entry_embeds = torch.nn.Embedding(entry_id_max + 1, hidden_channels)
        self.interface_embeds = torch.nn.Embedding(
            interface_id_max + 1, hidden_channels
        )
        self.rpctype_embeds = torch.nn.Embedding(rpctype_id_max + 1, hidden_channels)
        self.edge_linear = Linear(-1, hidden_channels * 2)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(
        self,
        x,
        cat_X,
        edge_index,
        edge_attr,
        pattern_num_nodes,
        pattern_probs,
        entry_id,
        batch,
    ):
        cat_embeds = 0
        for i, cat_embed in enumerate(self.cat_embedding):
            cat_embeds += cat_embed(cat_X[:, i])
        x = torch.cat([x, cat_embeds], dim=1)
        edge_embeds = torch.cat(
            [
                self.interface_embeds(edge_attr[:, 0]),
                self.rpctype_embeds(edge_attr[:, 1]),
            ],
            dim=1,
        )

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index=edge_index, edge_attr=edge_embeds)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index=edge_index, edge_attr=edge_embeds)
        local_predict = self.local_linear(x)
        x = x * pattern_probs / pattern_num_nodes
        mean_x = global_add_pool(x, batch)
        global_predict = torch.concat([mean_x, self.entry_embeds(entry_id)], dim=1)

        global_predict = self.global_linear2(
            F.relu(self.global_linear1(global_predict))
        )
        # ensure that it is non-negative
        return global_predict, local_predict
