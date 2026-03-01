import random

import math

import dgl
import dgl.function as fn

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dgl.nn.functional import edge_softmax


class HGTLayer(nn.Module):
    """One layer of HGT."""

    def __init__(
        self,
        in_dim,
        out_dim,
        node_dict,
        edge_dict,
        n_heads,
        dropout=0.2,
        use_norm=False,
    ):
        super(HGTLayer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.num_types = len(node_dict)

        self.num_relations = len(edge_dict)

        self.total_rel = self.num_types * self.num_relations * self.num_types

        self.n_heads = n_heads
        self.d_k = out_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        self.att = None

        self.k_linears = nn.ModuleList()
        self.q_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()
        self.a_linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.use_norm = use_norm

        for t in range(self.num_types):
            self.k_linears.append(nn.Linear(in_dim, out_dim))
            self.q_linears.append(nn.Linear(in_dim, out_dim))
            self.v_linears.append(nn.Linear(in_dim, out_dim))
            self.a_linears.append(nn.Linear(out_dim, out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))

        self.relation_pri = nn.Parameter(torch.ones(self.num_relations, self.n_heads))

        self.relation_att = nn.Parameter(
            torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k)
        )

        self.relation_msg = nn.Parameter(
            torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k)
        )

        self.skip = nn.Parameter(torch.ones(self.num_types))
        self.drop = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def forward(self, G, h):
        with G.local_scope():

            node_dict, edge_dict = self.node_dict, self.edge_dict

            for srctype, etype, dsttype in G.canonical_etypes:

                sub_graph = G[srctype, etype, dsttype]

                k_linear = self.k_linears[node_dict[srctype]]
                v_linear = self.v_linears[node_dict[srctype]]
                q_linear = self.q_linears[node_dict[dsttype]]

                k = k_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                v = v_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                q = q_linear(h[dsttype]).view(-1, self.n_heads, self.d_k)

                e_id = self.edge_dict[etype]

                relation_att = self.relation_att[e_id]
                relation_pri = self.relation_pri[e_id]
                relation_msg = self.relation_msg[e_id]

                k = torch.einsum("bij,ijk->bik", k, relation_att)
                v = torch.einsum("bij,ijk->bik", v, relation_msg)

                sub_graph.srcdata["k"] = k
                sub_graph.dstdata["q"] = q
                sub_graph.srcdata["v_%d" % e_id] = v

                sub_graph.apply_edges(fn.v_dot_u("q", "k", "t"))

                attn_score = (
                    sub_graph.edata.pop("t").sum(-1) * relation_pri / self.sqrt_dk
                )

                attn_score = edge_softmax(sub_graph, attn_score, norm_by="dst")

                sub_graph.edata["t"] = attn_score.unsqueeze(-1)

            G.multi_update_all(
                {
                    etype: (
                        fn.u_mul_e("v_%d" % e_id, "t", "m"),
                        fn.sum("m", "t"),
                    )
                    for etype, e_id in edge_dict.items()
                },
                cross_reducer="mean",
            )

            new_h = {}
            for ntype in G.ntypes:

                n_id = node_dict[ntype]
                alpha = torch.sigmoid(self.skip[n_id])
                t = G.nodes[ntype].data["t"].view(-1, self.out_dim)
                trans_out = self.drop(self.a_linears[n_id](t))
                trans_out = trans_out * alpha + h[ntype] * (1 - alpha)
                if self.use_norm:
                    new_h[ntype] = self.norms[n_id](trans_out)
                else:
                    new_h[ntype] = trans_out

            return new_h


class HGTIntra(nn.Module):
    def __init__(
        self,
        node_dict,
        edge_dict,
        region_node_dict,
        region_edge_dict,
        n_inp,
        n_hid,
        n_out,
        n_layers,
        n_heads,
        batch_size,
        agg_method="sum",
        use_norm=True,
    ):
        super(HGTIntra, self).__init__()
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.region_node_dict = region_node_dict
        self.region_edge_dict = region_edge_dict

        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_out = n_out

        self.n_layers = n_layers

        self.n_heads = n_heads

        self.use_norm = use_norm

        self.batch_size = batch_size

        self.agg_method = agg_method
        self.gcs = nn.ModuleList()
        self.adapt_ws = nn.ModuleList()

        for _ in range(len(node_dict)):
            self.adapt_ws.append(nn.Linear(n_inp, n_hid))
        for _ in range(n_layers):
            self.gcs.append(
                HGTLayer(n_hid, n_hid, node_dict, edge_dict, n_heads, use_norm=use_norm)
            )
        self.gcs_list = nn.ModuleList()
        for _ in range(batch_size):
            self.gcs_list.append(self.gcs)
        self.out = nn.ModuleList()
        for _ in range(len(node_dict)):
            self.out.append(nn.Linear(n_hid, n_out))
        self.out_ntype_to_one = nn.Linear(n_out * len(self.node_dict), n_out)

        self.out_ntype_to_one_region = nn.Linear(
            n_out * len(self.region_node_dict), n_out
        )
        self.out_ntype_to_one_pos_or_neg = nn.Linear(
            n_out * len(self.region_node_dict), n_out
        )

    def forward(self, GTList, NodeDictList, seed, device):

        GList = [GT[0] for GT in GTList]
        G1NodeDictList = [GT[1] for GT in GTList]
        G2NodeDictList = [GT[2] for GT in GTList]

        batched_graph = dgl.batch(GList)
        h_batch = {}
        for ntype in batched_graph.ntypes:
            n_id = self.node_dict[ntype]
            h_batch[ntype] = F.gelu(
                self.adapt_ws[n_id](batched_graph.nodes[ntype].data["f"])
            )
        for i in range(self.n_layers):
            h_batch = self.gcs[i](batched_graph, h_batch)

        h = [{} for _ in range(len(GList))]
        node_count = {ntype: 0 for ntype in batched_graph.ntypes}

        for index, G in enumerate(GList):
            for ntype in G.ntypes:
                num_nodes = G.number_of_nodes(ntype)
                h[index][ntype] = h_batch[ntype][
                    node_count[ntype] : node_count[ntype] + num_nodes
                ]
                node_count[ntype] += num_nodes

        out_embedding_list = []
        out_embedding_list_region = []
        out_embedding_list_pos_or_neg = []

        for index in range(len(GList)):

            h_graph = h[index]

            out_embedding_4_1_graph = []
            for ntype in h_graph.keys():
                node_embedding = self.out[self.node_dict[ntype]](h_graph[ntype])
                node_embedding = torch.sum(node_embedding, dim=0, keepdim=True)
                out_embedding_4_1_graph.append(node_embedding)

            out_embedding_4_1_graph = torch.stack(
                out_embedding_4_1_graph, dim=0
            ).reshape(-1)
            out_embedding_list.append(self.out_ntype_to_one(out_embedding_4_1_graph))

            r1_mapping = G1NodeDictList[index]
            out_embedding_4_1_region = []
            for ntype, node_indices in r1_mapping.items():
                if node_indices:
                    node_embedding = self.out[self.node_dict[ntype]](
                        h_graph[ntype][node_indices]
                    )
                    node_embedding = torch.sum(node_embedding, dim=0, keepdim=True)
                    out_embedding_4_1_region.append(node_embedding)

            if out_embedding_4_1_region:

                if len(out_embedding_4_1_region) < len(self.region_node_dict):
                    for _ in range(
                        len(self.region_node_dict) - len(out_embedding_4_1_region)
                    ):
                        out_embedding_4_1_region.append(
                            torch.zeros(1, self.n_out).to(device)
                        )
                out_embedding_4_1_region = torch.stack(
                    out_embedding_4_1_region, dim=0
                ).reshape(-1)
                out_embedding_list_region.append(
                    self.out_ntype_to_one_region(out_embedding_4_1_region)
                )
            else:
                out_embedding_list_region.append(torch.zeros(self.n_out).to(device))

            r2_mapping = G2NodeDictList[index]
            out_embedding_4_1_pos_or_neg = []
            for ntype, node_indices in r2_mapping.items():
                if node_indices:
                    node_embedding = self.out[self.node_dict[ntype]](
                        h_graph[ntype][node_indices]
                    )
                    node_embedding = torch.sum(node_embedding, dim=0, keepdim=True)
                    out_embedding_4_1_pos_or_neg.append(node_embedding)

            if out_embedding_4_1_pos_or_neg:

                if len(out_embedding_4_1_pos_or_neg) < len(self.region_node_dict):
                    for _ in range(
                        len(self.region_node_dict) - len(out_embedding_4_1_pos_or_neg)
                    ):
                        out_embedding_4_1_pos_or_neg.append(
                            torch.zeros(1, self.n_out).to(device)
                        )
                out_embedding_4_1_pos_or_neg = torch.stack(
                    out_embedding_4_1_pos_or_neg, dim=0
                ).reshape(-1)
                out_embedding_list_pos_or_neg.append(
                    self.out_ntype_to_one_pos_or_neg(out_embedding_4_1_pos_or_neg)
                )
            else:
                out_embedding_list_pos_or_neg.append(torch.zeros(self.n_out).to(device))

        out_embedding_list = torch.stack(out_embedding_list, dim=0).reshape(
            self.batch_size, -1
        )
        out_embedding_list_region = torch.stack(
            out_embedding_list_region, dim=0
        ).reshape(self.batch_size, -1)
        out_embedding_list_pos_or_neg = torch.stack(
            out_embedding_list_pos_or_neg, dim=0
        ).reshape(self.batch_size, -1)

        return (
            out_embedding_list,
            out_embedding_list_region,
            out_embedding_list_pos_or_neg,
        )
