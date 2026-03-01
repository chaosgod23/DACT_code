import dgl
import networkx as nx
import matplotlib.pyplot as plt
import torch
from collections import defaultdict
from torch.nn import functional as F


def visualize_dgl_graph(dgl_graph):
    # Move the graph to CPU
    dgl_graph = dgl_graph.to('cpu')

    # Convert DGL graph to NetworkX graph
    nx_graph = dgl.to_networkx(dgl_graph)

    # Draw the graph using NetworkX and Matplotlib
    pos = nx.spring_layout(nx_graph, k=2)  # Adjust k to control the distance between nodes
    # pos = nx.spring_layout(nx_graph)
    nx.draw(nx_graph, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)

    # Show the plot
    plt.show()


def get_device():
    # print("gpu device is:")
    # print(os.environ['CUDA_VISIBLE_DEVICES'])
    if torch.cuda.is_available():
        device = torch.device('cuda:1')
    else:
        device = torch.device('cpu')
    return device


def node_to_graph(node_dict, city_graph):
    node_dict_combined = node_dict[0]
    node_dict_r1 = node_dict[1]
    node_dict_r2 = node_dict[2]

    node_dict_combined = {ntype: torch.unique(nodes) for ntype, nodes in node_dict_combined.items()}
    node_dict_r1 = {ntype: torch.unique(nodes) for ntype, nodes in node_dict_r1.items()}
    node_dict_r2 = {ntype: torch.unique(nodes) for ntype, nodes in node_dict_r2.items()}

    g = dgl.node_subgraph(city_graph, node_dict_combined)

    nid_mapping = g.ndata[dgl.NID]

    r1_mapping = {}
    r2_mapping = {}

    for node_type in node_dict_combined.keys():

        combined_nodes = node_dict_combined[node_type].cpu().tolist()

        if node_type in node_dict_r1:
            r1_nodes = node_dict_r1[node_type].cpu().tolist()
        else:
            r1_nodes = []

        if node_type in node_dict_r2:
            r2_nodes = node_dict_r2[node_type].cpu().tolist()
        else:
            r2_nodes = []

        nid_mapping_type = nid_mapping[node_type].cpu().tolist()

        r1_mapping[node_type] = [nid_mapping_type.index(node) for node in r1_nodes if node in nid_mapping_type]
        r2_mapping[node_type] = [nid_mapping_type.index(node) for node in r2_nodes if node in nid_mapping_type]

    return g, r1_mapping, r2_mapping


def combine_inter_region_sub(city_1_graph, city_2_graph, region_node_dict_1, region_node_dict_2):
    g1 = city_1_graph
    g2 = city_2_graph

    region_node_dict_1 = {ntype: torch.unique(nodes) for ntype, nodes in region_node_dict_1.items()}
    region_node_dict_2 = {ntype: torch.unique(nodes) for ntype, nodes in region_node_dict_2.items()}

    sub_g1 = dgl.node_subgraph(g1, region_node_dict_1)
    sub_g2 = dgl.node_subgraph(g2, region_node_dict_2)

    def get_max_node_ids(graph):
        max_ids = {}
        for ntype in graph.ntypes:
            max_ids[ntype] = graph.number_of_nodes(ntype)
        return max_ids

    sub_g1_max_ids = get_max_node_ids(sub_g1)

    def offset_node_ids(graph, offsets):
        edge_data = {}
        for etype in graph.canonical_etypes:
            src, dst = graph.all_edges(etype=etype)
            src = src + offsets[etype[0]]
            dst = dst + offsets[etype[2]]
            edge_data[etype] = (src, dst)
        return edge_data

    sub_g2_edge_data = offset_node_ids(sub_g2, sub_g1_max_ids)


    graph_device = sub_g1.device

    cross_city_edge_data = {}

    l0_types = ["schema_region", "schema_brand", "schema_cate1", "schema_junc_cate", "schema_road_cate"]
    for ntype in l0_types:
        if ntype not in sub_g1.ntypes or ntype not in sub_g2.ntypes:
            continue

        city1_nodes = sub_g1.nodes(ntype).to(graph_device)
        city2_nodes = (sub_g2.nodes(ntype) + sub_g1_max_ids[ntype]).to(graph_device)

        if len(city1_nodes) == 0 or len(city2_nodes) == 0:
            continue
        src = torch.repeat_interleave(city1_nodes, len(city2_nodes))
        dst = city2_nodes.repeat(len(city1_nodes))
        etype = (ntype, f"cross_city_{ntype}", ntype)
        cross_city_edge_data[etype] = (src, dst)

    l1_types = ["l1_brand", "l1_cate1", "l1_junccate", "l1_roadcate"]
    for ntype in l1_types:
        if ntype not in sub_g1.ntypes or ntype not in sub_g2.ntypes:
            continue

        city1_nodes = sub_g1.nodes(ntype).to(graph_device)
        city2_nodes = (sub_g2.nodes(ntype) + sub_g1_max_ids[ntype]).to(graph_device)

        if len(city1_nodes) != len(city2_nodes):
            continue

        etype = (ntype, f"cross_city_{ntype}", ntype)
        cross_city_edge_data[etype] = (city1_nodes, city2_nodes)

    def merge_edge_data(*edge_dicts):
        merged = defaultdict(lambda: {'src': [], 'dst': []})

        for edge_dict in edge_dicts:
            for etype, (src, dst) in edge_dict.items():
                merged[etype]['src'].append(src)
                merged[etype]['dst'].append(dst)

        final = {}
        for etype in merged:
            combined_src = torch.cat(merged[etype]['src'])
            combined_dst = torch.cat(merged[etype]['dst'])
            final[etype] = (combined_src, combined_dst)

        return final

    combined_edge_data = merge_edge_data(
        {etype: sub_g1.edges(etype=etype) for etype in sub_g1.canonical_etypes},
        sub_g2_edge_data,
        cross_city_edge_data
    )

    combined_graph = dgl.heterograph(combined_edge_data)

    for ntype in set(sub_g1.ntypes).union(sub_g2.ntypes):
        num_nodes = combined_graph.number_of_nodes(ntype)

        if ntype in sub_g1.ntypes and 'f' in sub_g1.nodes[ntype].data:
            feat_tensor = sub_g1.nodes[ntype].data['f']
        elif ntype in sub_g2.ntypes and 'f' in sub_g2.nodes[ntype].data:
            feat_tensor = sub_g2.nodes[ntype].data['f']
        else:
            continue

        combined_graph.nodes[ntype].data['f'] = torch.zeros(
            (num_nodes, feat_tensor.size(1)),
            dtype=feat_tensor.dtype,
            device=graph_device
        )

    for ntype in sub_g1.ntypes:
        if 'f' in sub_g1.nodes[ntype].data:
            feat_tensor = sub_g1.nodes[ntype].data['f']
            if feat_tensor.size(0) > 0:
                combined_graph.nodes[ntype].data['f'][:feat_tensor.size(0)] = feat_tensor.to(graph_device)

    for ntype in sub_g2.ntypes:
        if 'f' in sub_g2.nodes[ntype].data:
            feat_tensor = sub_g2.nodes[ntype].data['f']
            if feat_tensor.size(0) > 0:
                offset = sub_g1_max_ids[ntype]
                combined_graph.nodes[ntype].data['f'][offset:offset + feat_tensor.size(0)] = feat_tensor.to(graph_device)

    r1_mapping = {}
    r2_mapping = {}
    for ntype in sub_g1.ntypes:
        r1_mapping[ntype] = [old_id for old_id in sub_g1.nodes(ntype).tolist()]
    for ntype in sub_g2.ntypes:
        r2_mapping[ntype] = [old_id + sub_g1_max_ids[ntype] for old_id in sub_g2.nodes(ntype).tolist()]

    return combined_graph, r1_mapping, r2_mapping


def triplet_loss(h_r_pos, h_r_plus_pos, h_r_neg, h_r_minus_neg, margin=1.0):
    pos_distance = torch.norm(h_r_pos - h_r_plus_pos, p=2, dim=1) ** 2
    neg_distance = torch.norm(h_r_neg - h_r_minus_neg, p=2, dim=1) ** 2
    loss = torch.relu(pos_distance - neg_distance + margin).mean()
    return loss


def mobility_loss(s_emb, d_emb, mob):
    inner_prod = torch.mm(s_emb, d_emb.T)
    ps_hat = F.softmax(inner_prod, dim=-1)
    inner_prod = torch.mm(d_emb, s_emb.T)
    pd_hat = F.softmax(inner_prod, dim=-1)
    loss = torch.sum(-torch.mul(mob, torch.log(ps_hat)) - torch.mul(mob, torch.log(pd_hat)))
    return loss


def cross_loss(h_r1, h_r2, h_s_1, h_s_2, margin=1.0):
    distance_schema1_schema2 = torch.norm(h_s_1 - h_s_2, p=2, dim=1) ** 2
    loss = torch.relu(distance_schema1_schema2).mean()
    return loss
