import copy
import os.path
import random
from collections import defaultdict
from typing import List

import numpy as np
import torch
import dgl

from dgl import DGLGraph
from openai import embeddings
from pydantic.v1.schema import schema
from sympy.parsing.sympy_parser import factorial_notation
from sympy.unify.usympy import construct
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import pickle as pkl
from joblib import dump, load

from triton.language import dtype

from function.load_graph_data import HeteroGraphData
from function.tools import visualize_dgl_graph, mobility_loss


class HeteroGraphDataList:
    def __init__(self, kg_dir_list, reverse=False, export=False, graph_device='cpu', spatial_list=None):
        # load KG data
        self.kg_dir_list = kg_dir_list
        self.reverse = reverse
        self.graph_device = graph_device
        self.spatial_list = spatial_list
        hkg_list = []
        for i, kg_dir in enumerate(kg_dir_list):
            hkg = HeteroGraphData(kg_dir=kg_dir, reverse=reverse, export=export, graph_device=graph_device,
                                  spatial=spatial_list[i])
            hkg_list.append(hkg)
        self.hkg_list = hkg_list
        self.schema_graph_list = []
        self.brand_graph_list = []
        self.cate1_graph_list = []
        self.junccate_graph_list = []
        self.roadcate_graph_list = []
        self.urban_graph_list = []
        self.all_urban_graph = None

    def init_hetero_graph_features(self, image_list=None, flow_list=None, feature_dir_list=None, node_feats_dim=10,
                                   edge_feats_dim=24, graph_device='cpu'):
        for i, hkg in enumerate(self.hkg_list):
            hkg.init_hetero_graph_features(image=image_list[i], flow=flow_list[i], feature_dir=feature_dir_list[i],
                                           node_feats_dim=node_feats_dim, edge_feats_dim=edge_feats_dim, device=graph_device)

    def construct_multi_level_graph_coordinator(self, llm=None, hg_LLM_feature_dir_list=None,
                                                coordinator_link_threshold=None, node_feats_dim=10, edge_feats_dim=24):
        # Construct Level 0 : Schema Graph Coordinator
        ent_range_list = []
        for hkg in self.hkg_list:
            ent_range_list.append(hkg.ent_range)

        ent_type_list = []
        for i in range(len(ent_range_list)):
            ent_type_list.extend(ent_range_list[i].keys())
        ent_type_list = list(set(ent_type_list))
        print(ent_type_list)

        schema_graph = dgl.heterograph(
            {
                ('schema_region', 'schema_NearBy', 'schema_region'): ([0], [0]),
                ('schema_region', 'schema_HasPoi', 'schema_poi'): ([0], [0]),
                ('schema_region', 'schema_HasRoad', 'schema_road'): ([0], [0]),
                ('schema_region', 'schema_HasJunction', 'schema_junction'): ([0], [0]),
                ('schema_region', 'schema_HasStreetView', 'schema_streetview'): ([0], [0]),
                ('schema_road', 'schema_HasRoadCate', 'schema_road_cate'): ([0], [0]),
                ('schema_junction', 'schema_HasJuncCate', 'schema_junc_cate'): ([0], [0]),
                ('schema_poi', 'schema_HasCate1', 'schema_cate1'): ([0], [0]),
                ('schema_poi', 'schema_HasBrand', 'schema_brand'): ([0], [0]),
            }
        )
        schema_graph = schema_graph.to(self.graph_device)
        if llm is None:
            for i in range(len(ent_type_list)):
                schema_graph.nodes['schema_' + ent_type_list[i]].data['f'] = torch.randn(1, node_feats_dim).to(
                    self.graph_device)
        else:
            schema_text = 'This is a schema graph node, type is {}'
            for i in range(len(ent_type_list)):
                embedding = llm.get_embedding(schema_text.format(ent_type_list[i])).data[0].embedding
                embedding = torch.tensor(embedding, dtype=torch.float32).to(self.graph_device).reshape(1,
                                                                                                       node_feats_dim)
                # print(ent_type_list[i])
                # print(len(schema_graph.nodes['schema_' + ent_type_list[i]]))
                schema_graph.nodes['schema_' + ent_type_list[i]].data['f'] = embedding
        print('Schema Graph Node Num', schema_graph.number_of_nodes())
        for i in range(len(self.hkg_list)):
            self.schema_graph_list.append(copy.deepcopy(schema_graph))
        print('Level 0 : Schema Graph Coordinator Constructed')

        # Construct Level 1 : Category,Brand Graph Coordinator
        # Brand, Cate1. JuncCate, RoadCate
        brand_id2embedding = {}
        for i in range(len(hg_LLM_feature_dir_list)):
            brand_id2embedding.update(
                pkl.load(open(hg_LLM_feature_dir_list[i][len(hg_LLM_feature_dir_list[i]) - 1]['brand'], 'rb')))
        cate1_id2embedding = {}
        for i in range(len(hg_LLM_feature_dir_list)):
            cate1_id2embedding.update(
                pkl.load(open(hg_LLM_feature_dir_list[i][len(hg_LLM_feature_dir_list[i]) - 1]['category'], 'rb')))
        junccate_id2embedding = {}
        for i in range(len(hg_LLM_feature_dir_list)):
            junccate_id2embedding.update(
                pkl.load(open(hg_LLM_feature_dir_list[i][len(hg_LLM_feature_dir_list[i]) - 1]['junccate'], 'rb')))
        roadcate_id2embedding = {}
        for i in range(len(hg_LLM_feature_dir_list)):
            roadcate_id2embedding.update(
                pkl.load(open(hg_LLM_feature_dir_list[i][len(hg_LLM_feature_dir_list[i]) - 1]['roadcate'], 'rb')))

        brand_id_list = list(brand_id2embedding.keys())
        brand_embeddings = [brand_id2embedding[brand_id] for brand_id in brand_id_list]
        brand_similarity_matrix = cosine_similarity(brand_embeddings)
        brand_edges = ([], [])
        brand_threshold = coordinator_link_threshold['brand']
        for i in range(len(brand_id_list)):
            for j in range(i + 1, len(brand_id_list)):
                if brand_similarity_matrix[i][j] > brand_threshold:
                    brand_edges[0].append(i)
                    brand_edges[1].append(j)
        brand_graph = dgl.heterograph(
            {
                ('l1_brand', 'l1_brand_sim', 'l1_brand'): (brand_edges[0], brand_edges[1])
            }
        )
        brand_graph = brand_graph.to(self.graph_device)
        brand_graph.nodes['l1_brand'].data['f'] = torch.tensor(brand_embeddings).to(self.graph_device
                                                                                    ).reshape(len(brand_id_list),
                                                                                              node_feats_dim)

        cate1_id_list = list(cate1_id2embedding.keys())
        cate1_embeddings = [cate1_id2embedding[cate1_id] for cate1_id in cate1_id_list]
        cate1_similarity_matrix = cosine_similarity(cate1_embeddings)
        cate1_edges = ([], [])
        cate1_threshold = coordinator_link_threshold['category']
        for i in range(len(cate1_id_list)):
            for j in range(i + 1, len(cate1_id_list)):
                if cate1_similarity_matrix[i][j] > cate1_threshold:
                    cate1_edges[0].append(i)
                    cate1_edges[1].append(j)
        cate1_graph = dgl.heterograph(
            {
                ('l1_cate1', 'l1_cate1_sim', 'l1_cate1'): (cate1_edges[0], cate1_edges[1])
            }
        )
        cate1_graph = cate1_graph.to(self.graph_device)
        cate1_graph.nodes['l1_cate1'].data['f'] = torch.tensor(cate1_embeddings).to(self.graph_device
                                                                                    ).reshape(len(cate1_id_list),
                                                                                              node_feats_dim)

        junccate_id_list = list(junccate_id2embedding.keys())
        junccate_embeddings = [junccate_id2embedding[junccate_id] for junccate_id in junccate_id_list]
        junccate_similarity_matrix = cosine_similarity(junccate_embeddings)
        junccate_edges = ([], [])
        junccate_threshold = coordinator_link_threshold['junccate']
        for i in range(len(junccate_id_list)):
            for j in range(i + 1, len(junccate_id_list)):
                if junccate_similarity_matrix[i][j] > junccate_threshold:
                    junccate_edges[0].append(i)
                    junccate_edges[1].append(j)
        junccate_graph = dgl.heterograph(
            {
                ('l1_junccate', 'l1_junccate_sim', 'l1_junccate'): (junccate_edges[0], junccate_edges[1])
            }
        )
        junccate_graph = junccate_graph.to(self.graph_device)
        junccate_graph.nodes['l1_junccate'].data['f'] = torch.tensor(junccate_embeddings).to(self.graph_device
                                                                                             ).reshape(len(junccate_id_list),
                                                                                                       node_feats_dim)

        roadcate_id_list = list(roadcate_id2embedding.keys())
        roadcate_embeddings = [roadcate_id2embedding[roadcate_id] for roadcate_id in roadcate_id_list]
        roadcate_similarity_matrix = cosine_similarity(roadcate_embeddings)
        roadcate_edges = ([], [])
        roadcate_threshold = coordinator_link_threshold['roadcate']
        for i in range(len(roadcate_id_list)):
            for j in range(i + 1, len(roadcate_id_list)):
                if roadcate_similarity_matrix[i][j] > roadcate_threshold:
                    roadcate_edges[0].append(i)
                    roadcate_edges[1].append(j)
        roadcate_graph = dgl.heterograph(
            {
                ('l1_roadcate', 'l1_roadcate_sim', 'l1_roadcate'): (roadcate_edges[0], roadcate_edges[1])
            }
        )
        roadcate_graph = roadcate_graph.to(self.graph_device)
        roadcate_graph.nodes['l1_roadcate'].data['f'] = torch.tensor(roadcate_embeddings).to(self.graph_device
                                                                                             ).reshape(len(roadcate_id_list),
                                                                                                       node_feats_dim)

        # visualize_dgl_graph(brand_graph)
        # visualize_dgl_graph(cate1_graph)
        # visualize_dgl_graph(junccate_graph)
        # visualize_dgl_graph(roadcate_graph)
        print('Brand Node Num', brand_graph.number_of_nodes('l1_brand'))
        print('Cate1 Node Num', cate1_graph.number_of_nodes('l1_cate1'))
        print('JuncCate Node Num', junccate_graph.number_of_nodes('l1_junccate'))
        print('RoadCate Node Num', roadcate_graph.number_of_nodes('l1_roadcate'))


        for i in range(len(self.hkg_list)):
            self.brand_graph_list.append(copy.deepcopy(brand_graph))
            self.cate1_graph_list.append(copy.deepcopy(cate1_graph))
            self.junccate_graph_list.append(copy.deepcopy(junccate_graph))
            self.roadcate_graph_list.append(copy.deepcopy(roadcate_graph))

        print('Level 1 : Category,Brand Graph Coordinator Constructed')

        for i in range(len(self.hkg_list)):
            # Link Level 0 and Level 1
            urban_graph = self.merge_hetero_graphs_l0tol1(self.schema_graph_list[i], self.roadcate_graph_list[i],
                                                          new_edge_type='level0to1_RoadCate',
                                                          new_node_type1='schema_road_cate',
                                                          new_node_type2='l1_roadcate')
            urban_graph = self.merge_hetero_graphs_l0tol1(urban_graph, self.junccate_graph_list[i],
                                                          new_edge_type='level0to1_JuncCate',
                                                          new_node_type1='schema_junc_cate',
                                                          new_node_type2='l1_junccate')
            urban_graph = self.merge_hetero_graphs_l0tol1(urban_graph, self.cate1_graph_list[i],
                                                          new_edge_type='level0to1_Cate1',
                                                          new_node_type1='schema_cate1', new_node_type2='l1_cate1')
            urban_graph = self.merge_hetero_graphs_l0tol1(urban_graph, self.brand_graph_list[i],
                                                          new_edge_type='level0to1_Brand',
                                                          new_node_type1='schema_brand', new_node_type2='l1_brand')
            print('Link Level 0 and Level 1')
            print('Urban Graph Node Num', urban_graph.number_of_nodes())
            print('Urban Graph Edge Num', urban_graph.number_of_edges())

            # Link Level 1 and Level 2
            urban_graph = self.merge_hetero_graphs_l1tol2(urban_graph, self.hkg_list[i])
            print('Link Level 1 and Level 2')
            print('Urban Graph Node Num', urban_graph.number_of_nodes())
            print('Urban Graph Edge Num', urban_graph.number_of_edges())

            # Link Level 0 and Level 2
            urban_graph = self.merge_hetero_graphs_l0tol2(urban_graph)
            print('Link Level 0 and Level 2')
            print('Urban Graph Node Num', urban_graph.number_of_nodes())
            print('Urban Graph Edge Num', urban_graph.number_of_edges())

            # visualize_dgl_graph(urban_graph)
            self.urban_graph_list.append(urban_graph)
            print('Urban Graph {} Constructed'.format(i))
        print('All Urban Graph Constructed')

        # Link Level 0 and Level 0
        # Link Level 1 and Level 1
        self.all_urban_graph = self.merge_hetero_graphs_citytocity()
        print('All Urban Graph Constructed and City to City Connected')



    def merge_hetero_graphs_l0tol1(self, graph1, graph2, new_edge_type=None, new_node_type1=None, new_node_type2=None):
        # Extract nodes and edges from graph1
        graph1_data = {}
        for etype in graph1.canonical_etypes:
            src, dst = graph1.all_edges(order='eid', etype=etype)
            graph1_data[etype] = (src, dst)

        # Extract nodes and edges from graph2
        graph2_data = {}
        for etype in graph2.canonical_etypes:
            src, dst = graph2.all_edges(order='eid', etype=etype)
            graph2_data[etype] = (src, dst)

        # Combine nodes and edges
        combined_data = {**graph1_data, **graph2_data}

        # Add new edges
        graph3_data = {}
        if new_edge_type is not None and new_node_type1 is not None and new_node_type2 is not None:
            etype = (new_node_type1, new_edge_type, new_node_type2)
            node_num1 = graph1.number_of_nodes(new_node_type1)
            node_num2 = graph2.number_of_nodes(new_node_type2)
            graph3_data[etype] = (torch.empty(0, dtype=torch.int64).to(self.graph_device),
                                  torch.empty(0, dtype=torch.int64).to(self.graph_device))
            for i in range(node_num1):
                for j in range(node_num2):
                    graph3_data[etype] = (torch.cat((graph3_data[etype][0], torch.tensor([i]).to(self.graph_device))),
                                          torch.cat((graph3_data[etype][1], torch.tensor([j]).to(self.graph_device))))
            combined_data = {**combined_data, **graph3_data}

        # Create a new heterogeneous graph with the combined nodes and edges
        combined_graph = dgl.heterograph(combined_data)

        # Copy node data
        for ntype in graph1.ntypes:
            combined_graph.nodes[ntype].data.update(graph1.nodes[ntype].data)
        for ntype in graph2.ntypes:
            combined_graph.nodes[ntype].data.update(graph2.nodes[ntype].data)

        # Copy edge data
        for etype in graph1.canonical_etypes:
            combined_graph.edges[etype].data.update(graph1.edges[etype].data)
        for etype in graph2.canonical_etypes:
            combined_graph.edges[etype].data.update(graph2.edges[etype].data)

        return combined_graph

    def merge_hetero_graphs_l1tol2(self, graph1, graph2):
        # Extract nodes and edges from graph1
        graph1_data = {}
        for etype in graph1.canonical_etypes:
            src, dst = graph1.all_edges(order='eid', etype=etype)
            graph1_data[etype] = (src, dst)

        # Extract nodes and edges from graph2
        graph2_data = {}
        for etype in graph2.hg.canonical_etypes:
            src, dst = graph2.hg.all_edges(order='eid', etype=etype)
            graph2_data[etype] = (src, dst)

        # Combine nodes and edges
        combined_data = {**graph1_data, **graph2_data}

        graphBrand_data = {}
        etype = ('l1_brand', 'level1to2_Brand', 'brand')
        graphBrand_data[etype] = (torch.empty(0, dtype=torch.int64).to(self.graph_device),
                                  torch.empty(0, dtype=torch.int64).to(self.graph_device))
        for i in tqdm(range(graph1.number_of_nodes('l1_brand'))):
            for j in range(graph2.hg.number_of_nodes('brand')):
                if torch.equal(graph1.nodes['l1_brand'].data['f'][i], graph2.hg.nodes['brand'].data['f'][j]):
                    # print('Brand', i, j)
                    graphBrand_data[etype] = (
                        torch.cat((graphBrand_data[etype][0], torch.tensor([i]).to(self.graph_device))),
                        torch.cat((graphBrand_data[etype][1], torch.tensor([j]).to(self.graph_device))
                                  )
                    )
                    break
        combined_data = {**combined_data, **graphBrand_data}
        graphCate1_data = {}
        etype = ('l1_cate1', 'level1to2_Cate1', 'cate1')
        graphCate1_data[etype] = (torch.empty(0, dtype=torch.int64).to(self.graph_device),
                                  torch.empty(0, dtype=torch.int64).to(self.graph_device))
        for i in tqdm(range(graph1.number_of_nodes('l1_cate1'))):
            for j in range(graph2.hg.number_of_nodes('cate1')):
                if torch.equal(graph1.nodes['l1_cate1'].data['f'][i], graph2.hg.nodes['cate1'].data['f'][j]):
                    # print('Cate1', i, j)
                    graphCate1_data[etype] = (
                        torch.cat((graphCate1_data[etype][0], torch.tensor([i]).to(self.graph_device))),
                        torch.cat((graphCate1_data[etype][1], torch.tensor([j]).to(self.graph_device))
                                  )
                    )
                    break
        combined_data = {**combined_data, **graphCate1_data}
        graphJuncCate_data = {}
        etype = ('l1_junccate', 'level1to2_JuncCate', 'junc_cate')
        graphJuncCate_data[etype] = (torch.empty(0, dtype=torch.int64).to(self.graph_device),
                                     torch.empty(0, dtype=torch.int64).to(self.graph_device))
        for i in tqdm(range(graph1.number_of_nodes('l1_junccate'))):
            for j in range(graph2.hg.number_of_nodes('junc_cate')):
                if torch.equal(graph1.nodes['l1_junccate'].data['f'][i], graph2.hg.nodes['junc_cate'].data['f'][j]):
                    # print('JuncCate', i, j)
                    graphJuncCate_data[etype] = (
                        torch.cat((graphJuncCate_data[etype][0], torch.tensor([i]).to(self.graph_device))),
                        torch.cat((graphJuncCate_data[etype][1], torch.tensor([j]).to(self.graph_device))
                                  )
                    )
                    break
        combined_data = {**combined_data, **graphJuncCate_data}
        graphRoadCate_data = {}
        etype = ('l1_roadcate', 'level1to2_RoadCate', 'road_cate')
        graphRoadCate_data[etype] = (torch.empty(0, dtype=torch.int64).to(self.graph_device),
                                     torch.empty(0, dtype=torch.int64).to(self.graph_device))
        for i in tqdm(range(graph1.number_of_nodes('l1_roadcate'))):
            for j in range(graph2.hg.number_of_nodes('road_cate')):
                if torch.equal(graph1.nodes['l1_roadcate'].data['f'][i], graph2.hg.nodes['road_cate'].data['f'][j]):
                    # print('RoadCate', i, j)
                    graphRoadCate_data[etype] = (
                        torch.cat((graphRoadCate_data[etype][0], torch.tensor([i]).to(self.graph_device))),
                        torch.cat((graphRoadCate_data[etype][1], torch.tensor([j]).to(self.graph_device)))
                    )
                    break
        combined_data = {**combined_data, **graphRoadCate_data}
        # Create a new heterogeneous graph with the combined nodes and edges
        combined_graph = dgl.heterograph(combined_data)

        # Copy node data
        for ntype in graph1.ntypes:
            combined_graph.nodes[ntype].data.update(graph1.nodes[ntype].data)
        for ntype in graph2.hg.ntypes:
            combined_graph.nodes[ntype].data.update(graph2.hg.nodes[ntype].data)

        # Copy edge data
        for etype in graph1.canonical_etypes:
            combined_graph.edges[etype].data.update(graph1.edges[etype].data)
        for etype in graph2.hg.canonical_etypes:
            combined_graph.edges[etype].data.update(graph2.hg.edges[etype].data)

        return combined_graph

    def merge_hetero_graphs_l0tol2(self, graph1):
        # Extract nodes and edges from graph1
        graph1_data = {}
        for etype in graph1.canonical_etypes:
            src, dst = graph1.all_edges(order='eid', etype=etype)
            graph1_data[etype] = (src, dst)

        graphRegion_data = {}
        etype = ('schema_region', 'level0to2_Region', 'region')
        graphRegion_data[etype] = (torch.empty(0, dtype=torch.int64).to(self.graph_device),
                                   torch.empty(0, dtype=torch.int64).to(self.graph_device))
        for i in range(graph1.number_of_nodes('schema_region')):
            for j in range(graph1.number_of_nodes('region')):
                graphRegion_data[etype] = (
                    torch.cat((graphRegion_data[etype][0], torch.tensor([i]).to(self.graph_device))),
                    torch.cat((graphRegion_data[etype][1], torch.tensor([j]).to(self.graph_device))
                              )
                )
        combined_data = {**graph1_data, **graphRegion_data}

        # Create a new heterogeneous graph with the combined nodes and edges
        combined_graph = dgl.heterograph(combined_data)

        # Copy node data
        for ntype in graph1.ntypes:
            combined_graph.nodes[ntype].data.update(graph1.nodes[ntype].data)

        # Copy edge data
        for etype in graph1.canonical_etypes:
            combined_graph.edges[etype].data.update(graph1.edges[etype].data)


        return combined_graph


    def merge_hetero_graphs_citytocity(self):

        all_urban_graph_data = {}
        for i in range(len(self.urban_graph_list)):
            for etype in self.urban_graph_list[i].canonical_etypes:
                src, dst = self.urban_graph_list[i].all_edges(order='eid', etype=etype)
                new_etype = ('city' + str(i) + '_' + etype[0], etype[1], 'city' + str(i) + '_' + etype[2])
                all_urban_graph_data[new_etype] = (src, dst)

        citytocity_schema_region_data = {}
        citytocity_schema_brand_data = {}
        citytocity_schema_cate1_data = {}
        citytocity_schema_junc_cate_data = {}
        citytocity_schema_road_cate_data = {}
        citytocity_schema_poi_data = {}
        citytocity_schema_road_data = {}
        citytocity_schema_junction_data = {}
        citytocity_schema_streetview_data = {}
        for i in range(len(self.urban_graph_list)):
            for j in range(i + 1, len(self.urban_graph_list)):

                etype = ('city' + str(i) + '_schema_region', 'citytocity_Schema_NearBy',
                         'city' + str(j) + '_schema_region')
                citytocity_schema_region_data[etype] = (torch.tensor([0]).to(self.graph_device),
                                                        torch.tensor([0]).to(self.graph_device))

                etype = ('city' + str(i) + '_schema_brand', 'citytocity_Schema_HasBrand',
                         'city' + str(j) + '_schema_brand')
                citytocity_schema_brand_data[etype] = (torch.tensor([0]).to(self.graph_device),
                                                       torch.tensor([0]).to(self.graph_device))

                etype = ('city' + str(i) + '_schema_cate1', 'citytocity_Schema_HasCate1',
                         'city' + str(j) + '_schema_cate1')
                citytocity_schema_cate1_data[etype] = (torch.tensor([0]).to(self.graph_device),
                                                       torch.tensor([0]).to(self.graph_device))

                etype = ('city' + str(i) + '_schema_junc_cate', 'citytocity_Schema_HasJuncCate',
                         'city' + str(j) + '_schema_junc_cate')
                citytocity_schema_junc_cate_data[etype] = (torch.tensor([0]).to(self.graph_device),
                                                           torch.tensor([0]).to(self.graph_device))

                etype = ('city' + str(i) + '_schema_road_cate', 'citytocity_Schema_HasRoadCate',
                         'city' + str(j) + '_schema_road_cate')
                citytocity_schema_road_cate_data[etype] = (torch.tensor([0]).to(self.graph_device),
                                                           torch.tensor([0]).to(self.graph_device))

                etype = ('city' + str(i) + '_schema_poi', 'citytocity_Schema_HasPoi',
                         'city' + str(j) + '_schema_poi')
                citytocity_schema_poi_data[etype] = (torch.tensor([0]).to(self.graph_device),
                                                     torch.tensor([0]).to(self.graph_device))

                etype = ('city' + str(i) + '_schema_road', 'citytocity_Schema_HasRoad',
                         'city' + str(j) + '_schema_road')
                citytocity_schema_road_data[etype] = (torch.tensor([0]).to(self.graph_device),
                                                      torch.tensor([0]).to(self.graph_device))

                etype = ('city' + str(i) + '_schema_junction', 'citytocity_Schema_HasJunction',
                         'city' + str(j) + '_schema_junction')
                citytocity_schema_junction_data[etype] = (torch.tensor([0]).to(self.graph_device),
                                                          torch.tensor([0]).to(self.graph_device))

                etype = ('city' + str(i) + '_schema_streetview', 'citytocity_Schema_HasStreetView',
                         'city' + str(j) + '_schema_streetview')
                citytocity_schema_streetview_data[etype] = (torch.tensor([0]).to(self.graph_device),
                                                            torch.tensor([0]).to(self.graph_device))

        all_urban_graph_data = {**all_urban_graph_data, **citytocity_schema_region_data}
        all_urban_graph_data = {**all_urban_graph_data, **citytocity_schema_brand_data}
        all_urban_graph_data = {**all_urban_graph_data, **citytocity_schema_cate1_data}
        all_urban_graph_data = {**all_urban_graph_data, **citytocity_schema_junc_cate_data}
        all_urban_graph_data = {**all_urban_graph_data, **citytocity_schema_road_cate_data}
        all_urban_graph_data = {**all_urban_graph_data, **citytocity_schema_poi_data}
        all_urban_graph_data = {**all_urban_graph_data, **citytocity_schema_road_data}
        all_urban_graph_data = {**all_urban_graph_data, **citytocity_schema_junction_data}
        all_urban_graph_data = {**all_urban_graph_data, **citytocity_schema_streetview_data}

        citytocity_l1_brand_data = {}
        citytocity_l1_cate1_data = {}
        citytocity_l1_junccate_data = {}
        citytocity_l1_roadcate_data = {}
        for i in range(len(self.urban_graph_list)):
            for j in range(i + 1, len(self.urban_graph_list)):
                etype = ('city' + str(i) + '_l1_brand', 'citytocity_L1_Brand', 'city' + str(j) + '_l1_brand')
                citytocity_l1_brand_data[etype] = (torch.empty(0, dtype=torch.int64).to(self.graph_device),
                                                   torch.empty(0, dtype=torch.int64).to(self.graph_device))
                for k in tqdm(range(self.brand_graph_list[i].number_of_nodes('l1_brand'))):
                    for l in range(self.brand_graph_list[j].number_of_nodes('l1_brand')):
                        if torch.equal(self.brand_graph_list[i].nodes['l1_brand'].data['f'][k],
                                       self.brand_graph_list[j].nodes['l1_brand'].data['f'][l]):
                            citytocity_l1_brand_data[etype] = (
                                torch.cat((citytocity_l1_brand_data[etype][0], torch.tensor([k]).to(self.graph_device))),
                                torch.cat((citytocity_l1_brand_data[etype][1], torch.tensor([l]).to(self.graph_device)))
                            )
                            break
                etype = ('city' + str(i) + '_l1_cate1', 'citytocity_L1_Cate1', 'city' + str(j) + '_l1_cate1')
                citytocity_l1_cate1_data[etype] = (torch.empty(0, dtype=torch.int64).to(self.graph_device),
                                                   torch.empty(0, dtype=torch.int64).to(self.graph_device))
                for k in tqdm(range(self.cate1_graph_list[i].number_of_nodes('l1_cate1'))):
                    for l in range(self.cate1_graph_list[j].number_of_nodes('l1_cate1')):
                        if torch.equal(self.cate1_graph_list[i].nodes['l1_cate1'].data['f'][k],
                                       self.cate1_graph_list[j].nodes['l1_cate1'].data['f'][l]):
                            citytocity_l1_cate1_data[etype] = (
                                torch.cat((citytocity_l1_cate1_data[etype][0], torch.tensor([k]).to(self.graph_device))),
                                torch.cat((citytocity_l1_cate1_data[etype][1], torch.tensor([l]).to(self.graph_device)))
                            )
                            break
                etype = ('city' + str(i) + '_l1_junccate', 'citytocity_L1_JuncCate', 'city' + str(j) + '_l1_junccate')
                citytocity_l1_junccate_data[etype] = (torch.empty(0, dtype=torch.int64).to(self.graph_device),
                                                      torch.empty(0, dtype=torch.int64).to(self.graph_device))
                for k in tqdm(range(self.junccate_graph_list[i].number_of_nodes('l1_junccate'))):
                    for l in range(self.junccate_graph_list[j].number_of_nodes('l1_junccate')):
                        if torch.equal(self.junccate_graph_list[i].nodes['l1_junccate'].data['f'][k],
                                       self.junccate_graph_list[j].nodes['l1_junccate'].data['f'][l]):
                            citytocity_l1_junccate_data[etype] = (
                                torch.cat((citytocity_l1_junccate_data[etype][0], torch.tensor([k]).to(self.graph_device))),
                                torch.cat((citytocity_l1_junccate_data[etype][1], torch.tensor([l]).to(self.graph_device)))
                            )
                            break
                etype = ('city' + str(i) + '_l1_roadcate', 'citytocity_L1_RoadCate', 'city' + str(j) + '_l1_roadcate')
                citytocity_l1_roadcate_data[etype] = (torch.empty(0, dtype=torch.int64).to(self.graph_device),
                                                      torch.empty(0, dtype=torch.int64).to(self.graph_device))
                for k in tqdm(range(self.roadcate_graph_list[i].number_of_nodes('l1_roadcate'))):
                    for l in range(self.roadcate_graph_list[j].number_of_nodes('l1_roadcate')):
                        if torch.equal(self.roadcate_graph_list[i].nodes['l1_roadcate'].data['f'][k],
                                       self.roadcate_graph_list[j].nodes['l1_roadcate'].data['f'][l]):
                            citytocity_l1_roadcate_data[etype] = (
                                torch.cat((citytocity_l1_roadcate_data[etype][0], torch.tensor([k]).to(self.graph_device))),
                                torch.cat((citytocity_l1_roadcate_data[etype][1], torch.tensor([l]).to(self.graph_device)))
                            )
                            break

        all_urban_graph_data = {**all_urban_graph_data, **citytocity_l1_brand_data}
        all_urban_graph_data = {**all_urban_graph_data, **citytocity_l1_cate1_data}
        all_urban_graph_data = {**all_urban_graph_data, **citytocity_l1_junccate_data}
        all_urban_graph_data = {**all_urban_graph_data, **citytocity_l1_roadcate_data}

        all_urban_graph = dgl.heterograph(all_urban_graph_data)

        # Copy node data
        for i in range(len(self.urban_graph_list)):
            for ntype in self.urban_graph_list[i].ntypes:
                cityi_ntype = 'city' + str(i) + '_' + ntype
                all_urban_graph.nodes[cityi_ntype].data.update(self.urban_graph_list[i].nodes[ntype].data)

        # Copy edge data
        for i in range(len(self.urban_graph_list)):
            for etype in self.urban_graph_list[i].canonical_etypes:
                cityi_etype = ('city' + str(i) + '_' + etype[0], etype[1], 'city' + str(i) + '_' + etype[2])
                all_urban_graph.edges[cityi_etype].data.update(self.urban_graph_list[i].edges[etype].data)


        return all_urban_graph


class CrossUrbanData:
    def __init__(self, args=None, hkg_list=None, graph_device='cpu'):
        self.args = args
        self.seed = args['seed']
        self.hkg_list = hkg_list
        self.graph_device = graph_device

        self.intra_all_region_nearby_list = []
        if self.args['intra_all_region_nearby_list_dir'] is not None:
            self.intra_all_region_nearby_list = load(self.args['intra_all_region_nearby_list_dir'])
            # to device
            for i in range(len(self.intra_all_region_nearby_list)):
                for j in range(len(self.intra_all_region_nearby_list[i])):
                    self.intra_all_region_nearby_list[i][j] = self.intra_all_region_nearby_list[i][j].to(self.graph_device)
            print('Intra All Region Nearby List Loaded')
        else:
            for i in range(len(self.hkg_list.urban_graph_list)):
                self.intra_all_region_nearby_list.append(self.get_intra_all_region_nearby(i))
            dump(
                self.intra_all_region_nearby_list,
                'intra_all_region_nearby_list.joblib',
                compress=3,
                protocol=pkl.HIGHEST_PROTOCOL
            )

        pass

    def get_all_intra_samples_batch(self, sp_pos_samples, sp_neg_samples, mobility_samples, flow_in_samples, flow_out_samples, batch_size, batch_num, if_pre_solve = False):
        # spatial
        sp_pos_batch = []
        sp_neg_batch = []
        for i in range(batch_num):
            sp_pos_city_batch = []
            sp_neg_city_batch = []
            for j in range(len(self.hkg_list.urban_graph_list)):
                sp_pos_region_batch = []
                sp_neg_region_batch = []
                for k in range(i % batch_size[j]  * batch_size[j], (i + 1) % batch_size[j] * batch_size[j]):
                    if if_pre_solve:
                        # random seed
                        # random.seed(self.seed)
                        # random select 1 sample
                        sp_pos_region_batch.append(random.sample(sp_pos_samples[j][k], 1)[0])
                        sp_neg_region_batch.append(random.sample(sp_neg_samples[j][k], 1)[0])
                    else:
                        sp_pos_region_batch_id = random.sample(sp_pos_samples[j][k], 1)[0]
                        sp_neg_region_batch_id = random.sample(sp_neg_samples[j][k], 1)[0]
                        # sp_pos_region_batch.append(sp_pos_region_batch_id)
                        # sp_neg_region_batch.append(sp_neg_region_batch_id)
                        sp_pos_region_batch.append(self.get_intra_region_sub(j, k, sp_pos_region_batch_id))
                        sp_neg_region_batch.append(self.get_intra_region_sub(j, k, sp_neg_region_batch_id))
                sp_pos_city_batch.append(sp_pos_region_batch)
                sp_neg_city_batch.append(sp_neg_region_batch)
            sp_pos_batch.append(sp_pos_city_batch)
            sp_neg_batch.append(sp_neg_city_batch)
        # mobility
        mobility_batch = []
        flow_in_batch = []
        flow_out_batch = []
        for i in range(batch_num):
            mobility_city_batch = []
            flow_in_city_batch = []
            flow_out_city_batch = []
            for j in range(len(self.hkg_list.urban_graph_list)):
                mobility_city = torch.tensor(mobility_samples[j], dtype=torch.float32).to(self.graph_device)
                mobility_city_batch_size = mobility_city[i * batch_size[j]: (i + 1) * batch_size[j], i * batch_size[j]: (i + 1) * batch_size[j]]
                mobility_city_batch.append(mobility_city_batch_size)
                flow_in_city_batch.append(flow_in_samples[j][i * batch_size[j]: (i + 1) * batch_size[j]])
                flow_out_city_batch.append(flow_out_samples[j][i * batch_size[j]: (i + 1) * batch_size[j]])
            mobility_batch.append(mobility_city_batch)
            flow_in_batch.append(flow_in_city_batch)
            flow_out_batch.append(flow_out_city_batch)

        return sp_pos_batch, sp_neg_batch, mobility_batch, flow_in_batch, flow_out_batch

    def get_all_inter_samples_batch(self, inter_pair_samples, batch_size, batch_num):
        inter_pair_samples_batch = []
        for m in range(batch_num):
            inter_pair_city_batch = []
            for i in range(len(self.hkg_list.urban_graph_list)):
                inter_pair_city = []
                batch_size_city = batch_size[i]
                for k in range(self.hkg_list.urban_graph_list[i].number_of_nodes('region')):
                    k_region_num = 0
                    k_region = []
                    for j in range(i + 1, len(self.hkg_list.urban_graph_list)):
                        for l in range(self.hkg_list.urban_graph_list[j].number_of_nodes('region')):
                            if inter_pair_samples[i][j][k][l] is not None:
                                k_region_num += 1
                                k_region.append((i, j, k, l, inter_pair_samples[i][j][k][l]))
                    if k_region_num == 0:
                        continue

                    inter_pair_city.append(random.sample(k_region, 1))

                inter_pair_city_batch.append(inter_pair_city[m * batch_size_city: (m + 1) * batch_size_city])

            inter_pair_samples_batch.append(inter_pair_city_batch)

        return inter_pair_samples_batch

    def get_all_samples(self, if_pre_solve = False):
        sp_pos_samples, sp_neg_samples, mobility_samples, flow_in_samples, flow_out_samples = self.construct_intra_samples(if_pre_solve= if_pre_solve)
        inter_pair_samples = None

        return sp_pos_samples, sp_neg_samples, mobility_samples, flow_in_samples, flow_out_samples, inter_pair_samples

    def construct_intra_samples(self, if_pre_solve = False):
        sp_pos_samples, sp_neg_samples = self.get_intra_spatial_samples(if_pre_solve= if_pre_solve)
        mobility_samples, flow_in_samples, flow_out_samples = self.get_intra_human_mobility_samples()
        return sp_pos_samples, sp_neg_samples, mobility_samples, flow_in_samples, flow_out_samples

    def construct_inter_samples(self):
        inter_pair_samples = self.get_inter_pair_samples()
        return inter_pair_samples

    def get_intra_spatial_samples(self, if_pre_solve = False):
        if if_pre_solve:
            sp_pos_samples = []
            sp_neg_samples = []
            for i in range(len(self.hkg_list.urban_graph_list)):
                sp_pos_city = []
                sp_neg_city = []
                for j in range(self.hkg_list.urban_graph_list[i].number_of_nodes('region')):
                    region_nearby = self.intra_all_region_nearby_list[i][j]
                    sp_pos_list = []
                    sp_neg_list = []
                    for k in range(self.hkg_list.urban_graph_list[i].number_of_nodes('region')):
                        x = min(j, k)
                        y = max(j, k)
                        if k == j:
                            continue
                        if k in region_nearby:
                            sp_pos_list.append(self.intra_all_region_pair_sub_list[i][x][y - x - 1])
                        else:
                            sp_neg_list.append(self.intra_all_region_pair_sub_list[i][x][y - x - 1])
                    sp_pos_city.append(sp_pos_list)
                    sp_neg_city.append(sp_neg_list)
                sp_pos_samples.append(sp_pos_city)
                sp_neg_samples.append(sp_neg_city)
            return sp_pos_samples, sp_neg_samples
        else:
            sp_pos_samples = []
            sp_neg_samples = []
            for i in range(len(self.hkg_list.urban_graph_list)):
                sp_pos_city = []
                sp_neg_city = []
                for j in range(self.hkg_list.urban_graph_list[i].number_of_nodes('region')):
                    region_nearby = self.intra_all_region_nearby_list[i][j]
                    sp_pos_list = []
                    sp_neg_list = []
                    for k in range(self.hkg_list.urban_graph_list[i].number_of_nodes('region')):
                        x = min(j, k)
                        y = max(j, k)
                        if k == j:
                            continue
                        if k in region_nearby:
                            sp_pos_list.append(k)
                        else:
                            sp_neg_list.append(k)
                    sp_pos_city.append(sp_pos_list)
                    sp_neg_city.append(sp_neg_list)
                sp_pos_samples.append(sp_pos_city)
                sp_neg_samples.append(sp_neg_city)
            return sp_pos_samples, sp_neg_samples

    def get_intra_human_mobility_samples(self):
        mobility = []
        for i in range(len(self.hkg_list.urban_graph_list)):
            mobility_city = np.load(self.args['mobility_dir'][i])
            mobility.append(mobility_city)
        flow_in = []
        flow_out = []
        for i in range(len(self.hkg_list.urban_graph_list)):
            flow_in_city = np.load(self.args['flow_dir'][i][0])
            flow_out_city = np.load(self.args['flow_dir'][i][1])
            scaler = StandardScaler()
            # flow_in_city = scaler.fit_transform(flow_in_city)
            # flow_out_city = scaler.fit_transform(flow_out_city)
            flow_in_city = (flow_in_city - np.min(flow_in_city)) / (np.max(flow_in_city) - np.min(flow_in_city))
            flow_out_city = (flow_out_city - np.min(flow_out_city)) / (np.max(flow_out_city) - np.min(flow_out_city))

            flow_in_city = torch.tensor(flow_in_city, dtype=torch.float32).to(self.graph_device)
            flow_out_city = torch.tensor(flow_out_city, dtype=torch.float32).to(self.graph_device)
            flow_in_city = torch.cat([flow_in_city for _ in range(6)], dim=1)
            flow_out_city = torch.cat([flow_out_city for _ in range(6)], dim=1)
            flow_in.append(flow_in_city)
            flow_out.append(flow_out_city)
        return mobility, flow_in, flow_out


    def get_inter_pair_samples(self):
        combined_all_inter_pair_samples = self.combine_all_inter_pair_samples()
        return combined_all_inter_pair_samples

    def combine_all_inter_pair_samples(self):
        inter_pair_samples = [[[[None for _ in range(len(self.inter_all_region_pair_sub_list[0][0][0]))]
                                for _ in range(len(self.inter_all_region_pair_sub_list[0][0]))]
                               for _ in range(len(self.inter_all_region_pair_sub_list[0]))]
                              for _ in range(len(self.inter_all_region_pair_sub_list))]

        for i in range(len(self.inter_all_region_pair_sub_list)):
            for j in range(len(self.inter_all_region_pair_sub_list[i])):
                for k in range(len(self.inter_all_region_pair_sub_list[i][j])):
                    for l in range(len(self.inter_all_region_pair_sub_list[i][j][k])):
                        if self.inter_all_region_pair_sub_list[i][j][k][l] is not None:
                            region_node_dict_1 = self.inter_all_region_pair_sub_list[i][j][k][l][0]
                            region_node_dict_2 = self.inter_all_region_pair_sub_list[i][j][k][l][1]
                            inter_pair_samples[i][j][k][l] = (region_node_dict_1, region_node_dict_2)

        return inter_pair_samples

    def get_intra_all_region_nearby(self, city_id):
        region_num = self.hkg_list.urban_graph_list[city_id].number_of_nodes('region')
        region_nearby_list = []
        print('Intra All Region Nearby Constructing')
        for i in tqdm(range(region_num)):
            region_nearby = self.get_intra_region_nearby(city_id, i)
            region_nearby_list.append(region_nearby)
        print('Intra All Region Nearby Constructed')
        return region_nearby_list

    def get_intra_region_nearby(self, city_id, region_id):
        nearby_region_list = self.hkg_list.urban_graph_list[city_id].successors(region_id, etype='NearBy')
        return nearby_region_list

    def get_intra_all_region_pair_sub(self, city_id):
        region_num = self.hkg_list.urban_graph_list[city_id].number_of_nodes('region')
        region_pair_sub_node_list = [[] for i in range(region_num)]
        print('Intra All Region Pair Sub Graph Constructing')
        for i in tqdm(range(region_num)):
            for j in range(i + 1, region_num):
                region_pair_sub_node = self.get_intra_region_sub(city_id, i, j)
                region_pair_sub_node_list[i].append(region_pair_sub_node)
        print('Intra All Region Pair Sub Graph Constructed')
        return region_pair_sub_node_list

    def get_intra_region_sub(self, city_id, region_1_id, region_2_id):
        l0_nodes_dict, l1_nodes_dict = self.get_l0_and_l1_nodes(city_id)
        region_1_l2_nodes = self.get_l2_nodes(city_id, region_1_id)
        region_2_l2_nodes = self.get_l2_nodes(city_id, region_2_id)
        region_1_2_l2_nodes = {
            key: torch.unique(torch.cat((
                region_1_l2_nodes.get(key, torch.tensor([])),
                region_2_l2_nodes.get(key, torch.tensor([]))
            )))
            for key in set(region_1_l2_nodes) | set(region_2_l2_nodes)
        }
        region_pair_nodes = {**l0_nodes_dict, **l1_nodes_dict, **region_1_2_l2_nodes}
        # region_pair_sub_graph = dgl.node_subgraph(self.hkg_list.urban_graph_list[city_id], region_pair_nodes)
        # return region_pair_sub_graph, region_pair_nodes
        region_pair_nodes_tuple = (region_pair_nodes, region_1_l2_nodes, region_2_l2_nodes)
        return region_pair_nodes_tuple

    def get_l0_and_l1_nodes(self, city_id):
        l0_nodes_dict = {
            'schema_region': self.hkg_list.urban_graph_list[city_id].nodes('schema_region'),
            'schema_brand': self.hkg_list.urban_graph_list[city_id].nodes('schema_brand'),
            'schema_cate1': self.hkg_list.urban_graph_list[city_id].nodes('schema_cate1'),
            'schema_junc_cate': self.hkg_list.urban_graph_list[city_id].nodes('schema_junc_cate'),
            'schema_road_cate': self.hkg_list.urban_graph_list[city_id].nodes('schema_road_cate'),
            'schema_poi': self.hkg_list.urban_graph_list[city_id].nodes('schema_poi'),
            'schema_road': self.hkg_list.urban_graph_list[city_id].nodes('schema_road'),
            'schema_junction': self.hkg_list.urban_graph_list[city_id].nodes('schema_junction'),
            'schema_streetview': self.hkg_list.urban_graph_list[city_id].nodes('schema_streetview')
        }
        l1_nodes_dict = {
            'l1_brand': self.hkg_list.brand_graph_list[city_id].nodes('l1_brand'),
            'l1_cate1': self.hkg_list.cate1_graph_list[city_id].nodes('l1_cate1'),
            'l1_junccate': self.hkg_list.junccate_graph_list[city_id].nodes('l1_junccate'),
            'l1_roadcate': self.hkg_list.roadcate_graph_list[city_id].nodes('l1_roadcate')
        }
        return l0_nodes_dict, l1_nodes_dict

    def get_l2_nodes(self, city_id, region_id):
        all_edge_types = self.hkg_list.urban_graph_list[city_id].etypes
        fanout = {}
        for etype in all_edge_types:
            if etype in ['HasPoi', 'HasRoad', 'HasJunc']:
                fanout[etype] = -1
            else:
                fanout[etype] = 0
        sub_graph = dgl.sampling.sample_neighbors(self.hkg_list.urban_graph_list[city_id],
                                                  {'region': torch.tensor([region_id], dtype=torch.int64).to(self.graph_device)}
                                                  , fanout, edge_dir='out', copy_ndata=True, copy_edata=True)
        sub_poi_nodes = sub_graph.edges(etype='HasPoi')[1]
        sub_road_nodes = sub_graph.edges(etype='HasRoad')[1]
        sub_junc_nodes = sub_graph.edges(etype='HasJunc')[1]
        for etype in all_edge_types:
            fanout[etype] = 0
        fanout['BrandOf'] = -1
        poi_brand_graph = dgl.sampling.sample_neighbors(self.hkg_list.urban_graph_list[city_id], {'poi': sub_poi_nodes}, fanout,
                                                        edge_dir='out', copy_ndata=True, copy_edata=True)

        for etype in all_edge_types:
            fanout[etype] = 0
        fanout['Cate1Of'] = -1
        poi_cate1_graph = dgl.sampling.sample_neighbors(self.hkg_list.urban_graph_list[city_id], {'poi': sub_poi_nodes}, fanout,
                                                        edge_dir='out', copy_ndata=True, copy_edata=True)

        for etype in all_edge_types:
            fanout[etype] = 0
        fanout['JCateOf'] = -1
        junc_cate_graph = dgl.sampling.sample_neighbors(self.hkg_list.urban_graph_list[city_id], {'junction': sub_junc_nodes}, fanout,
                                                        edge_dir='out', copy_ndata=True, copy_edata=True)

        for etype in all_edge_types:
            fanout[etype] = 0
        fanout['RCateOf'] = -1
        road_cate_graph = dgl.sampling.sample_neighbors(self.hkg_list.urban_graph_list[city_id], {'road': sub_road_nodes}, fanout,
                                                        edge_dir='out', copy_ndata=True, copy_edata=True)

        sub_brand_nodes = poi_brand_graph.edges(etype='BrandOf')[1]
        sub_cate1_nodes = poi_cate1_graph.edges(etype='Cate1Of')[1]
        sub_junc_cate_nodes = junc_cate_graph.edges(etype='JCateOf')[1]
        sub_road_cate_nodes = road_cate_graph.edges(etype='RCateOf')[1]
        # region_nodes = torch.cat((torch.tensor([region_id], dtype=torch.int64).to(self.graph_device), sub_region_nodes), dim=0)
        region_nodes = torch.tensor([region_id], dtype=torch.int64).to(self.graph_device)

        l2_nodes_dict = {
            'region': region_nodes,
            'poi': sub_poi_nodes,
            'road': sub_road_nodes,
            'junction': sub_junc_nodes,
            'brand': sub_brand_nodes,
            'cate1': sub_cate1_nodes,
            'junc_cate': sub_junc_cate_nodes,
            'road_cate': sub_road_cate_nodes
        }
        return l2_nodes_dict

    def get_inter_all_region_pair_sub(self):
        city_num = len(self.hkg_list.urban_graph_list)
        city_region_num = [self.hkg_list.urban_graph_list[i].number_of_nodes('region') for i in range(city_num)]
        # inter_all_region_pair_sub_list [city_num][city_num][region_num][region_num]
        region_pair_sub_node_list = [[[[None for l in range(city_region_num[j])] for k in range(city_region_num[i])] for j in range(city_num)] for i
                                     in range(city_num)]
        print('Inter All Region Pair Sub Graph Constructing')
        for i in tqdm(range(city_num)):
            for j in range(i + 1, city_num):
                # for j in range(city_num):
                for k in range(city_region_num[i]):
                    for l in range(city_region_num[j]):
                        region_pair_sub_node = self.get_inter_region_sub(i, j, k, l)
                        region_pair_sub_node_list[i][j][k][l] = region_pair_sub_node
        print('Inter All Region Pair Sub Graph Constructed')
        return region_pair_sub_node_list

    def get_inter_region_sub(self, city_1_id, city_2_id, region_1_id, region_2_id):
        city_1_l0_nodes_dict, city_1_l1_nodes_dict = self.get_l0_and_l1_nodes(city_1_id)
        city_2_l0_nodes_dict, city_2_l1_nodes_dict = self.get_l0_and_l1_nodes(city_2_id)
        region_1_l2_nodes = self.get_l2_nodes(city_1_id, region_1_id)
        region_2_l2_nodes = self.get_l2_nodes(city_2_id, region_2_id)
        region_pair_nodes = ({**city_1_l0_nodes_dict, **city_1_l1_nodes_dict, **region_1_l2_nodes},
                             {**city_2_l0_nodes_dict, **city_2_l1_nodes_dict, **region_2_l2_nodes})
        return region_pair_nodes
