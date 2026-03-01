import os.path
import random

import numpy as np
import torch
import dgl
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pickle as pkl


class HeteroGraphData(object):
    def __init__(self, kg_dir, reverse=False, export=False, graph_device='cpu', spatial=None):
        self.kg_dir = kg_dir
        self.reverse = reverse
        self.ents, self.ent_range, self.rels, self.ent2id, self.rel2id, self.kg_data = self.load_kg(reverse=reverse)
        self.num_ent = len(self.ent2id)
        self.mht_region_ents = self.ents[self.ent_range['region'][0]:self.ent_range['region'][1]]
        self.num_mht_region_ent = len(self.mht_region_ents)
        self.graph_device = graph_device
        self.spatial = spatial

        if reverse:
            self.num_rel = len(self.rel2id)
        else:
            self.num_rel = len(self.rel2id)

        self.id2ent = {v: k for k, v in self.ent2id.items()}
        self.id2rel = {v: k for k, v in self.rel2id.items()}

        if export:
            print('exporting ent2id.txt and rel2id.txt...')
            file_dir = os.path.dirname(kg_dir)
            if reverse:
                ent2id_file = 'ent2id_reverse.txt'
                rel2id_file = 'rel2id_reverse.txt'
            else:
                ent2id_file = 'ent2id.txt'
                rel2id_file = 'rel2id.txt'
            if not os.path.exists(os.path.join(file_dir, ent2id_file)):
                with open(os.path.join(file_dir, ent2id_file), 'w') as f:
                    for k, v in self.ent2id.items():
                        f.write(k + '\t' + str(v) + '\n')
            else:
                print('ent2id.txt already exists.')
            if not os.path.exists(os.path.join(file_dir, rel2id_file)):
                with open(os.path.join(file_dir, rel2id_file), 'w') as f:
                    for k, v in self.rel2id.items():
                        f.write(k + '\t' + str(v) + '\n')
            else:
                print('rel2id.txt already exists.')

        src = [x[0] for x in self.kg_data]
        dst = [x[2] for x in self.kg_data]
        rels = [x[1] for x in self.kg_data]

        self.g = dgl.graph((src, dst), num_nodes=self.num_ent)
        self.g = self.g.to(graph_device)
        print('num_nodes:', self.g.num_nodes())
        print('homo graph constructed.')

        rel2pairs = {}
        for i, relid in enumerate(rels):
            if self.id2rel[relid] not in rel2pairs.keys():
                rel2pairs[self.id2rel[relid]] = []
            rel2pairs[self.id2rel[relid]].append((src[i], dst[i]))

        rel2pairs_od = {}
        for rel, pairs in rel2pairs.items():
            src = [x[0] for x in pairs]
            dst = [x[1] for x in pairs]
            rel2pairs_od[rel] = (src, dst)

        self.rel2pairs_od = rel2pairs_od

        self.hg = dgl.heterograph({
            ('region', 'NearBy', 'region'): rel2pairs_od['Region_Nearby'],
            ('region', 'HasJunc', 'junction'): (rel2pairs_od['Junction_RegionOf'][1],
                                                [x - self.ent_range['junction'][0] for x in rel2pairs_od['Junction_RegionOf'][0]]),
            ('junction', 'JCateOf', 'junc_cate'): ([x - self.ent_range['junction'][0] for x in rel2pairs_od['Junction_JCateOf'][0]],
                                                   [x - self.ent_range['junc_cate'][0] for x in rel2pairs_od['Junction_JCateOf'][1]]),
            ('region', 'HasRoad', 'road'): (rel2pairs_od['Road_RegionOf'][1],
                                            [x - self.ent_range['road'][0] for x in rel2pairs_od['Road_RegionOf'][0]]),
            ('road', 'RCateOf', 'road_cate'): ([x - self.ent_range['road'][0] for x in rel2pairs_od['Road_RCateOf'][0]],
                                               [x - self.ent_range['road_cate'][0] for x in rel2pairs_od['Road_RCateOf'][1]]),
            ('region', 'HasPoi', 'poi'): (rel2pairs_od['POI_RegionOf'][1],
                                          [x - self.ent_range['poi'][0] for x in rel2pairs_od['POI_RegionOf'][0]]),
            ('poi', 'BrandOf', 'brand'): ([x - self.ent_range['poi'][0] for x in rel2pairs_od['POI_BrandOf'][0]],
                                          [x - self.ent_range['brand'][0] for x in rel2pairs_od['POI_BrandOf'][1]]),
            ('poi', 'Cate1Of', 'cate1'): ([x - self.ent_range['poi'][0] for x in rel2pairs_od['POI_Cate1Of'][0]],
                                          [x - self.ent_range['cate1'][0] for x in rel2pairs_od['POI_Cate1Of'][1]]),
        })

        self.hg = self.hg.to(graph_device)

        if self.spatial is not None:
            self.load_spatial(self.spatial)

        print('hetero graph constructed.')
        print(self.hg)

    def load_spatial(self, spatial):
        region_coo_file = spatial[0]
        region_coo = []
        with open(region_coo_file, 'r') as f:
            f.readline()
            for line in f.readlines():
                x = line.strip().split(',')
                region_coo.append([float(x[1]), float(x[2])])
        region_coo = [torch.tensor(x, dtype=torch.float32) for x in region_coo]
        self.hg.nodes['region'].data['coo'] = torch.stack(region_coo).to(self.graph_device)

        poi_coo_file = spatial[1]
        poi_coo = {}
        with open(poi_coo_file, 'r') as f:
            f.readline()
            for line in f.readlines():
                x = line.strip().split(',')
                poi_coo[x[0]] = [float(x[1]), float(x[2])]
        poi_ent_range = self.ent_range['poi']
        poi_keys = [self.id2ent[i] for i in range(poi_ent_range[0], poi_ent_range[1])]
        poi_coo_list = [poi_coo[x] for x in poi_keys]
        poi_coo_list = [torch.tensor(x, dtype=torch.float32) for x in poi_coo_list]
        self.hg.nodes['poi'].data['coo'] = torch.stack(poi_coo_list).to(self.graph_device)

    def load_kg(self, reverse=False):
        facts_str = []
        print('loading knowledge graph...')
        with open(self.kg_dir, 'r') as f:
            f.readline()
            for line in tqdm(f.readlines()):
                x = line.strip().split(',')
                facts_str.append([x[0], x[1], x[2]])

        origin_rels = sorted(list(set([x[1] for x in facts_str])))
        if reverse:
            all_rels = sorted(origin_rels + [x + '_rev' for x in origin_rels])
        else:
            all_rels = sorted(origin_rels)

        all_ents = sorted(list(set([x[0] for x in facts_str] + [x[2] for x in facts_str])))

        mht_region_ents = [x for x in all_ents if x[:4] == 'mhtr' or x[:4] == 'chir' or x[:3] == 'szr']
        mht_region_ents = sorted(mht_region_ents, key=lambda y: int(y[4:]))

        poi_ents = [x[0] for x in facts_str if x[1] == 'POI_Cate1Of']
        poi_ents += [x[0] for x in facts_str if x[1] == 'POI_BrandOf']
        poi_ents += [x[0] for x in facts_str if x[1] == 'POI_RegionOf']
        poi_ents = sorted(list(set(poi_ents)))
        road_ents = [x[0] for x in facts_str if x[1] == 'Road_RCateOf']
        road_ents += [x[0] for x in facts_str if x[1] == 'Road_RegionOf']
        road_ents = sorted(list(set(road_ents)))
        junc_ents = [x[0] for x in facts_str if x[1] == 'Junction_JCateOf']
        junc_ents += [x[0] for x in facts_str if x[1] == 'Junction_RegionOf']
        junc_ents = sorted(list(set(junc_ents)))
        cate1_ents = [x[2] for x in facts_str if x[1] == 'POI_Cate1Of']
        cate1_ents = sorted(list(set(cate1_ents)))
        brand_ents = [x[2] for x in facts_str if x[1] == 'POI_BrandOf']
        brand_ents = sorted(list(set(brand_ents)))
        road_cate_ents = [x[2] for x in facts_str if x[1] == 'Road_RCateOf']
        road_cate_ents = sorted(list(set(road_cate_ents)))
        junc_cate_ents = [x[2] for x in facts_str if x[1] == 'Junction_JCateOf']
        junc_cate_ents = sorted(list(set(junc_cate_ents)))
        streetview_ents = [x[2] for x in facts_str if x[1] == 'Region_StreetViewOf']
        streetview_ents = sorted(list(set(streetview_ents)))

        region_ent2id = dict([(x, i) for i, x in enumerate(mht_region_ents)])
        poi_ent2id = dict([(x, i) for i, x in enumerate(poi_ents)])
        road_ent2id = dict([(x, i) for i, x in enumerate(road_ents)])
        junc_ent2id = dict([(x, i) for i, x in enumerate(junc_ents)])
        cate1_ent2id = dict([(x, i) for i, x in enumerate(cate1_ents)])
        brand_ent2id = dict([(x, i) for i, x in enumerate(brand_ents)])
        road_cate_ent2id = dict([(x, i) for i, x in enumerate(road_cate_ents)])
        junc_cate_ent2id = dict([(x, i) for i, x in enumerate(junc_cate_ents)])
        streetview_ent2id = dict([(x, i) for i, x in enumerate(streetview_ents)])
        ent2id_dicts = {'region': region_ent2id, 'poi': poi_ent2id, 'road': road_ent2id, 'junction': junc_ent2id,
                        'cate1': cate1_ent2id, 'brand': brand_ent2id, 'road_cate': road_cate_ent2id,
                        'junc_cate': junc_cate_ent2id, 'streetview': streetview_ent2id}

        ents = mht_region_ents + poi_ents + road_ents + junc_ents + cate1_ents + brand_ents + road_cate_ents + junc_cate_ents + streetview_ents
        ent_range = {}
        start = 0
        for k, v in ent2id_dicts.items():
            ent_range[k] = (start, start + len(v))
            start += len(v)

        ent2id = dict([(x, i) for i, x in enumerate(ents)])
        rel2id = dict([(x, i) for i, x in enumerate(all_rels)])

        if reverse:
            kg_data = ([[ent2id[x[0]], rel2id[x[1]], ent2id[x[2]]] for x in facts_str] +
                       [[ent2id[x[2]], rel2id[x[1] + '_rev'], ent2id[x[0]]] for x in facts_str])
        else:
            kg_data = [[ent2id[x[0]], rel2id[x[1]], ent2id[x[2]]] for x in facts_str]

        print("ent_range: ")
        print(ent_range)
        return ents, ent_range, all_rels, ent2id, rel2id, kg_data

    def init_hetero_graph_features(self, image=None, flow=None, feature_dir=None, node_feats_dim=10,
                                   edge_feats_dim=24, device='cpu'):
        if feature_dir is None:
            # node features
            print('feature_dir is None, using random initialization...')
            region_feats = torch.randn(self.num_mht_region_ent, node_feats_dim).type(torch.float32).to(device)
            junction_feats = torch.randn(self.hg.num_nodes('junction'), node_feats_dim).type(torch.float32).to(device)
            road_feats = torch.randn(self.hg.num_nodes('road'), node_feats_dim).type(torch.float32).to(device)
            poi_feats = torch.randn(self.hg.num_nodes('poi'), node_feats_dim).type(torch.float32).to(device)
            brand_feats = torch.randn(self.hg.num_nodes('brand'), node_feats_dim).type(torch.float32).to(device)
            cate1_feats = torch.randn(self.hg.num_nodes('cate1'), node_feats_dim).type(torch.float32).to(device)
            junc_cate_feats = torch.randn(self.hg.num_nodes('junc_cate'), node_feats_dim).type(torch.float32).to(device)
            road_cate_feats = torch.randn(self.hg.num_nodes('road_cate'), node_feats_dim).type(torch.float32).to(device)

            # edge features
            NearBy_feats = torch.randn(len(self.rel2pairs_od['Region_Nearby'][0]), edge_feats_dim).type(torch.float32).to(device)
            HasJunc_feats = torch.randn(len(self.rel2pairs_od['Junction_RegionOf'][0]), edge_feats_dim).type(torch.float32).to(device)
            JCateOf_feats = torch.randn(len(self.rel2pairs_od['Junction_JCateOf'][0]), edge_feats_dim).type(torch.float32).to(device)
            HasRoad_feats = torch.randn(len(self.rel2pairs_od['Road_RegionOf'][0]), edge_feats_dim).type(torch.float32).to(device)
            RCateOf_feats = torch.randn(len(self.rel2pairs_od['Road_RCateOf'][0]), edge_feats_dim).type(torch.float32).to(device)
            HasPoi_feats = torch.randn(len(self.rel2pairs_od['POI_RegionOf'][0]), edge_feats_dim).type(torch.float32).to(device)
            BrandOf_feats = torch.randn(len(self.rel2pairs_od['POI_BrandOf'][0]), edge_feats_dim).type(torch.float32).to(device)
            Cate1Of_feats = torch.randn(len(self.rel2pairs_od['POI_Cate1Of'][0]), edge_feats_dim).type(torch.float32).to(device)

        else:
            ent_range = self.ent_range
            # region_feats
            region_id2embedding = pkl.load(open(feature_dir[5], 'rb'))
            region_feats = torch.tensor([region_id2embedding[self.id2ent[i]] for i in range(ent_range['region'][0], ent_range['region'][1])], dtype=torch.float32).to(device)
            # print('region_LLM_feats loaded.')
            junction_id2embedding = pkl.load(open(feature_dir[3], 'rb'))
            junction_feats = torch.tensor([junction_id2embedding[self.id2ent[i]] for i in range(ent_range['junction'][0], ent_range['junction'][1])], dtype=torch.float32).to(device)
            # print('junction_LLM_feats loaded.')
            road_id2embedding = pkl.load(open(feature_dir[6], 'rb'))
            road_feats = torch.tensor([road_id2embedding[self.id2ent[i]] for i in range(ent_range['road'][0], ent_range['road'][1])], dtype=torch.float32).to(device)
            # print('road_LLM_feats loaded.')
            poi_id2embedding = pkl.load(open(feature_dir[4], 'rb'))
            poi_feats = torch.tensor([poi_id2embedding[self.id2ent[i]] for i in range(ent_range['poi'][0], ent_range['poi'][1])], dtype=torch.float32).to(device)
            # print('poi_LLM_feats loaded.')
            brand_id2embedding = pkl.load(open(feature_dir[0], 'rb'))
            brand_feats = torch.tensor([brand_id2embedding[self.id2ent[i]] for i in range(ent_range['brand'][0], ent_range['brand'][1])], dtype=torch.float32).to(device)
            # print('brand_LLM_feats loaded.')
            cate1_id2embedding = pkl.load(open(feature_dir[1], 'rb'))
            cate1_feats = torch.tensor([cate1_id2embedding[int(self.id2ent[i])] for i in range(ent_range['cate1'][0], ent_range['cate1'][1])], dtype=torch.float32).to(device)
            # print('cate1_LLM_feats loaded.')
            junc_cate_id2embedding = pkl.load(open(feature_dir[2], 'rb'))
            junc_cate_feats = torch.tensor([junc_cate_id2embedding[self.id2ent[i]] for i in range(ent_range['junc_cate'][0], ent_range['junc_cate'][1])], dtype=torch.float32).to(device)
            # print('junc_cate_LLM_feats loaded.')
            road_cate_id2embedding = pkl.load(open(feature_dir[7], 'rb'))
            road_cate_feats = torch.tensor([road_cate_id2embedding[self.id2ent[i]] for i in range(ent_range['road_cate'][0], ent_range['road_cate'][1])], dtype=torch.float32).to(device)
            # print('road_cate_LLM_feats loaded.')


            # rels_emb = np.load(feature_dir[1])
            NearBy_feats = torch.randn(len(self.rel2pairs_od['Region_Nearby'][0]), edge_feats_dim).type(torch.float32).to(device)
            HasJunc_feats = torch.randn(len(self.rel2pairs_od['Junction_RegionOf'][0]), edge_feats_dim).type(torch.float32).to(device)
            JCateOf_feats = torch.randn(len(self.rel2pairs_od['Junction_JCateOf'][0]), edge_feats_dim).type(torch.float32).to(device)
            HasRoad_feats = torch.randn(len(self.rel2pairs_od['Road_RegionOf'][0]), edge_feats_dim).type(torch.float32).to(device)
            RCateOf_feats = torch.randn(len(self.rel2pairs_od['Road_RCateOf'][0]), edge_feats_dim).type(torch.float32).to(device)
            HasPoi_feats = torch.randn(len(self.rel2pairs_od['POI_RegionOf'][0]), edge_feats_dim).type(torch.float32).to(device)
            BrandOf_feats = torch.randn(len(self.rel2pairs_od['POI_BrandOf'][0]), edge_feats_dim).type(torch.float32).to(device)
            Cate1Of_feats = torch.randn(len(self.rel2pairs_od['POI_Cate1Of'][0]), edge_feats_dim).type(torch.float32).to(device)

        self.hg.nodes['region'].data['f'] = region_feats
        self.hg.nodes['junction'].data['f'] = junction_feats
        self.hg.nodes['road'].data['f'] = road_feats
        self.hg.nodes['poi'].data['f'] = poi_feats
        self.hg.nodes['brand'].data['f'] = brand_feats
        self.hg.nodes['cate1'].data['f'] = cate1_feats
        self.hg.nodes['junc_cate'].data['f'] = junc_cate_feats
        self.hg.nodes['road_cate'].data['f'] = road_cate_feats
        # self.hg.nodes['streetview'].data['f'] = streetview_feats

        self.hg.edges['NearBy'].data['f'] = NearBy_feats
        self.hg.edges['HasJunc'].data['f'] = HasJunc_feats
        self.hg.edges['JCateOf'].data['f'] = JCateOf_feats
        self.hg.edges['HasRoad'].data['f'] = HasRoad_feats
        self.hg.edges['RCateOf'].data['f'] = RCateOf_feats
        self.hg.edges['HasPoi'].data['f'] = HasPoi_feats
        self.hg.edges['BrandOf'].data['f'] = BrandOf_feats
        self.hg.edges['Cate1Of'].data['f'] = Cate1Of_feats


        if image is not None:
            # add in region attribute
            si_img_feats = np.load(image)
            si_img_feats = torch.tensor(si_img_feats, dtype=torch.float32).to(device)
            self.hg.nodes['region'].data['si_img'] = si_img_feats
            print('image features loaded.')

        if flow is not None:
            in_flow_feat = np.load(flow[0]) # [180, 24]
            out_flow_feat = np.load(flow[1])
            scaler = StandardScaler()
            in_flow_scaled = scaler.fit_transform(in_flow_feat)
            out_flow_scaled = scaler.fit_transform(out_flow_feat)
            in_flow = torch.tensor(in_flow_scaled, dtype=torch.float32).to(device)
            out_flow = torch.tensor(out_flow_scaled, dtype=torch.float32).to(device)
            if flow[2] == 'aug':
                in_flow = torch.cat([in_flow for _ in range(6)], dim=1)
                out_flow = torch.cat([out_flow for _ in range(6)], dim=1)
            self.hg.nodes['region'].data['inflow'] = in_flow
            self.hg.nodes['region'].data['outflow'] = out_flow
            print('flow features loaded.')

        return self.hg

    def get_region_nearby_sub_and_neighbor(self):
        region_nearby_sub_g = dgl.edge_type_subgraph(self.hg, [('region', 'NearBy', 'region')])
        nearby_neighbors_list = []
        for i in range(self.num_mht_region_ent):
            nearby_neighbors_list.append(region_nearby_sub_g.successors(i, etype='NearBy'))
        return region_nearby_sub_g, nearby_neighbors_list

    def get_region_flow_sub(self):
        region_flow_sub_g = dgl.edge_type_subgraph(self.hg, [('region', 'flowFrom', 'region'), ('region', 'flowTo', 'region')])
        region_flow_to_sub = dgl.edge_type_subgraph(self.hg, [('region', 'flowTo', 'region')])
        region_flow_from_sub = dgl.edge_type_subgraph(self.hg, [('region', 'flowFrom', 'region')])
        return region_flow_sub_g, region_flow_to_sub, region_flow_from_sub

    def get_region_sub_all(self, sample_method='metagraph', args=None):

        region_subgraphs = {}
        region_sub_node_dicts = {}

        if sample_method == 'metagraph':
            for i in range(self.num_mht_region_ent):
                fanout = {'JCateOf':0, 'BrandOf':0, 'Cate1Of':0, 'HasJunc':-1, 'HasPoi':-1, 'HasRoad':-1, 'NearBy':-1,
                          'RCateOf':0}
                sub_graph = dgl.sampling.sample_neighbors(self.hg, {'region': i}, fanout,
                                                          edge_dir='out', copy_ndata=True, copy_edata=True)

                sub_poi_nodes = sub_graph.edges(etype='HasPoi')[1]
                sub_road_nodes = sub_graph.edges(etype='HasRoad')[1]
                sub_junc_nodes = sub_graph.edges(etype='HasJunc')[1]
                sub_region_nodes = sub_graph.edges(etype='NearBy')[1]


                poi_brand_graph = dgl.sampling.sample_neighbors(self.hg, {'poi': sub_poi_nodes},
                                                                {'JCateOf':0, 'BrandOf':-1, 'Cate1Of':0, 'HasJunc':0,
                                                                 'HasPoi':0, 'HasRoad':0, 'NearBy':0, 'RCateOf':0},
                                                                edge_dir='out', copy_ndata=True, copy_edata=True)
                poi_cate1_graph = dgl.sampling.sample_neighbors(self.hg, {'poi': sub_poi_nodes},
                                                                {'JCateOf':0, 'BrandOf':0, 'Cate1Of':-1, 'HasJunc':0,
                                                                 'HasPoi':0, 'HasRoad':0, 'NearBy':0, 'RCateOf':0},
                                                                edge_dir='out', copy_ndata=True, copy_edata=True)
                junc_cate_graph = dgl.sampling.sample_neighbors(self.hg, {'junction': sub_junc_nodes},
                                                                {'JCateOf':-1, 'BrandOf':0, 'Cate1Of':0, 'HasJunc':0,
                                                                 'HasPoi':0, 'HasRoad':0, 'NearBy':0, 'RCateOf':0},
                                                                edge_dir='out', copy_ndata=True, copy_edata=True)
                road_cate_graph = dgl.sampling.sample_neighbors(self.hg, {'road': sub_road_nodes},
                                                                {'JCateOf':0, 'BrandOf':0, 'Cate1Of':0, 'HasJunc':0,
                                                                 'HasPoi':0, 'HasRoad':0, 'NearBy':0, 'RCateOf':-1},
                                                                edge_dir='out', copy_ndata=True, copy_edata=True)

                sub_brand_nodes = poi_brand_graph.edges(etype='BrandOf')[1]
                sub_cate1_nodes = poi_cate1_graph.edges(etype='Cate1Of')[1]
                sub_junc_cate_nodes = junc_cate_graph.edges(etype='JCateOf')[1]
                sub_road_cate_nodes = road_cate_graph.edges(etype='RCateOf')[1]
                region_nodes = torch.cat((torch.tensor([i], dtype=torch.int64).to(self.graph_device), sub_region_nodes), dim=0)

                node_dicts = {'region': region_nodes, 'poi': sub_poi_nodes, 'road': sub_road_nodes, 'junction': sub_junc_nodes,
                              'brand': sub_brand_nodes, 'cate1': sub_cate1_nodes, 'junc_cate': sub_junc_cate_nodes,
                              'road_cate': sub_road_cate_nodes}

                region_sub_node_dicts[i] = node_dicts
                region_subgraphs[i] = dgl.node_subgraph(self.hg, node_dicts)

        elif sample_method == 'random_walk':
            for i in range(self.num_mht_region_ent):
                meta_path_junc = ['HasJunc', 'JCateOf']
                meta_path_poi_cat = ['HasPoi', 'Cate1Of']
                meta_path_poi_brand = ['HasPoi', 'BrandOf']
                meta_path_road = ['HasRoad', 'RCateOf']
                junc_trace, _ = dgl.sampling.random_walk(self.hg, [i], metapath=meta_path_junc,
                                                          restart_prob=args['restart_prob']['junc'])
                poi_cat_trace, _ = dgl.sampling.random_walk(self.hg, [i], metapath=meta_path_poi_cat,
                                                             restart_prob=args['restart_prob']['poi_cat'])
                poi_brand_trace, _ = dgl.sampling.random_walk(self.hg, [i], metapath=meta_path_poi_brand,
                                                               restart_prob=args['restart_prob']['poi_brand'])
                road_trace, _ = dgl.sampling.random_walk(self.hg, [i], metapath=meta_path_road,
                                                          restart_prob=args['restart_prob']['road'])

        print('region subgraphs and node_dicts constructed.')

        return region_subgraphs, region_sub_node_dicts

    def get_region_sub_test_all(self):
        region_subgraphs = {}
        region_sub_node_dicts = {}
        for i in range(self.num_mht_region_ent):
            fanout = {'JCateOf': 0, 'BrandOf': -1, 'Cate1Of': -1, 'HasJunc': -1, 'HasPoi': -1, 'HasRoad': -1,
                      'NearBy': -1, 'RCateOf': 0}
            sub_graph = dgl.sampling.sample_neighbors(self.hg, {'region': i}, fanout,
                                                      edge_dir='out', copy_ndata=True, copy_edata=True)

            sub_poi_nodes = sub_graph.edges(etype='HasPoi')[1]
            sub_road_nodes = sub_graph.edges(etype='HasRoad')[1]
            sub_junc_nodes = sub_graph.edges(etype='HasJunc')[1]
            sub_region_nodes = sub_graph.edges(etype='NearBy')[1]

            sub_poi_nodes = random.sample(sub_poi_nodes.tolist(), int(len(sub_poi_nodes) * 0.6))

            poi_brand_graph = dgl.sampling.sample_neighbors(self.hg, {'poi': sub_poi_nodes},
                                                            {'JCateOf': 0, 'BrandOf': -1, 'Cate1Of': 0, 'HasJunc': 0,
                                                             'HasPoi': 0, 'HasRoad': 0, 'NearBy': 0, 'RCateOf': 0},
                                                            edge_dir='out', copy_ndata=True, copy_edata=True)
            poi_cate1_graph = dgl.sampling.sample_neighbors(self.hg, {'poi': sub_poi_nodes},
                                                            {'JCateOf': 0, 'BrandOf': 0, 'Cate1Of': -1, 'HasJunc': 0,
                                                             'HasPoi': 0, 'HasRoad': 0, 'NearBy': 0, 'RCateOf': 0},
                                                            edge_dir='out', copy_ndata=True, copy_edata=True)
            junc_cate_graph = dgl.sampling.sample_neighbors(self.hg, {'junction': sub_junc_nodes},
                                                            {'JCateOf': 0, 'BrandOf': 0, 'Cate1Of': 0, 'HasJunc': 0,
                                                             'HasPoi': 0, 'HasRoad': 0, 'NearBy': 0, 'RCateOf': 0},
                                                            edge_dir='out', copy_ndata=True, copy_edata=True)
            road_cate_graph = dgl.sampling.sample_neighbors(self.hg, {'road': sub_road_nodes},
                                                            {'JCateOf': 0, 'BrandOf': 0, 'Cate1Of': 0, 'HasJunc': 0,
                                                             'HasPoi': 0, 'HasRoad': 0, 'NearBy': 0, 'RCateOf': 0},
                                                            edge_dir='out', copy_ndata=True, copy_edata=True)

            sub_brand_nodes = poi_brand_graph.edges(etype='BrandOf')[1]
            sub_cate1_nodes = poi_cate1_graph.edges(etype='Cate1Of')[1]
            sub_junc_cate_nodes = junc_cate_graph.edges(etype='JCateOf')[1]
            sub_road_cate_nodes = road_cate_graph.edges(etype='RCateOf')[1]
            region_nodes = torch.cat((torch.tensor([i], dtype=torch.int64).to(self.graph_device), sub_region_nodes),
                                     dim=0)

            node_dicts = {'region': region_nodes, 'poi': sub_poi_nodes, 'road': sub_road_nodes,
                          'junction': sub_junc_nodes, 'brand': sub_brand_nodes,
                          'cate1': sub_cate1_nodes,
                          'junc_cate': sub_junc_cate_nodes, 'road_cate': sub_road_cate_nodes}

            region_sub_node_dicts[i] = node_dicts
            region_subgraphs[i] = dgl.node_subgraph(self.hg, node_dicts)

        print('region subgraphs and node_dicts constructed.')

        return region_subgraphs, region_sub_node_dicts


    def get_region_sub_with_mask(self, rule_matrix, args):

        assert (rule_matrix.shape[0] == args['logits_num']), 'The shape of rule_logits_num is not correct.'
        assert (rule_matrix.shape[1] == self.hg.num_nodes('cate1') + self.hg.num_nodes('brand') +
                self.hg.num_nodes('junc_cate') + self.hg.num_nodes('road_cate')), 'The shape of type_num is not correct.'

        cate1_mask = rule_matrix[0, :self.hg.num_nodes('cate1')]
        brand_mask = rule_matrix[0, self.hg.num_nodes('cate1'):self.hg.num_nodes('cate1') + self.hg.num_nodes('brand')]
        jcate_mask = rule_matrix[0, self.hg.num_nodes('cate1') + self.hg.num_nodes('brand'):
                                  self.hg.num_nodes('cate1') + self.hg.num_nodes('brand') + self.hg.num_nodes(
                                      'junc_cate')]
        rcate_mask = rule_matrix[0,
                     self.hg.num_nodes('cate1') + self.hg.num_nodes('brand') + self.hg.num_nodes('junc_cate'):]

        cate1_idc = [x for x in range(len(cate1_mask)) if cate1_mask[x] == args['mask_selected']]
        brand_idc = [x for x in range(len(brand_mask)) if brand_mask[x] == args['mask_selected']]
        jcate_idc = [x for x in range(len(jcate_mask)) if jcate_mask[x] == args['mask_selected']]
        rcate_idc = [x for x in range(len(rcate_mask)) if rcate_mask[x] == args['mask_selected']]

        idc_subgraph = dgl.sampling.sample_neighbors(self.hg, {'cate1': cate1_idc, 'brand': brand_idc,
                                                               'junc_cate': jcate_idc, 'road_cate': rcate_idc},
                                                     {'Cate1Of': -1, 'BrandOf': -1, 'JCateOf': -1, 'RCateOf': -1,
                                                      'HasJunc': 0, 'HasPoi': 0, 'HasRoad': 0, 'NearBy': 0, 'StreetViewOf': 0},
                                                     edge_dir='in', copy_ndata=True, copy_edata=True)

        poi_cate_nodes = idc_subgraph.edges(etype='Cate1Of')[0]
        poi_brand_nodes = idc_subgraph.edges(etype='BrandOf')[0]
        poi_nodes = list(set(poi_cate_nodes).intersection(set(poi_brand_nodes)))
        junc_nodes = idc_subgraph.edges(etype='JCateOf')[0]
        road_nodes = idc_subgraph.edges(etype='RCateOf')[0]

        rule_node_dicts = {'region': [x for x in range(self.hg.num_nodes('region'))],
                           'poi': poi_nodes, 'road': road_nodes, 'junction': junc_nodes, 'brand': brand_idc,
                           'cate1': cate1_idc, 'junc_cate': jcate_idc, 'road_cate': rcate_idc}

        rule_subgraph = dgl.node_subgraph(self.hg, rule_node_dicts)

        idc = [x for x in range(len(rule_matrix[0, :])) if rule_matrix[0, x] == args['mask_selected']]
        opr = rule_matrix[1, :]
        value = rule_matrix[2, :]

        region_subgraphs = []
        region_sub_node_dicts = []

        if sum(opr) == 0:
            for i in range(self.num_mht_region_ent):
                # one-hop
                sub_graph = dgl.sampling.sample_neighbors(rule_subgraph, {'region': i}, -1, edge_dir='out',
                                                          copy_ndata=True, copy_edata=True)
                sub_poi_nodes = sub_graph.edges(etype='HasPoi')[1]
                sub_road_nodes = sub_graph.edges(etype='HasRoad')[1]
                sub_junc_nodes = sub_graph.edges(etype='HasJunc')[1]

                node_dicts = {'region': i, 'poi': sub_poi_nodes, 'road': sub_road_nodes, 'junction': sub_junc_nodes}
                # region_subgraphs[i] = dgl.node_subgraph(self.hg, node_dicts)
                region_subgraphs.append(dgl.node_subgraph(self.hg, node_dicts))
                # region_sub_node_dicts[i] = node_dicts
                region_sub_node_dicts.append(node_dicts)
            return region_subgraphs, region_sub_node_dicts

        else: 
            for i in range(self.num_mht_region_ent):
                # cate1 as center
                for cate_idc in cate1_idc:
                    sub_graph = dgl.sampling.sample_neighbors(rule_subgraph, {'cate1': cate_idc},
                                                              {'Cate1Of': value[cate_idc] if opr[cate_idc] == 1 else -1,
                                                               'BrandOf': 0, 'JCateOf': 0, 'RCateOf': 0, 'HasJunc': 0,
                                                               'HasPoi': 0, 'HasRoad': 0, 'NearBy': 0, 'StreetViewOf': 0},
                                                              edge_dir='in', copy_ndata=True, copy_edata=True)

                    sub_poi_nodes = sub_graph.edges(etype='HasPoi')[1]
                    sub_road_nodes = sub_graph.edges(etype='HasRoad')[1]
                    sub_junc_nodes = sub_graph.edges(etype='HasJunc')[1]
                    node_dicts = {'region': i, 'poi': sub_poi_nodes, 'road': sub_road_nodes, 'junction': sub_junc_nodes}
                    region_subgraphs[i] = dgl.node_subgraph(self.hg, node_dicts)
                    region_sub_node_dicts[i] = node_dicts

                # first order
                sub_graph = dgl.sampling.sample_neighbors(rule_subgraph, {'region': i}, -1, edge_dir='out',
                                                          copy_ndata=True, copy_edata=True)
                sub_poi_nodes = sub_graph.edges(etype='HasPoi')[1]
                sub_road_nodes = sub_graph.edges(etype='HasRoad')[1]
                sub_junc_nodes = sub_graph.edges(etype='HasJunc')[1]
                # second order
                poi_brand_graph = dgl.sampling.sample_neighbors(rule_subgraph, {'poi': sub_poi_nodes},
                                                                {'JCateOf':0, 'BrandOf':-1, 'Cate1Of':0, 'HasJunc':0,
                                                                 'HasPoi':0, 'HasRoad':0, 'NearBy':0, 'RCateOf':0, 'StreetViewOf':0},
                                                                edge_dir='out', copy_ndata=True, copy_edata=True)
                poi_cate1_graph = dgl.sampling.sample_neighbors(rule_subgraph, {'poi': sub_poi_nodes},
                                                                {'JCateOf':0, 'BrandOf':0, 'Cate1Of':-1, 'HasJunc':0,
                                                                 'HasPoi':0, 'HasRoad':0, 'NearBy':0, 'RCateOf':0, 'StreetViewOf':0},
                                                                edge_dir='out', copy_ndata=True, copy_edata=True)
                junc_cate_graph = dgl.sampling.sample_neighbors(rule_subgraph, {'junction': sub_junc_nodes},
                                                                {'JCateOf':-1, 'BrandOf':0, 'Cate1Of':0, 'HasJunc':0,
                                                                 'HasPoi':0, 'HasRoad':0, 'NearBy':0, 'RCateOf':0, 'StreetViewOf':0},
                                                                edge_dir='out', copy_ndata=True, copy_edata=True)
                road_cate_graph = dgl.sampling.sample_neighbors(rule_subgraph, {'road': sub_road_nodes},
                                                                {'JCateOf':0, 'BrandOf':0, 'Cate1Of':0, 'HasJunc':0,
                                                                 'HasPoi':0, 'HasRoad':0, 'NearBy':0, 'RCateOf':-1, 'StreetViewOf':0},
                                                                edge_dir='out', copy_ndata=True, copy_edata=True)

                sub_brand_nodes = poi_brand_graph.edges(etype='BrandOf')[1]
                sub_cate1_nodes = poi_cate1_graph.edges(etype='Cate1Of')[1]
                sub_junc_cate_nodes = junc_cate_graph.edges(etype='JCateOf')[1]
                sub_road_cate_nodes = road_cate_graph.edges(etype='RCateOf')[1]

                opr_idc = [x for x in range(len(opr)) if opr[x] != 0]
                for j in opr_idc:
                    if opr[j] == 1:
                        sub_graph.add_edges(sub_graph.edges(etype=self.hg.etypes[j])[0], sub_graph.edges(etype=self.hg.etypes[j])[1])
                    elif opr[j] == 2:
                        sub_graph.remove_edges(sub_graph.edges(etype=self.hg.etypes[j])[0], sub_graph.edges(etype=self.hg.etypes[j])[1])

    def get_region_sub_with_rule(self, ent_id, max_node_size=None):

        fanout = {'JCateOf': 0, 'BrandOf': 0, 'Cate1Of': 0, 'HasJunc': -1, 'HasPoi': -1, 'HasRoad': -1,
                  'NearBy': -1, 'RCateOf': 0}
        sub_graph = dgl.sampling.sample_neighbors(self.hg, {'region': ent_id}, fanout,
                                                  edge_dir='out', copy_ndata=True, copy_edata=True)

        sub_poi_nodes = sub_graph.edges(etype='HasPoi')[1]
        sub_road_nodes = sub_graph.edges(etype='HasRoad')[1]
        sub_junc_nodes = sub_graph.edges(etype='HasJunc')[1]
        sub_region_nodes = sub_graph.edges(etype='NearBy')[1]

        poi_brand_graph = dgl.sampling.sample_neighbors(self.hg, {'poi': sub_poi_nodes},
                                                        {'JCateOf': 0, 'BrandOf': -1, 'Cate1Of': 0, 'HasJunc': 0,
                                                         'HasPoi': 0, 'HasRoad': 0, 'NearBy': 0, 'RCateOf': 0},
                                                        edge_dir='out', copy_ndata=True, copy_edata=True)
        poi_cate1_graph = dgl.sampling.sample_neighbors(self.hg, {'poi': sub_poi_nodes},
                                                        {'JCateOf': 0, 'BrandOf': 0, 'Cate1Of': -1, 'HasJunc': 0,
                                                         'HasPoi': 0, 'HasRoad': 0, 'NearBy': 0, 'RCateOf': 0},
                                                        edge_dir='out', copy_ndata=True, copy_edata=True)
        junc_cate_graph = dgl.sampling.sample_neighbors(self.hg, {'junction': sub_junc_nodes},
                                                        {'JCateOf': -1, 'BrandOf': 0, 'Cate1Of': 0, 'HasJunc': 0,
                                                         'HasPoi': 0, 'HasRoad': 0, 'NearBy': 0, 'RCateOf': 0},
                                                        edge_dir='out', copy_ndata=True, copy_edata=True)
        road_cate_graph = dgl.sampling.sample_neighbors(self.hg, {'road': sub_road_nodes},
                                                        {'JCateOf': 0, 'BrandOf': 0, 'Cate1Of': 0, 'HasJunc': 0,
                                                         'HasPoi': 0, 'HasRoad': 0, 'NearBy': 0, 'RCateOf': -1},
                                                        edge_dir='out', copy_ndata=True, copy_edata=True)

        sub_brand_nodes = poi_brand_graph.edges(etype='BrandOf')[1]
        sub_cate1_nodes = poi_cate1_graph.edges(etype='Cate1Of')[1]
        sub_junc_cate_nodes = junc_cate_graph.edges(etype='JCateOf')[1]
        sub_road_cate_nodes = road_cate_graph.edges(etype='RCateOf')[1]
        region_nodes = torch.cat((torch.tensor([ent_id], dtype=torch.int64).to(self.graph_device), sub_region_nodes), dim=0)

        node_dicts = {'region': region_nodes, 'poi': sub_poi_nodes, 'road': sub_road_nodes, 'junction': sub_junc_nodes,
                      'brand': sub_brand_nodes, 'cate1': sub_cate1_nodes, 'junc_cate': sub_junc_cate_nodes,
                      'road_cate': sub_road_cate_nodes}

        for key in node_dicts.keys():
            if len(node_dicts[key]) > max_node_size[key]:
                node_dicts[key] = node_dicts[key][:max_node_size[key]]

        cur_sub_graph = dgl.node_subgraph(self.hg, node_dicts)
        # print('cur_sub_graph.ndata:', cur_sub_graph.ndata)
        cur_homo_sub_graph = dgl.to_homogeneous(cur_sub_graph, ndata=['f'])
        # print('cur_homo_sub_graph.ndata:', cur_homo_sub_graph.ndata)
        # print('f' in cur_homo_sub_graph.ndata)
        cur_node_size = cur_homo_sub_graph.num_nodes()
        # print('ent_id:', ent_id, 'cur_node_size:', cur_node_size, 'max_node_size:', max_node_size)

        if cur_node_size < sum(max_node_size.values()):
            num_virtual_node = sum(max_node_size.values()) - cur_node_size
            ori_num_nodes = cur_homo_sub_graph.num_nodes()
            vnodes_features = torch.zeros(num_virtual_node, cur_homo_sub_graph.ndata['f'].shape[1]).to(self.graph_device)
            cur_homo_sub_graph = dgl.add_nodes(cur_homo_sub_graph, num_virtual_node, {'f': vnodes_features})
            src_nodes = torch.tensor([i for i in range(ori_num_nodes)]).to(self.graph_device)
            dst_nodes = torch.tensor([ori_num_nodes + i for i in range(num_virtual_node)]).to(self.graph_device)
            vedges_src = []
            vedges_dst = []
            for i in range(src_nodes.shape[0]):
                for j in range(dst_nodes.shape[0]):
                    vedges_src.append(src_nodes[i])
                    vedges_dst.append(dst_nodes[j])
            cur_homo_sub_graph = dgl.add_edges(cur_homo_sub_graph, vedges_src, vedges_dst)

        sub_adj = cur_homo_sub_graph.adjacency_matrix().to_dense()
        sub_adj = (sub_adj + sub_adj.t()) / 2
        sub_feature = cur_homo_sub_graph.ndata['f']

        return sub_adj, sub_feature

    def get_region_sub_homo(self, ent_id, hop_k=1, max_node_size=1000):
        in_neighbors = [ent_id]
        for i in range(1, hop_k + 1):
            predecessors = [x for x in in_neighbors for x in self.g.predecessors(x)]
            in_neighbors = list(set(in_neighbors + predecessors))
        out_neighbors = [ent_id]
        for i in range(1, hop_k + 1):
            successors = [x for x in out_neighbors for x in self.g.successors(x)]
            out_neighbors = list(set(out_neighbors + successors))
        sub_nodes = list(set(in_neighbors + out_neighbors))
        if len(sub_nodes) > max_node_size:
            sub_nodes = sub_nodes[:max_node_size]
        else:
            num_virtual_node = max_node_size - len(sub_nodes)
            ori_num_nodes = self.g.num_nodes()
            vnodes_features = torch.zeros(num_virtual_node, self.g.ndata['features'].shape[1]).to(self.graph_device)
            self.g = dgl.add_nodes(self.g, num_virtual_node, {'features': vnodes_features})
            src_nodes = torch.tensor(sub_nodes).to(self.graph_device)
            dst_nodes = torch.tensor([ori_num_nodes + i for i in range(num_virtual_node)]).to(self.graph_device)
            vedges_src = []
            vedges_dst = []
            for i in range(src_nodes.shape[0]):
                for j in range(dst_nodes.shape[0]):
                    vedges_src.append(src_nodes[i])
                    vedges_dst.append(dst_nodes[j])
            self.g = dgl.add_edges(self.g, vedges_src, vedges_dst)
            sub_nodes = sub_nodes + [ori_num_nodes + i for i in range(num_virtual_node)]
        sub_g = dgl.node_subgraph(self.g, torch.tensor(sub_nodes).to(self.graph_device))
        sub_adj = sub_g.adjacency_matrix().to_dense()
        sub_adj = (sub_adj + sub_adj.t()) / 2
        sub_feature = sub_g.ndata['features']
        return sub_adj, sub_feature

class RCLData(object):
    def __init__(self, hkg, use_rule=False, if_test=False):
        self.hkg = hkg
        self.hg = hkg.hg
        self.use_rule = use_rule
        if if_test:
            self.region_attr_subgraph_all, self.region_sub_nodes_dicts = hkg.get_region_sub_test_all()
        else:
            self.region_attr_subgraph_all, self.region_sub_nodes_dicts = hkg.get_region_sub_all()

        print('subgraph data constructed.')

    def generated_rule_masks(self, num_masks, seed, args):
        rule_masks = []
        rule_type_len = self.hg.num_nodes('cate1') + self.hg.num_nodes('brand') + self.hg.num_nodes('junc_cate') + self.hg.num_nodes('road_cate')

        for i in range(num_masks):
            # random.seed(seed + i)
            rule_mask = torch.zeros(3, rule_type_len)
            mask_ratio = random.random()
            mask_selected = random.sample(range(rule_type_len), int(rule_type_len * mask_ratio))
            rule_mask[0, mask_selected] = args['mask_selected']
            for j in mask_selected:
                # rule_mask[1, j] = random.randint(0, 3)
                rule_mask[2, j] = random.random()
            rule_masks.append(rule_mask)

        return rule_masks

    def get_all_samples(self, ratio, aug_sel_ratio=0.2, seed=2024, rule_masks=None, args=None):
        # region_sp samples
        if not self.use_rule:
            sp_pos_samples = []
            sp_neg_samples = []
            random.seed(seed)
            for i in range(self.hkg.num_mht_region_ent):
                # seed += 1
                # random.seed(seed)
                region_nearby_neighbors = self.hg.successors(i, etype='NearBy')
                sp_pos_list = []
                sp_neg_list = []
                for j in range(self.hkg.num_mht_region_ent):
                    if j in region_nearby_neighbors:
                        sp_pos_list.append(self.region_attr_subgraph_all[j])
                    else:
                        sp_neg_list.append(self.region_attr_subgraph_all[j])
                pos_size = int(len(sp_pos_list) * ratio['sp_pos'])
                if pos_size == 0:
                    pos_size = 1
                neg_size = int(len(sp_neg_list) * ratio['sp_neg'])
                if neg_size == 0:
                    neg_size = 1
                sp_pos_list = random.sample(sp_pos_list, pos_size)
                sp_neg_list = random.sample(sp_neg_list, neg_size)
                sp_pos_samples.append(sp_pos_list)
                sp_neg_samples.append(sp_neg_list)

            # region_attr samples
            attr_pos_samples = []
            for i in tqdm(range(self.hkg.num_mht_region_ent)):
                attr_pos_augs = list()
                # attr_pos_augs.append(self.region_attr_subgraph_all[i])
                # attr_pos_aug = self.generated_attr_samples(self.region_sub_nodes_dicts[i], ratio, aug_sel_ratio, seed+i)
                attr_pos_aug = self.generated_attr_samples(self.region_sub_nodes_dicts[i], ratio, aug_sel_ratio, seed)
                attr_pos_augs.extend(attr_pos_aug)
                attr_pos_samples.append(attr_pos_augs)

            return sp_pos_samples, sp_neg_samples, attr_pos_samples

        if self.use_rule:
            region_ruled_subgraphs_list = []
            region_ruled_node_dicts_list = []
            sp_pos_samples_list = []
            sp_neg_samples_list = []
            attr_pos_samples_list = []

            for rule_mask in tqdm(rule_masks):
                region_ruled_subgraphs, region_ruled_node_dicts = self.hkg.get_region_sub_with_mask(rule_mask, args)
                region_ruled_subgraphs_list.append(region_ruled_subgraphs)
                region_ruled_node_dicts_list.append(region_ruled_node_dicts)

                sp_pos_samples = []
                sp_neg_samples = []
                for i in range(self.hkg.num_chi_region_ent):
                    region_nearby_neighbors = self.hg.successors(i, etype='NearBy')
                    sp_pos_list = []
                    sp_neg_list = []
                    for j in range(self.hkg.num_mht_region_ent):
                        if j in region_nearby_neighbors:
                            sp_pos_list.append(region_ruled_subgraphs[j])
                        else:
                            sp_neg_list.append(region_ruled_subgraphs[j])

                    sp_pos_list = random.sample(sp_pos_list, int(len(sp_pos_list) * ratio['sp_pos']))
                    sp_neg_list = random.sample(sp_neg_list, int(len(sp_neg_list) * ratio['sp_neg']))
                    sp_pos_samples.append(sp_pos_list)
                    sp_neg_samples.append(sp_neg_list)

                sp_pos_samples_list.append(sp_pos_samples)
                sp_neg_samples_list.append(sp_neg_samples)

                attr_pos_samples = []
                for i in range(self.hkg.num_mht_region_ent):
                    attr_pos_augs = list()
                    # attr_pos_augs.append(region_ruled_subgraphs[i])
                    attr_pos_aug = self.generated_attr_samples(region_ruled_node_dicts[i], ratio, aug_sel_ratio, seed)
                    attr_pos_augs.extend(attr_pos_aug)
                    attr_pos_samples.append(attr_pos_augs)
                attr_pos_samples_list.append(attr_pos_samples)

            return sp_pos_samples_list, sp_neg_samples_list, attr_pos_samples_list, region_ruled_subgraphs_list

    def generated_attr_samples(self, sub_node_dicts, ratio, aug_sel_ratio=0.2, seed=2024):
        # pos samples, input subgraph and node_dicts of current region
        attr_pos_samples = []
        # graph nodes info
        region_nodes = self.hg.nodes('region')
        poi_nodes = self.hg.nodes('poi')
        road_nodes = self.hg.nodes('road')
        junc_nodes = self.hg.nodes('junction')
        brand_nodes = self.hg.nodes('brand')
        cate1_nodes = self.hg.nodes('cate1')
        junc_cate_nodes = self.hg.nodes('junc_cate')
        road_cate_nodes = self.hg.nodes('road_cate')
        # streetview_nodes = self.hg.nodes('streetview')
        g_nodes_dicts = {'region': region_nodes, 'poi': poi_nodes, 'road': road_nodes, 'junction': junc_nodes,
                         'brand': brand_nodes, 'cate1': cate1_nodes, 'junc_cate': junc_cate_nodes,
                         'road_cate': road_cate_nodes
                        # , 'streetview': streetview_nodes
                         }
        # node-level
        attr_pos_sub = dgl.node_subgraph(self.hg, sub_node_dicts)
        random.seed(seed)
        for ntype, nodes in sub_node_dicts.items():
            # seed += 1
            if ntype == 'region':
                continue
            else:
                # random.seed(seed)
                cur_sel_ratio = random.random()
                if cur_sel_ratio < aug_sel_ratio:
                    node_size = len(nodes)
                    add_size = int(node_size * ratio[ntype]['add'])
                    candi_list = list(set(g_nodes_dicts[ntype]) - set(nodes))
                    if len(candi_list) == 0:
                        continue
                    elif len(candi_list) < add_size:
                        add_nodes = candi_list
                        add_nodes = add_nodes * (add_size // len(add_nodes)) + add_nodes[:add_size % len(add_nodes)]
                    else:
                        add_nodes = random.sample(list(set(g_nodes_dicts[ntype]) - set(nodes)), add_size)
                    add_nodes = [x.item() for x in add_nodes]
                    attr_pos_sub.add_nodes(len(add_nodes), data={'f': self.hg.nodes[ntype].data['f'][add_nodes]}, ntype=ntype)
                    candi_list = [x for x in range(node_size)]
                    del_nodes = random.sample(candi_list, int(node_size * ratio[ntype]['del']))
                    attr_pos_sub.remove_nodes(del_nodes, ntype=ntype)
        attr_pos_samples.append(attr_pos_sub)

        # edge-level
        attr_pos_sub = dgl.node_subgraph(self.hg, sub_node_dicts)
        for ctype in attr_pos_sub.canonical_etypes:
            src_ntype, etype, dst_ntype = ctype
            if etype == 'NearBy':
                continue
            # seed += 1
            cur_sel_ratio = random.random()
            if cur_sel_ratio < aug_sel_ratio:
                edges = attr_pos_sub.edges(etype=etype)
                # add
                add_size = int(edges[0].shape[0] * ratio[etype]['add'])
                src_nodes = attr_pos_sub.nodes(src_ntype)
                dst_nodes = attr_pos_sub.nodes(dst_ntype)
                predecessors_list = []
                for node_id in dst_nodes:
                    predecessors = attr_pos_sub.predecessors(node_id, etype=etype)
                    predecessors_list.append(predecessors)
                predecessors = list(set([x for y in predecessors_list for x in y]))
                successors_list = []
                for node_id in src_nodes:
                    successors = attr_pos_sub.successors(node_id, etype=etype)
                    successors_list.append(successors)
                successors = list(set([x for y in successors_list for x in y]))
                candi_dst_lst = list(set(dst_nodes)-set(successors))
                if len(candi_dst_lst) < add_size:
                    add_edges_dst = candi_dst_lst
                else:
                    add_edges_dst = random.sample(candi_dst_lst, add_size)

                candi_src_lst = list(set(src_nodes)-set(predecessors))
                if len(candi_src_lst) < add_size:
                    add_edges_src = candi_src_lst
                    add_edges_src = add_edges_src * (add_size // len(add_edges_src)) + add_edges_src[:add_size % len(add_edges_src)]
                else:
                    add_edges_src = random.sample(candi_src_lst, add_size)
                attr_pos_sub.add_edges(add_edges_src, add_edges_dst, data=None, etype=etype)
                # del
                del_edges_id = random.sample([x for x in range(edges[0].shape[0])], int(edges[0].shape[0] * ratio[etype]['del']))
                del_edges = (edges[0][del_edges_id], edges[1][del_edges_id])
                del_ids = attr_pos_sub.edge_ids(del_edges[0], del_edges[1], etype=etype)
                attr_pos_sub.remove_edges(del_ids, etype=etype)
        attr_pos_samples.append(attr_pos_sub)

        return attr_pos_samples

    def gaussian_noise(self, data, seed, std, mean=0):
        random.seed(seed)
        aug_data = data.clone()
        aug_data += torch.randn_like(data) * std + mean
        aug_data[aug_data < 0] = 0
        return aug_data

    def get_batch_samples(self, sp_pos_all, sp_neg_all, attr_pos_all, batch_size, args):
        reg_sub_batch = []
        sp_pos_batch = []
        sp_neg_batch = []
        attr_pos_batch = []
        reg_inf_batch = []
        reg_outf_batch = []
        reg_img_batch = []
        reg_inf_aug_batch = []
        reg_outf_aug_batch = []
        batch_num = len(sp_pos_all) // batch_size
        seed = args['seed']
        noise_std = args['noise_std']
        sample_size = args['sample_size']
        for i in range(batch_num):
            reg_cur_batch = []
            reg_inf_cur_batch = []
            reg_outf_cur_batch = []
            reg_inf_aug_cur_batch = []
            reg_outf_aug_cur_batch = []
            reg_img_cur_batch = []
            for j in range(batch_size):

                reg_cur_batch.append(self.region_attr_subgraph_all[i * batch_size + j])

                reg_inf = self.hg.nodes['region'].data['inflow'][i * batch_size + j]
                reg_outf = self.hg.nodes['region'].data['outflow'][i * batch_size + j]
                reg_inf_cur_batch.append(reg_inf)
                reg_outf_cur_batch.append(reg_outf)

                reg_img = self.hg.nodes['region'].data['si_img'][i * batch_size + j]
                reg_img_cur_batch.append(reg_img)

                # add gaussian noise to inflow and outflow
                reg_inf_aug = self.gaussian_noise(reg_inf, seed + j, noise_std)
                reg_outf_aug = self.gaussian_noise(reg_outf, seed + j, noise_std)
                reg_inf_aug_cur_batch.append(reg_inf_aug)
                reg_outf_aug_cur_batch.append(reg_outf_aug)

            reg_inf_batch.append(torch.stack(reg_inf_cur_batch))
            reg_outf_batch.append(torch.stack(reg_outf_cur_batch))
            reg_sub_batch.append(reg_cur_batch)
            reg_img_batch.append(torch.stack(reg_img_cur_batch))
            reg_inf_aug_batch.append(torch.stack(reg_inf_aug_cur_batch))
            reg_outf_aug_batch.append(torch.stack(reg_outf_aug_cur_batch))

            if sample_size is None:
                sp_pos_batch.append(sp_pos_all[i * batch_size: (i + 1) * batch_size])
                sp_neg_batch.append(sp_neg_all[i * batch_size: (i + 1) * batch_size])
                attr_pos_batch.append(attr_pos_all[i * batch_size: (i + 1) * batch_size])
            else:
                sp_pos_cur_batch = []
                sp_neg_cur_batch = []
                attr_pos_cur_batch = []
                for j in range(batch_size):
                    sp_pos_cur_samples = random.sample(sp_pos_all[i * batch_size + j], sample_size)
                    sp_neg_cur_samples = random.sample(sp_neg_all[i * batch_size + j], sample_size)
                    attr_pos_cur_samples = random.sample(attr_pos_all[i * batch_size + j], sample_size)
                    sp_pos_cur_batch.append(sp_pos_cur_samples)
                    sp_neg_cur_batch.append(sp_neg_cur_samples)
                    attr_pos_cur_batch.append(attr_pos_cur_samples)
                sp_pos_batch.append(sp_pos_cur_batch)
                sp_neg_batch.append(sp_neg_cur_batch)
                attr_pos_batch.append(attr_pos_cur_batch)
        return reg_sub_batch, sp_pos_batch, sp_neg_batch, attr_pos_batch, reg_inf_batch, reg_outf_batch, reg_inf_aug_batch, reg_outf_aug_batch, reg_img_batch

    def get_rule_batch_samples(self, sp_pos_eph_list, sp_neg_eph_list, attr_pos_eph_list, reg_ruled_eph_list,
                               batch_num, batch_size, sample_size=None):
        epochs = len(sp_pos_eph_list)
        reg_sub_batch = []
        reg_inf_batch = []
        reg_outf_batch = []

        for i in range(batch_num):
            reg_cur_batch = []
            reg_inf_cur_batch = []
            reg_outf_cur_batch = []
            for j in range(batch_size):
                reg_cur_batch.append(self.region_attr_subgraph_all[i * batch_size + j])
                reg_inf_cur_batch.append(self.hg.nodes['region'].data['inflow'][i * batch_size + j])
                reg_outf_cur_batch.append(self.hg.nodes['region'].data['outflow'][i * batch_size + j])
            reg_inf_batch.append(torch.stack(reg_inf_cur_batch))
            reg_outf_batch.append(torch.stack(reg_outf_cur_batch))
            reg_sub_batch.append(reg_cur_batch)

        sp_pos_eph_batch = []
        sp_neg_eph_batch = []
        attr_pos_eph_batch = []
        reg_ruled_eph_batch = []
        for epoch in range(epochs):
            sp_pos_batch = []
            sp_neg_batch = []
            attr_pos_batch = []
            reg_ruled_batch = []
            for i in range(batch_num):
                reg_ruled_batch.append(reg_ruled_eph_list[epoch][i * batch_size: (i + 1) * batch_size])

                if sample_size is None:
                    sp_pos_batch.append(sp_pos_eph_list[epoch][i * batch_size: (i + 1) * batch_size])
                    sp_neg_batch.append(sp_neg_eph_list[epoch][i * batch_size: (i + 1) * batch_size])
                    attr_pos_batch.append(attr_pos_eph_list[epoch][i * batch_size: (i + 1) * batch_size])
                else:
                    sp_pos_cur_batch = []
                    sp_neg_cur_batch = []
                    attr_pos_cur_batch = []
                    for j in range(batch_size):
                        sp_pos_cur_samples = random.sample(sp_pos_eph_list[epoch][i * batch_size + j], sample_size)
                        sp_neg_cur_samples = random.sample(sp_neg_eph_list[epoch][i * batch_size + j], sample_size)
                        attr_pos_cur_samples = random.sample(attr_pos_eph_list[epoch][i * batch_size + j], sample_size)
                        sp_pos_cur_batch.append(sp_pos_cur_samples)
                        sp_neg_cur_batch.append(sp_neg_cur_samples)
                        attr_pos_cur_batch.append(attr_pos_cur_samples)
                    sp_pos_batch.append(sp_pos_cur_batch)
                    sp_neg_batch.append(sp_neg_cur_batch)
                    attr_pos_batch.append(attr_pos_cur_batch)
            sp_pos_eph_batch.append(sp_pos_batch)
            sp_neg_eph_batch.append(sp_neg_batch)
            attr_pos_eph_batch.append(attr_pos_batch)
            reg_ruled_eph_batch.append(reg_ruled_batch)
        return reg_sub_batch, sp_pos_eph_batch, sp_neg_eph_batch, attr_pos_eph_batch, reg_inf_batch, reg_outf_batch, reg_ruled_eph_batch


if __name__ == '__main__':
    kg_dir = 'data/nymhtkg/mht180.csv'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hkg = HeteroGraphData(kg_dir, reverse=False, export=True, graph_device=device)
    print('kg data loaded...')
    hkg.init_hetero_graph_features(feature_dir=None)
    print('hetero graph feature loaded...')
    rcl_data = RCLData(hkg)
    print('rcl data constructed...')
    ratio_dict = {'poi': {'add': 0.2, 'del': 0.2}, 'road': {'add': 0.2, 'del': 0.2},
                  'junc': {'add': 0.2, 'del': 0.2}, 'brand': {'add': 0.2, 'del': 0.2},
                  'cate1': {'add': 0.2, 'del': 0.2}, 'junc_cate': {'add': 0.2, 'del': 0.2},
                  'road_cate': {'add': 0.2, 'del': 0.2}, 'NearBy': {'add': 0.2, 'del': 0.2},
                  'HasJunc': {'add': 0.2, 'del': 0.2}, 'JCateOf': {'add': 0.2, 'del': 0.2},
                  'HasRoad': {'add': 0.2, 'del': 0.2}, 'RCateOf': {'add': 0.2, 'del': 0.2},
                  'HasPoi': {'add': 0.2, 'del': 0.2}, 'BrandOf': {'add': 0.2, 'del': 0.2},
                  'Cate1Of': {'add': 0.2, 'del': 0.2}, 'StreetViewOf': {'add': 0.2, 'del': 0.2}}
    sp_pos_all, sp_neg_all, attr_pos_all = rcl_data.get_all_samples(ratio=ratio_dict)
    print('all samples constructed...')
    reg_sub_batch, sp_pos_batch, sp_neg_batch, attr_pos_batch, reg_inf_batch, reg_outf_batch \
        = rcl_data.get_batch_samples(sp_pos_all, sp_neg_all, attr_pos_all, 32)
    print('batch samples constructed...')