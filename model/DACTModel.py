import torch
import torch.nn as nn
import torch.nn.functional as F

from model.HGT import HGT
from model.HGTIntra import HGTIntra
from model.HGTInter import HGTInter

from function.tools import triplet_loss, mobility_loss, cross_loss


class AutoEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, norm=True):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(out_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, in_dim),
            nn.ReLU()
        )

    def encode(self, feature):
        return self.encoder(feature)

    def decode(self, latent):
        return self.decoder(latent)

    def forward(self, feature):
        latent = self.encoder(feature)
        fea_new = self.decoder(latent)
        return fea_new, latent


class DACTModel(nn.Module):
    def __init__(self, args, city_num):
        super(DACTModel, self).__init__()
        self.args = args
        self.seed = args['seed']
        self.intra_node_dict = args['intra_node_dict']
        self.intra_edge_dict = args['intra_edge_dict']
        self.inter_node_dict = args['inter_node_dict']
        self.inter_edge_dict = args['inter_edge_dict']
        self.region_node_dict = args['region_node_dict']
        self.region_edge_dict = args['region_edge_dict']
        self.schema_node_dict = args['schema_node_dict']
        self.schema_edge_dict = args['schema_edge_dict']
        self.city_num = city_num
        self.intra_graph_encoder = nn.ModuleList()
        for i in range(self.city_num):
            self.intra_graph_encoder.append(HGTIntra(node_dict=self.intra_node_dict,
                                                     edge_dict=self.intra_edge_dict,
                                                     region_node_dict=self.region_node_dict,
                                                     region_edge_dict=self.region_edge_dict,
                                                     n_inp=args['node_dim'][0],
                                                     n_hid=args['node_dim'][1],
                                                     n_out=args['node_dim'][2],
                                                     n_layers=args['n_layers'],
                                                     n_heads=args['n_heads'],
                                                     batch_size=args['intra_batch_size'][i],
                                                     agg_method=args['agg_method'],
                                                     use_norm=args['use_norm']))
        self.inter_graph_encoder = nn.ModuleList()
        for i in range(self.city_num):
            self.inter_graph_encoder.append(HGTInter(node_dict=self.inter_node_dict,
                                                     edge_dict=self.inter_edge_dict,
                                                     region_node_dict=self.region_node_dict,
                                                     region_edge_dict=self.region_edge_dict,
                                                     schema_node_dict=self.schema_node_dict,
                                                     schema_edge_dict=self.schema_edge_dict,
                                                     n_inp=args['node_dim'][0],
                                                     n_hid=args['node_dim'][1],
                                                     n_out=args['node_dim'][2],
                                                     n_layers=args['n_layers'],
                                                     n_heads=args['n_heads'],
                                                     batch_size=args['inter_batch_size'][i],
                                                     agg_method=args['agg_method'],
                                                     use_norm=args['use_norm']))
        # flow_encoder
        self.flow_encoder = nn.ModuleList()
        for i in range(self.city_num):
            self.flow_encoder.append(AutoEncoder(in_dim=args['flow_dim'][0],
                                                 hid_dim=args['flow_dim'][1],
                                                 out_dim=args['flow_dim'][2]))

        # fusion
        self.fusion = nn.ModuleList()
        for i in range(self.city_num):
            self.fusion.append(nn.Sequential(
                # spatial_pos, spatial_neg, flow_in, flow_out -> out_dim
                nn.Linear(args['node_dim'][2] + args['node_dim'][2] + args['flow_dim'][2] + args['flow_dim'][2],
                          args['out_dim']),
                nn.BatchNorm1d(args['out_dim']),
                nn.ReLU()
            ))

        # predictor
        self.in_flow_predictor = nn.ModuleList()
        self.out_flow_predictor = nn.ModuleList()
        self.region_pos_r_predictor = nn.ModuleList()
        self.region_neg_r_predictor = nn.ModuleList()
        self.inter_predictor = nn.ModuleList()
        for i in range(self.city_num):
            self.in_flow_predictor.append(nn.Sequential(
                nn.Linear(args['out_dim'], args['flow_dim'][2]),
                nn.ReLU()
            ))
            self.out_flow_predictor.append(nn.Sequential(
                nn.Linear(args['out_dim'], args['flow_dim'][2]),
                nn.ReLU()
            ))
            self.region_pos_r_predictor.append(nn.Sequential(
                nn.Linear(args['out_dim'], args['node_dim'][2]),
                nn.ReLU()
            ))
            self.region_neg_r_predictor.append(nn.Sequential(
                nn.Linear(args['out_dim'], args['node_dim'][2]),
                nn.ReLU()
            ))
            self.inter_predictor.append(nn.Sequential(
                nn.Linear(args['out_dim'], args['node_dim'][2]),
                nn.ReLU()
            ))

    def forward(self,
                sp_pos_samples_graph, sp_neg_samples_graph,
                sp_pos_samples_batch, sp_neg_samples_batch,
                mobility_samples_batch,
                flow_in_samples_batch, flow_out_samples_batch,
                inter_samples_graph,
                device, args):
        loss_spatial = 0
        loss_mobility = 0
        loss_cross = 0
        loss_pred = 0
        loss_final = 0
        out_embedding_list = []
        for i in range(self.city_num):
            emb1 = self.intra_graph_encoder[i](sp_pos_samples_graph[i], sp_pos_samples_batch[i], self.seed, device)
            emb2 = self.intra_graph_encoder[i](sp_neg_samples_graph[i], sp_neg_samples_batch[i], self.seed, device)
            loss_spatial += triplet_loss(emb1[1], emb1[2], emb2[1], emb2[2], margin=args['spatial_margin'])

            flow_in_emb = self.flow_encoder[i].encode(flow_in_samples_batch[i])
            flow_out_emb = self.flow_encoder[i].encode(flow_out_samples_batch[i])
            mobility_gen_loss = mobility_loss(flow_in_emb, flow_out_emb, mobility_samples_batch[i])
            loss_mobility += mobility_gen_loss

            inter_emb = self.inter_graph_encoder[i](inter_samples_graph[i], self.seed, device)
            loss_cross += cross_loss(inter_emb[1], inter_emb[2], inter_emb[3], inter_emb[4], margin=args['cross_margin'])

            out_emb = self.fusion[i](torch.cat((emb1[1], emb2[1], flow_in_emb, flow_out_emb), dim=1))
            out_embedding_list.append(out_emb)

            in_flow_pred = self.in_flow_predictor[i](out_emb)
            out_flow_pred = self.out_flow_predictor[i](out_emb)
            region_pos_r_pred = self.region_pos_r_predictor[i](out_emb)
            region_neg_r_pred = self.region_neg_r_predictor[i](out_emb)
            inter_pred = self.inter_predictor[i](out_emb)
            pred_loss = F.mse_loss(torch.cat((in_flow_pred, out_flow_pred, region_pos_r_pred, region_neg_r_pred), dim=1),
                                   torch.cat((flow_in_emb, flow_out_emb, emb1[1], emb2[1]), dim=1))
            loss_pred += pred_loss


        loss_final = (loss_spatial + args['mobility_loss_weight'] * loss_mobility + args['cross_loss_weight']*loss_cross + args['pred_loss_weight'] * loss_pred)

        return loss_final, loss_spatial, loss_mobility, loss_cross, loss_pred, out_embedding_list
