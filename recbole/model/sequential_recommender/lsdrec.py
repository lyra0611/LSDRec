import logging
from collections import defaultdict
from copy import deepcopy
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder, CLLayer
from dgl.nn.pytorch import GraphConv
import dgl
import networkx as nx
import dgl.function as fn




def min_max_normalization(w):
    min_val = w.min()
    max_val = w.max()
    return (w - min_val) / (max_val - min_val)
    
    
def cal_w2(G):
    nx_g = G.cpu().to_networkx().to_undirected()
    w2 = nx.degree_centrality(nx_g)
    w2 = [w2[i] for i in range(G.num_nodes())]
    w2 = torch.tensor(w2, dtype=torch.float32)
    w2 = torch.log(w2+1e-8)
    w2 = min_max_normalization(w2)
    return w2
    

   
def graph_dual_neighbor_readout(g: dgl.DGLGraph, aug_g: dgl.DGLGraph, node_ids, features):
    _, all_neighbors = g.out_edges(node_ids)
    all_nbr_num = g.out_degrees(node_ids)
    _, foreign_neighbors = aug_g.out_edges(node_ids)
    for_nbr_num = aug_g.out_degrees(node_ids)

    all_neighbors = [set(t.tolist())
                     for t in all_neighbors.split(all_nbr_num.tolist())]
    foreign_neighbors = [set(t.tolist())
                         for t in foreign_neighbors.split(for_nbr_num.tolist())]
    for i, nbrs in enumerate(foreign_neighbors):
        if len(nbrs) > 10:
            nbrs = random.sample(nbrs, 10)
            foreign_neighbors[i] = set(nbrs)
    civil_neighbors = [all_neighbors[i]-foreign_neighbors[i]
                       for i in range(len(all_neighbors))]

    for i, nbrs in enumerate(civil_neighbors):
        if len(nbrs) > 10:
            nbrs = random.sample(nbrs, 10)
            civil_neighbors[i] = set(nbrs)
    for_lens = [len(t) for t in foreign_neighbors]
    cv_lens = torch.tensor([len(t)
                           for t in civil_neighbors], dtype=torch.int16)
    zero_indicies = (cv_lens == 0).nonzero().view(-1).tolist()
    cv_lens = cv_lens[cv_lens > 0].tolist()
    foreign_neighbors = torch.cat(
        [torch.tensor(list(s), dtype=torch.long) for s in foreign_neighbors])
    civil_neighbors = torch.cat(
        [torch.tensor(list(s), dtype=torch.long) for s in civil_neighbors])
    cv_feats = features[civil_neighbors].split(cv_lens)
    cv_feats = [t.mean(dim=0) for t in cv_feats]

    if len(zero_indicies) > 0:
        for i in zero_indicies:
            cv_feats.insert(i, torch.zeros_like(features[0]))
    for_feats = features[foreign_neighbors].split(for_lens)
    for_feats = [t.mean(dim=0) for t in for_feats]
    return torch.stack(cv_feats, dim=0), torch.stack(for_feats, dim=0)


def graph_augment(g: dgl.DGLGraph, user_ids, user_edges):

    user_ids = user_ids.cpu().numpy()
    node_indicies_a = np.concatenate(
        user_edges.loc[user_ids, "item_edges_a"].to_numpy())
    node_indicies_b = np.concatenate(
        user_edges.loc[user_ids, "item_edges_b"].to_numpy())
    node_indicies_a = torch.from_numpy(
        node_indicies_a).to(g.device)
    node_indicies_b = torch.from_numpy(
        node_indicies_b).to(g.device)
    edge_ids = g.edge_ids(node_indicies_a, node_indicies_b)

    aug_g: dgl.DGLGraph = deepcopy(g)

    aug_g.remove_edges(edge_ids)
    return aug_g


def graph_dropout(g: dgl.DGLGraph, keep_prob):

    origin_edge_w = g.edata['w']

    drop_size = int((1-keep_prob) * g.num_edges())
    random_index = torch.randint(
        0, g.num_edges(), (drop_size,), device=g.device)
    mask = torch.zeros(g.num_edges(), dtype=torch.uint8,
                       device=g.device).bool()
    mask[random_index] = True
    g.edata['w'].masked_fill_(mask, 0)

    return origin_edge_w, g


class GCN(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_prob=0.7):
        super(GCN, self).__init__()
        self.dropout_prob = dropout_prob
        self.layer = GraphConv(in_dim, out_dim, weight=False,
                               bias=False, allow_zero_in_degree=True)

    def forward(self, graph, feature):
        origin_w, graph = graph_dropout(graph, 1-self.dropout_prob)
        embs = [feature]
        for i in range(2):
            feature = self.layer(graph, feature, edge_weight=graph.edata['w'])
            F.dropout(feature, p=0.2, training=self.training)
            embs.append(feature)
        embs = torch.stack(embs, dim=1)
        final_emb = torch.mean(embs, dim=1)

        graph.edata['w'] = origin_w
        return final_emb


class LSDRec(SequentialRecommender):

    def __init__(self, config, dataset, external_data):
        super(LSDRec, self).__init__(config, dataset)

        self.config = config

        self.device = config["device"]
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  
 
        self.inner_size = config['inner_size']
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']

        self.mask_ratio = config['mask_ratio']

        self.loss_type = config['loss_type']
        self.initializer_range = config['initializer_range']

        self.short_len = config["short_len"]
 
        self.cl_l = config["cl_l"]       
        print('--------------------------------------\n')
        print('short len is : ', self.short_len)
        print('--------------------------------------\n')


        self.item_embedding = nn.Embedding(
            self.n_items + 1, self.hidden_size, padding_idx=0)  # mask token add 1
        self.position_embedding = nn.Embedding(
            self.max_seq_length + 1, self.hidden_size)  # add mask_token at the last
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )
        self.trm_encoder2 = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )
        self.LayerNorm = nn.LayerNorm(
            self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        # Contrastive Learning
        self.contrastive_learning_layer = CLLayer(self.hidden_size, tau=config['cl_temp'])

        # Fusion Attn
        self.attn_weights = nn.Parameter(
            torch.Tensor(self.hidden_size, self.hidden_size))
        self.attn = nn.Parameter(torch.Tensor(1, self.hidden_size))
        nn.init.normal_(self.attn, std=0.02)
        nn.init.normal_(self.attn_weights, std=0.02)

        # Global Graph Learning
        self.item_adjgraph = external_data["adj_graph"].to(self.device)
        self.user_edges = external_data["user_edges"]

        self.graph_dropout = config["graph_dropout_prob"]

        self.adj_graph_test = external_data["adj_graph_test"].to(self.device)


        self.gcn = GCN(self.hidden_size, self.hidden_size, self.graph_dropout)

        self.layernorm = nn.LayerNorm(
            self.hidden_size, eps=self.layer_norm_eps)

        self.loss_fct = nn.CrossEntropyLoss()

    
        try:
            assert self.loss_type in ['CE']
        except AssertionError:
            raise AssertionError("Make sure 'loss_type' be CE!")


        self.apply(self._init_weights)
        
        

        
        self.w2 = cal_w2(self.item_adjgraph)

        




    def _subgraph_agreement(self, aug_g, raw_output_all, raw_output_seq, valid_items_flatten):

        aug_output_seq = self.gcn_forward(g=aug_g)[valid_items_flatten]
        civil_nbr_ro, foreign_nbr_ro = graph_dual_neighbor_readout(
            self.item_adjgraph, aug_g, valid_items_flatten, raw_output_all)

        view1_sim = F.cosine_similarity(
            raw_output_seq, aug_output_seq, eps=1e-12)
            
    
        
        agreement = view1_sim
        agreement = torch.sigmoid(agreement)
      
        agreement = (agreement - agreement.min()) / \
            (agreement.max() - agreement.min())
        agreement = (self.config["weight_mean"] / agreement.mean()) * agreement
    
        
        return agreement
        

    def _init_weights(self, module):
     
        if isinstance(module, (nn.Linear, nn.Embedding)):
   
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_attention_mask(self, item_seq, task_label=False):
        
        if task_label:
            label_pos = torch.ones((item_seq.size(0), 1), device=self.device)
            item_seq = torch.cat((label_pos, item_seq), dim=1)
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(
            1).unsqueeze(2)  
       
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def gcn_forward(self, g=None):
        item_emb = self.item_embedding.weight
        item_emb = self.dropout(item_emb)
        light_out = self.gcn(g, item_emb)
        return self.layernorm(light_out+item_emb)

    def forward(self, item_seq, item_seq_len, short = 0, return_all=False):
        position_ids = torch.arange(item_seq.size(
            1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        if short == 0:
            trm_output = self.trm_encoder(
                input_emb, extended_attention_mask, output_all_encoded_layers=True)
        else:
            trm_output = self.trm_encoder2(
                input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]

        if return_all:
            return output
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B H]

    def calculate_loss(self, interaction):
        return self.calculate_loss_new(interaction)

    def calculate_loss_new(self, interaction):
        
        torch.set_printoptions(profile="full")
        np.set_printoptions(threshold=np.inf)
        
        user_ids = interaction[self.USER_ID]
        item_seq = interaction[self.ITEM_SEQ]
        pos_items = interaction[self.ITEM_ID]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        last_items_indices = torch.tensor([i*self.max_seq_length+j for i, j in enumerate(
            item_seq_len - 1)], dtype=torch.long, device=item_seq.device).view(-1)
 
        last_items_flatten = item_seq.view(-1)[last_items_indices]

        valid_items_flatten = last_items_flatten
        valid_items_indices = last_items_indices
        
      
        seq_output = self.forward(item_seq, item_seq_len, return_all=False)

        aug_seq_output = self.forward(item_seq, item_seq_len, return_all=True).view(
            -1, self.config["hidden_size"])[valid_items_indices]

    
        len = self.short_len
        k = torch.clamp(item_seq_len - len, min=0)
        item_seq_s = torch.zeros_like(item_seq)
        for i in range(item_seq.size(0)):
            item_seq_s[i, :item_seq_len[i] - k[i]] = item_seq[i, k[i]:item_seq_len[i]]
        item_seq_s_len = torch.where(item_seq_len > len, torch.full_like(item_seq_len, len), item_seq_len)

        last_items_indices_s = torch.tensor([i*self.max_seq_length+j for i, j in enumerate(
            item_seq_s_len - 1)], dtype=torch.long, device=item_seq_s.device).view(-1)
        valid_items_indices_s = last_items_indices_s
        


        self.w2 = self.w2.to(item_seq.device)
        w_l = torch.sum(self.w2[item_seq], dim=1) / item_seq_len
        w_s = torch.sum(self.w2[item_seq_s], dim=1) / item_seq_s_len

        
        seq_s_output = self.forward(item_seq_s, item_seq_s_len, short = 1, return_all=False)
        aug_seq_s_output = self.forward(item_seq_s, item_seq_s_len, short = 1, return_all=True).view(-1, self.config["hidden_size"])[valid_items_indices_s]

    
        masked_g = self.item_adjgraph
        iadj_graph_output_raw = self.gcn_forward(masked_g)

        iadj_graph_output_seq = iadj_graph_output_raw[valid_items_flatten]
        

   
        
        # contrastive learning
        cl_loss_adj = self.contrastive_learning_layer.vanilla_loss(aug_seq_output, iadj_graph_output_seq)
        cl_loss_adj2 = self.contrastive_learning_layer.vanilla_loss(iadj_graph_output_seq, aug_seq_s_output)

        cl_loss_l =  (self.cl_l*w_l*cl_loss_adj).mean()
        cl_loss_s = (self.config["graphcl_coefficient"] * w_s*cl_loss_adj2).mean()

        # Fusion After CL
        if self.config["graph_view_fusion"]:
      
            mixed_x = torch.stack(
                (seq_output, seq_s_output), dim=0)
            weights = (torch.matmul(
                mixed_x, self.attn_weights.unsqueeze(0))*self.attn).sum(-1)
          
            score = F.softmax(weights, dim=0).unsqueeze(-1)
            seq_output = (mixed_x*score).sum(0)
        
        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        loss = self.loss_fct(logits, pos_items)

        if torch.isnan(loss):
            logging.error("loss is nan")

    
        return loss, cl_loss_l,cl_loss_s

    def fast_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction["item_id_with_negs"]
        seq_output = self.forward(item_seq, item_seq_len, short = 0)



        len = self.short_len
        k = torch.clamp(item_seq_len - len, min=0)
        item_seq_s = torch.zeros_like(item_seq)
        for i in range(item_seq.size(0)):
            item_seq_s[i, :item_seq_len[i] - k[i]] = item_seq[i, k[i]:item_seq_len[i]]
        item_seq_s_len = torch.where(item_seq_len > len, torch.full_like(item_seq_len, len), item_seq_len)

        seq_s_output = self.forward(item_seq_s, item_seq_s_len, short = 1)
       

        if self.config["graph_view_fusion"]:

            mixed_x = torch.stack(
                (seq_output, seq_s_output), dim=0)
            weights = (torch.matmul(
                mixed_x, self.attn_weights.unsqueeze(0))*self.attn).sum(-1)
    
            score = F.softmax(weights, dim=0).unsqueeze(-1)
            seq_output = (mixed_x*score).sum(0)

        test_item_emb = self.item_embedding(test_item)  # [B, num, H]
        scores = torch.matmul(seq_output.unsqueeze(
            1), test_item_emb.transpose(1, 2)).squeeze()
        return scores

