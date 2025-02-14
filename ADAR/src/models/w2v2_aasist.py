# Adapted from https://github.com/piotrkawa/audio-deepfake-source-tracing

import math
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearXavier(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(LinearXavier, self).__init__(in_features, out_features, bias)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.01)
        
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        # attention map
        self.att_proj = LinearXavier(in_dim, out_dim)
        self.att_weight = self._init_new_params(out_dim, 1)

        # project
        self.proj_with_att = LinearXavier(in_dim, out_dim)
        self.proj_without_att = LinearXavier(in_dim, out_dim)

        # batch norm
        self.bn = nn.BatchNorm1d(out_dim)

        # dropout for inputs
        self.input_drop = nn.Dropout(p=0.2)

        # activate
        self.act = nn.SELU(inplace=True)

        # temperature
        self.temp = 1.0
        if "temperature" in kwargs:
            self.temp = kwargs["temperature"]

    def forward(self, x):
        """
        x   :(#bs, #node, #dim)
        """
        # apply input dropout
        x = self.input_drop(x)
        #        print(x.shape,'GraphAttentionLayer_x')

        # derive attention map
        att_map = self._derive_att_map(x)

        # projection
        x = self._project(x, att_map)

        # apply batch norm
        x = self._apply_BN(x)
        x = self.act(x)
        return x

    def _pairwise_mul_nodes(self, x):
        """
        Calculates pairwise multiplication of nodes.
        - for attention map
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, #dim)
        """

        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)

        return x * x_mirror

    def _derive_att_map(self, x):
        """
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, 1)
        """
        att_map = self._pairwise_mul_nodes(x)
        # size: (#bs, #node, #node, #dim_out)
        att_map = torch.tanh(self.att_proj(att_map))
        # size: (#bs, #node, #node, 1)
        att_map = torch.matmul(att_map, self.att_weight)

        # apply temperature
        att_map = att_map / self.temp

        att_map = F.softmax(att_map, dim=-2)

        return att_map

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)

        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)

        return x

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class HtrgGraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        self.proj_type1 = LinearXavier(in_dim, in_dim)
        self.proj_type2 = LinearXavier(in_dim, in_dim)

        # attention map
        self.att_proj = LinearXavier(in_dim, out_dim)
        self.att_projM = LinearXavier(in_dim, out_dim)

        self.att_weight11 = self._init_new_params(out_dim, 1)
        self.att_weight22 = self._init_new_params(out_dim, 1)
        self.att_weight12 = self._init_new_params(out_dim, 1)
        self.att_weightM = self._init_new_params(out_dim, 1)

        # project
        self.proj_with_att = LinearXavier(in_dim, out_dim)
        self.proj_without_att = LinearXavier(in_dim, out_dim)

        self.proj_with_attM = LinearXavier(in_dim, out_dim)
        self.proj_without_attM = LinearXavier(in_dim, out_dim)

        # batch norm
        self.bn = nn.BatchNorm1d(out_dim)

        # dropout for inputs
        self.input_drop = nn.Dropout(p=0.2)

        # activate
        self.act = nn.SELU(inplace=True)

        # temperature
        self.temp = 1.0
        if "temperature" in kwargs:
            self.temp = kwargs["temperature"]

    def forward(self, x1, x2, master=None):
        """
        x1  :(#bs, #node, #dim)
        x2  :(#bs, #node, #dim)
        """
        # print('x1',x1.shape)
        # print('x2',x2.shape)
        num_type1 = x1.size(1)
        num_type2 = x2.size(1)
        # print('num_type1',num_type1)
        # print('num_type2',num_type2)
        x1 = self.proj_type1(x1)
        # print('proj_type1',x1.shape)
        x2 = self.proj_type2(x2)
        # print('proj_type2',x2.shape)
        x = torch.cat([x1, x2], dim=1)
        # print('Concat x1 and x2',x.shape)

        if master is None:
            master = torch.mean(x, dim=1, keepdim=True)
            # print('master',master.shape)
        # apply input dropout
        x = self.input_drop(x)

        # derive attention map
        att_map = self._derive_att_map(x, num_type1, num_type2)
        # print('master',master.shape)
        # directional edge for master node
        master = self._update_master(x, master)
        # print('master',master.shape)
        # projection
        x = self._project(x, att_map)
        # print('proj x',x.shape)
        # apply batch norm
        x = self._apply_BN(x)
        x = self.act(x)

        x1 = x.narrow(1, 0, num_type1)
        # print('x1',x1.shape)
        x2 = x.narrow(1, num_type1, num_type2)
        # print('x2',x2.shape)
        return x1, x2, master

    def _update_master(self, x, master):

        att_map = self._derive_att_map_master(x, master)
        master = self._project_master(x, master, att_map)

        return master

    def _pairwise_mul_nodes(self, x):
        """
        Calculates pairwise multiplication of nodes.
        - for attention map
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, #dim)
        """

        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)

        return x * x_mirror

    def _derive_att_map_master(self, x, master):
        """
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, 1)
        """
        att_map = x * master
        att_map = torch.tanh(self.att_projM(att_map))

        att_map = torch.matmul(att_map, self.att_weightM)

        # apply temperature
        att_map = att_map / self.temp

        att_map = F.softmax(att_map, dim=-2)

        return att_map

    def _derive_att_map(self, x, num_type1, num_type2):
        """
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, 1)
        """
        att_map = self._pairwise_mul_nodes(x)
        # size: (#bs, #node, #node, #dim_out)
        att_map = torch.tanh(self.att_proj(att_map))
        # size: (#bs, #node, #node, 1)

        att_board = torch.zeros_like(att_map[:, :, :, 0]).unsqueeze(-1)

        att_board[:, :num_type1, :num_type1, :] = torch.matmul(
            att_map[:, :num_type1, :num_type1, :], self.att_weight11
        )
        att_board[:, num_type1:, num_type1:, :] = torch.matmul(
            att_map[:, num_type1:, num_type1:, :], self.att_weight22
        )
        att_board[:, :num_type1, num_type1:, :] = torch.matmul(
            att_map[:, :num_type1, num_type1:, :], self.att_weight12
        )
        att_board[:, num_type1:, :num_type1, :] = torch.matmul(
            att_map[:, num_type1:, :num_type1, :], self.att_weight12
        )

        att_map = att_board

        # apply temperature
        att_map = att_map / self.temp

        att_map = F.softmax(att_map, dim=-2)

        return att_map

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)

        return x1 + x2

    def _project_master(self, x, master, att_map):

        x1 = self.proj_with_attM(torch.matmul(att_map.squeeze(-1).unsqueeze(1), x))
        x2 = self.proj_without_attM(master)

        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)

        return x

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class GraphPool(nn.Module):
    def __init__(self, k: float, in_dim: int, p: Union[float, int]):
        super().__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = LinearXavier(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()
        self.in_dim = in_dim

    def forward(self, h):
        Z = self.drop(h)
        weights = self.proj(Z)
        scores = self.sigmoid(weights)
        new_h = self.top_k_graph(scores, h, self.k)

        return new_h

    def top_k_graph(self, scores, h, k):
        """
        args
        =====
        scores: attention-based weights (#bs, #node, 1)
        h: graph data (#bs, #node, #dim)
        k: ratio of remaining nodes, (float)
        returns
        =====
        h: graph pool applied data (#bs, #node', #dim)
        """
        _, n_nodes, n_feat = h.size()
        n_nodes = max(int(n_nodes * k), 1)
        _, idx = torch.topk(scores, n_nodes, dim=1)
        idx = idx.expand(-1, -1, n_feat)

        h = h * scores
        h = torch.gather(h, 1, idx)

        return h


class Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first

        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])
        self.conv1 = nn.Conv2d(
            in_channels=nb_filts[0],
            out_channels=nb_filts[1],
            kernel_size=(2, 3),
            padding=(1, 1),
            stride=1,
        )
        self.selu = nn.SELU(inplace=True)

        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(
            in_channels=nb_filts[1],
            out_channels=nb_filts[1],
            kernel_size=(2, 3),
            padding=(0, 1),
            stride=1,
        )

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(
                in_channels=nb_filts[0],
                out_channels=nb_filts[1],
                padding=(0, 1),
                kernel_size=(1, 3),
                stride=1,
            )

        else:
            self.downsample = False

    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.selu(out)
        else:
            out = x

        # print('out',out.shape)
        out = self.conv1(x)

        # print('aft conv1 out',out.shape)
        out = self.bn2(out)
        out = self.selu(out)
        # print('out',out.shape)
        out = self.conv2(out)
        # print('conv2 out',out.shape)

        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity
        # out = self.mp(out)
        return out


class W2VAASIST_Backbone(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        # AASIST parameters
        filts = [128, [1, 32], [32, 32], [32, 64], [64, 64]]
        gat_dims = [64, 32]
        self.output_dim = 5 * gat_dims[1]
        pool_ratios = [0.5, 0.5, 0.5, 0.5]
        temperatures = [2.0, 2.0, 100.0, 100.0]
        ####
        # create network wav2vec 2.0
        ####
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.first_bn1 = nn.BatchNorm2d(num_features=64)
        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        self.selu = nn.SELU(inplace=True)

        # RawNet2 encoder
        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
        )
        self.LL = LinearXavier(feature_dim, 128)

        self.attention = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 1)),
            nn.SELU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=(1, 1)),
        )
        # position encoding
        self.pos_S = nn.Parameter(torch.randn(1, 42, filts[-1][-1]))

        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))

        # Graph module
        self.GAT_layer_S = GraphAttentionLayer(
            filts[-1][-1], gat_dims[0], temperature=temperatures[0]
        )
        self.GAT_layer_T = GraphAttentionLayer(
            filts[-1][-1], gat_dims[0], temperature=temperatures[1]
        )
        # HS-GAL layer
        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2]
        )
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2]
        )
        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2]
        )
        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2]
        )
        # Graph pooling layers
        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hS1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        self.pool_hS2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

    def forward(self, x):

        # -------pre-trained Wav2vec model fine tunning ------------------------##
        x = x.squeeze(dim=1)
        x = x.transpose(1, 2)
        x = self.LL(x)
        x = x.transpose(1, 2)  # (bs,feat_out_dim,frame_number)
        x = x.unsqueeze(dim=1)  # add channel
        x = F.max_pool2d(x, (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)

        # RawNet2-based encoder
        x = self.encoder(x)
        x = self.first_bn1(x)
        x = self.selu(x)
        w = self.attention(x)

        # ------------SA for spectral feature-------------#
        w1 = F.softmax(w, dim=-1)
        m = torch.sum(x * w1, dim=-1)
        e_S = m.transpose(1, 2) + self.pos_S
        gat_S = self.GAT_layer_S(e_S)
        out_S = self.pool_S(gat_S)  # (#bs, #node, #dim)

        # ------------SA for temporal feature-------------#
        w2 = F.softmax(w, dim=-2)
        m1 = torch.sum(x * w2, dim=-2)
        e_T = m1.transpose(1, 2)

        # graph module layer
        gat_T = self.GAT_layer_T(e_T)
        out_T = self.pool_T(gat_T)

        # learnable master node
        master1 = self.master1.expand(x.size(0), -1, -1)
        master2 = self.master2.expand(x.size(0), -1, -1)

        # inference 1
        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(
            out_T, out_S, master=self.master1
        )

        out_S1 = self.pool_hS1(out_S1)
        out_T1 = self.pool_hT1(out_T1)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST12(
            out_T1, out_S1, master=master1
        )
        out_T1 = out_T1 + out_T_aug
        out_S1 = out_S1 + out_S_aug
        master1 = master1 + master_aug

        # inference 2
        out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(
            out_T, out_S, master=self.master2
        )
        out_S2 = self.pool_hS2(out_S2)
        out_T2 = self.pool_hT2(out_T2)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST22(
            out_T2, out_S2, master=master2
        )
        out_T2 = out_T2 + out_T_aug
        out_S2 = out_S2 + out_S_aug
        master2 = master2 + master_aug

        out_T1 = self.drop_way(out_T1)
        out_T2 = self.drop_way(out_T2)
        out_S1 = self.drop_way(out_S1)
        out_S2 = self.drop_way(out_S2)
        master1 = self.drop_way(master1)
        master2 = self.drop_way(master2)

        out_T = torch.max(out_T1, out_T2)
        out_S = torch.max(out_S1, out_S2)
        master = torch.max(master1, master2)

        # Readout operation
        T_max, _ = torch.max(torch.abs(out_T), dim=1)
        T_avg = torch.mean(out_T, dim=1)

        S_max, _ = torch.max(torch.abs(out_S), dim=1)
        S_avg = torch.mean(out_S, dim=1)
        last_hidden = torch.cat([T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1)
        last_hidden = self.drop(last_hidden)

        return last_hidden
    
class W2VAASIST(nn.Module):
    def __init__(self, feature_dim: int, num_labels: int, normalize_before_output=False):
        super().__init__()
        self.backbone = W2VAASIST_Backbone(feature_dim)
        
        self.normalize_before_output = normalize_before_output # When using ArcMarginProduct
        self.out_layer = LinearXavier(self.backbone.output_dim, num_labels, bias=False)
        
    def forward(self, x):
        last_hidden = self.backbone(x)
        
        if self.normalize_before_output:
            last_hidden_norm = F.normalize(last_hidden, p=2, dim=1)
            weight = F.normalize(self.out_layer.weight, p=2, dim=1)
            logits = F.linear(last_hidden_norm, weight)
            
        else:
            logits = self.out_layer(last_hidden)
            
        output = logits
        
        return last_hidden, output
        
class W2VAASIST_LCPN(nn.Module): # Local Classifier per Parent Node
    def __init__(self, feature_dim: int, label_mapping: Dict[int, int], normalize_before_output=False, K=None):
        super().__init__()
        self.K = K or 1
        self.backbone = W2VAASIST_Backbone(feature_dim)
        
        self.normalize_before_output = normalize_before_output # When using ArcMarginProduct
        
        self._prepare_labels(label_mapping)
        
        # STAGE 1: One head for the superclass prediction
        self.sup_layer = LinearXavier(self.backbone.output_dim, len(self.label_hierarchy) * self.K, bias=False)
        
        # STAGE 2: One head for each superclass's sublabel prediction (with > 1 sublabels)
        self.sub_layers = nn.ModuleDict({
            str(suplabel): LinearXavier(self.backbone.output_dim, len(sublabels) * self.K, bias=False) 
            for suplabel, sublabels in self.label_hierarchy.items()
            if len(sublabels) > 1
        })
        
    def _build_label_hierarchy(self, label_mapping: Dict[int, int]) -> None:
        # Build a hierarchy of labels based on global:superclass mapping
        label_hierarchy = {}
        for global_label, sup_label in label_mapping.items():
            if sup_label not in label_hierarchy:
                label_hierarchy[sup_label] = []
            label_hierarchy[sup_label].append(global_label)
                
        self.label_hierarchy = label_hierarchy # sup ID -> [global IDs]
        # e.g., {0: [0,1], 1: [2,3], 2: [4]} 
                   
    def _prepare_labels(self, label_mapping: Dict[int, int]) -> None:
        self._build_label_hierarchy(label_mapping)
        self.max_sublabels = max(len(sublabels) for sublabels in self.label_hierarchy.values())
        
        self.global_mapping = {}
        self.local_mapping = {}
        for suplabel, sublabels in self.label_hierarchy.items():
            
            for idx, global_label in enumerate(sublabels): # local sub ID, sub ID
                self.global_mapping[(suplabel, idx)] = global_label # (sup ID, local sub ID) -> global ID
                self.local_mapping[global_label] = (suplabel, idx) # global ID -> (sup ID, local sub ID)
                
    def get_global_label(self, supclass: int, subclass: int) -> int:
        # Returns the global label for a given superclass and subclass
        return self.global_mapping[(supclass, subclass)]
    
    def get_local_label(self, global_label: int) -> Tuple[int, int]:
        # Returns the superclass and subclass for a given global label
        return self.local_mapping[global_label]
    
    def forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def classify_supclass(self, embedding: torch.Tensor) -> torch.Tensor:
        if self.normalize_before_output:
            embedding_norm = F.normalize(embedding, p=2, dim=1)
            sup_weight = F.normalize(self.sup_layer.weight, p=2, dim=1)    
            sup_logits = F.linear(embedding_norm, sup_weight)
        else:
            sup_logits = self.sup_layer(embedding)
            
        return sup_logits
    
    def classify_subclass(self, embedding: torch.Tensor, supclass: int, sup_logits: torch.Tensor) -> torch.Tensor:
        sub_logits = torch.full(
            size=(embedding.size(0), self.max_sublabels * self.K),
            fill_value=float('-inf'),
            device=embedding.device
        )
        
        sup_id = str(supclass)
        if sup_id not in self.sub_layers: # No local classifier for this parent node
            # sub_logits[:, 0:self.K] = sup_logits[:, int(sup_id) * self.K:int(sup_id)+1 * self.K] # Copy superclass logits
            sub_logits[:, 0:self.K].fill_(1.0)
            return sub_logits
        
        if self.normalize_before_output:
            embedding_norm = F.normalize(embedding, p=2, dim=1)
            sub_weight = F.normalize(self.sub_layers[sup_id].weight, p=2, dim=1)
            logits = F.linear(embedding_norm, sub_weight)
        else:
            logits = self.sub_layers[sup_id](embedding)
            
        num_sublabels = logits.size(1)
        sub_logits[:, 0:num_sublabels] = logits

        return sub_logits
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        last_hidden = self.backbone(x)
        
        # STAGE 1: Predict superclass
        sup_logits = self.classify_supclass(last_hidden)
        
        # Aggregate K centers
        if self.K > 1:
            sup_logits_reduced = torch.reshape(sup_logits, (-1, len(self.label_hierarchy), self.K))
            sup_logits_reduced, _ = torch.max(sup_logits_reduced, axis=2)
        
            sup_preds = sup_logits_reduced.argmax(dim=1)
        else:
            sup_preds = sup_logits.argmax(dim=1)
        
        # STAGE 2: Predict sublabels for superclasses with > 1 sublabels
        # Padded tensor for sublabel logits
        sub_logits = torch.zeros((x.size(0), self.max_sublabels * self.K), device=x.device)
        for sup_label in sup_preds.unique():
            
            # Get indices of samples with the current superclass
            mask = (sup_preds == sup_label)
            if mask.sum() == 0:
                continue
            
            # Select corresponding embeddings
            embeddings_subset = last_hidden[mask]
            
            # Predict sublabels
            logits = self.classify_subclass(embeddings_subset, int(sup_label), sup_logits[mask])
            sub_logits[mask] = logits
            
        output = (sup_logits, sub_logits)
        return last_hidden, output
    
class W2VAASIST_LCL(nn.Module): # Local Classifier per Level
    def __init__(self, feature_dim: int, num_suplabels: int, num_labels: int, normalize_before_output=False):
        super().__init__()
        self.backbone = W2VAASIST_Backbone(feature_dim)
        
        self.normalize_before_output = normalize_before_output # When using ArcMarginProduct
        
        self.sup_layer = LinearXavier(self.backbone.output_dim, num_suplabels, bias=False)
        nn.init.xavier_uniform_(self.sup_layer.weight)
        self.sub_layer = LinearXavier(self.backbone.output_dim, num_labels, bias=False)
        nn.init.xavier_uniform_(self.sub_layer.weight)
        
    def forward(self, x):
        last_hidden = self.backbone(x)
        
        if self.normalize_before_output:
            last_hidden_norm = F.normalize(last_hidden, p=2, dim=1)
            sup_weight = F.normalize(self.sup_layer.weight, p=2, dim=1)
            sub_weight = F.normalize(self.sub_layer.weight, p=2, dim=1)
            
            sup_logits = F.linear(last_hidden_norm, sup_weight)
            sub_logits = F.linear(last_hidden_norm, sub_weight)
            
        else:
            sup_logits = self.sup_layer(last_hidden)
            sub_logits = self.sub_layer(last_hidden)
        
        output = (sup_logits, sub_logits)
        
        return last_hidden, output