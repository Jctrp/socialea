from models.encoders.encoder import PredictionEncoder
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from typing import Dict


# Initialize device:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ACEncoder(PredictionEncoder):
    def __init__(self, args: Dict):
        super().__init__()
        # encoding
        self.t_mlp = MLP(args['target_agent_feat_size'], args['emb_size'])
        self.n_mlp = MLP(args['nbr_feat_size'] + 1, args['emb_size'])
        self.l_mlp = MLP(args['node_feat_size'], args['emb_size'])
        encoder_layer = nn.TransformerEncoderLayer(d_model=args['emb_size'],
                                                   nhead=args["num_heads"],
                                                   dim_feedforward=args['emb_size'],
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=args["num_layers"])
        self.t_encoder = nn.GRU(args['emb_size'], args['emb_size'], batch_first=True)
        self.n_encoder = nn.GRU(args['emb_size'], args['emb_size'], batch_first=True)
        self.l_encoder = nn.GRU(args['emb_size'], args['emb_size'], batch_first=True)

        # attention
        self.query_emb = nn.Linear(args['emb_size'], args['emb_size'])
        self.key_emb = nn.Linear(args['emb_size'], args['emb_size'])
        self.val_emb = nn.Linear(args['emb_size'], args['emb_size'])
        self.a_n_att = nn.MultiheadAttention(args['emb_size'], num_heads=1)
        self.mix = nn.Linear(args['emb_size'] * 2, args['emb_size'])
        self.gat = nn.ModuleList([GAT(args['emb_size'], args['emb_size'])
                                  for _ in range(args['num_gat_layers'])])

    def forward(self, inputs: Dict) -> Dict:
        # target agent
        target_agent_feats = inputs['target_agent_representation']  # (32, 5, 5)
        target_agent_feats = self.t_mlp(target_agent_feats)  # (32, 5, 32)
        target = self.transformer_encoder(target_agent_feats) + target_agent_feats  # (32, 5, 32)
        target_state = self.t_encoder(target)[1]  # (1, 32, 32)

        # surrounding agents
        nbr_vehicle_feats = inputs['surrounding_agent_representation']['vehicles']  # [32, 84, 5, 5]
        nbr_vehicle_feats = torch.cat((nbr_vehicle_feats, torch.zeros_like(nbr_vehicle_feats[:, :, :, 0:1])), dim=-1)
        nbr_vehicle_masks = inputs['surrounding_agent_representation']['vehicle_masks']  # [32, 84, 5, 5]
        nbr_vehicle_embedding = self.n_mlp(nbr_vehicle_feats)  # [32, 84, 5, 32]
        nbr_vehicle_enc = self.variable_size_gru_encode(nbr_vehicle_embedding, nbr_vehicle_masks, self.n_encoder)
        nbr_ped_feats = inputs['surrounding_agent_representation']['pedestrians']  # [32, 77, 5, 5]
        nbr_ped_feats = torch.cat((nbr_ped_feats, torch.ones_like(nbr_ped_feats[:, :, :, 0:1])), dim=-1)
        nbr_ped_masks = inputs['surrounding_agent_representation']['pedestrian_masks']  # [32, 77, 5, 5]
        nbr_ped_embedding = self.n_mlp(nbr_ped_feats)
        nbr_ped_enc = self.variable_size_gru_encode(nbr_ped_embedding, nbr_ped_masks, self.n_encoder)
        nbr_enc = torch.cat((nbr_vehicle_enc, nbr_ped_enc), dim=1)  # [32, 161, 32]

        # surrounding avgpool
        #nbr_avgpool = self.avgpool(nbr_enc.transpose(0, 1))  # [161, 32, 1]
        #nbr_enc = self.avgpool_mix(torch.cat((nbr_avgpool.transpose(0, 1), nbr_enc), -1))

        # lane nodes
        lane_node_feats = inputs['map_representation']['lane_node_feats']  # (32, 164, 20, 6)
        lane_node_masks = inputs['map_representation']['lane_node_masks']
        lane_node_embedding = self.l_mlp(lane_node_feats)
        lane_node_enc = self.variable_size_gru_encode(lane_node_embedding, lane_node_masks, self.l_encoder)

        # A-L Muti-Head Attention
        queries = self.query_emb(lane_node_enc).permute(1, 0, 2)  # [164, 32, 32]
        keys = self.key_emb(nbr_enc).permute(1, 0, 2)  # [161, 32, 32]
        vals = self.val_emb(nbr_enc).permute(1, 0, 2)  # [161, 32, 32]
        attn_masks = torch.cat((inputs['agent_node_masks']['vehicles'],
                                inputs['agent_node_masks']['pedestrians']), dim=2)  # [32, 164, 161]
        attn_op, _ = self.a_n_att(queries, keys, vals, attn_mask=attn_masks)  # [161, 32, 32]
        attn_op = attn_op.permute(1, 0, 2)  # [32, 161, 32]
        lane_node_enc = self.mix(torch.cat((lane_node_enc, attn_op), dim=2))

        # L-L Self-Attention
        adj_mat = self.build_adj_mat(inputs['map_representation']['s_next'], inputs['map_representation']['edge_type'])
        for gat_layer in self.gat:
            lane_node_enc += gat_layer(lane_node_enc, adj_mat)

        encodings = {'target_agent_state': target_state,  # (1, 32, 32)
                     'surrounding_agent_encoding': nbr_enc,  # (32, 161, 32)
                     'context_encoding': lane_node_enc,  # (32, 164, 32)
                     }

        return encodings

    @staticmethod
    def variable_size_gru_encode(feat_embedding: torch.Tensor, masks: torch.Tensor, gru: nn.GRU) -> torch.Tensor:
        """
        Returns GRU encoding for a batch of inputs where each sample in the batch is a set of a variable number
        of sequences, of variable lengths.
        """

        # Form a large batch of all sequences in the batch
        masks_for_batching = ~masks[:, :, :, 0].bool()  # [32, 164, 20]
        masks_for_batching = masks_for_batching.any(dim=-1).unsqueeze(2).unsqueeze(3)  # [32, 164, 1, 1]
        feat_embedding_batched = torch.masked_select(feat_embedding, masks_for_batching)
        feat_embedding_batched = feat_embedding_batched.view(-1, feat_embedding.shape[2], feat_embedding.shape[3])

        # Pack padded sequences
        seq_lens = torch.sum(1 - masks[:, :, :, 0], dim=-1)  # [32, 164]
        seq_lens_batched = seq_lens[seq_lens != 0].cpu()  # [1528,]
        if len(seq_lens_batched) != 0:
            feat_embedding_packed = pack_padded_sequence(feat_embedding_batched, seq_lens_batched,
                                                         batch_first=True, enforce_sorted=False)

            # Encode
            _, encoding_batched = gru(feat_embedding_packed)
            # _: {(22875, 32)},{(20, )},{(1528,)},{(1528,)}
            # encoding_batched: [1, 1528, 32] D*num_layers, batch_size, hidden_out
            encoding_batched = encoding_batched.squeeze(0)  # [1528, 32]

            masks_for_scattering = masks_for_batching.squeeze(3).repeat(1, 1, encoding_batched.shape[-1])
            # [32, 164, 1, 1] -> [32, 164, 1] -> [32, 164, 32]
            encoding = torch.zeros(masks_for_scattering.shape, device=device)  # [32, 164, 32]
            encoding = encoding.masked_scatter(masks_for_scattering, encoding_batched)

        else:
            batch_size = feat_embedding.shape[0]
            max_num = feat_embedding.shape[1]
            hidden_state_size = gru.hidden_size
            encoding = torch.zeros((batch_size, max_num, hidden_state_size), device=device)

        return encoding

    @staticmethod
    def build_adj_mat(s_next, edge_type):
        """
        Builds adjacency matrix for GAT layers.
        """
        batch_size = s_next.shape[0]  # 32
        max_nodes = s_next.shape[1]  # 164
        max_edges = s_next.shape[2]  # 15
        adj_mat = torch.diag(torch.ones(max_nodes, device=device)).unsqueeze(0).repeat(batch_size, 1, 1).bool()
        # [32, 164, 164]

        dummy_vals = torch.arange(max_nodes, device=device).unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, max_edges)
        # 164 -> [1, 164, 1] -> [32, 164, 15]
        dummy_vals = dummy_vals.float()
        s_next[edge_type == 0] = dummy_vals[edge_type == 0]
        batch_indices = torch.arange(batch_size).unsqueeze(1).unsqueeze(2).repeat(1, max_nodes, max_edges)
        src_indices = torch.arange(max_nodes).unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, max_edges)
        # 164 -> [1, 164, 1] -> [32, 164, 15]
        adj_mat[batch_indices[:, :, :-1], src_indices[:, :, :-1], s_next[:, :, :-1].long()] = True
        adj_mat = adj_mat | torch.transpose(adj_mat, 1, 2)

        return adj_mat


class GAT(nn.Module):
    """
    GAT layer for aggregating local context at each lane node. Uses scaled dot product attention using pytorch's
    multihead attention module.
    """

    def __init__(self, in_channels, out_channels):
        """
        Initialize GAT layer.
        :param in_channels: size of node encodings
        :param out_channels: size of aggregated node encodings
        """
        super().__init__()
        self.query_emb = nn.Linear(in_channels, out_channels)
        self.key_emb = nn.Linear(in_channels, out_channels)
        self.val_emb = nn.Linear(in_channels, out_channels)
        self.att = nn.MultiheadAttention(out_channels, 1)

    def forward(self, node_encodings, adj_mat):
        """
        Forward pass for GAT layer
        :param node_encodings: Tensor of node encodings, shape [batch_size, max_nodes, node_enc_size]
        :param adj_mat: Bool tensor, adjacency matrix for edges, shape [batch_size, max_nodes, max_nodes]
        :return:
        """
        queries = self.query_emb(node_encodings.permute(1, 0, 2))
        keys = self.key_emb(node_encodings.permute(1, 0, 2))
        vals = self.val_emb(node_encodings.permute(1, 0, 2))
        att_op, _ = self.att(queries, keys, vals, attn_mask=~adj_mat)

        return att_op.permute(1, 0, 2)


class MLP(nn.Module):
    def __init__(self, hidden_size, out_features=None):
        super(MLP, self).__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear = nn.Linear(hidden_size, out_features)
        self.layer_norm = nn.LayerNorm(out_features)
        self.relu = nn.ReLU()

    def forward(self, hidden_states):
        hidden_states = self.linear(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.relu(hidden_states)
        return hidden_states


