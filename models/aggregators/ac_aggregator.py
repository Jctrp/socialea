from models.encoders.encoder import PredictionEncoder
import torch
import torch.nn as nn
from typing import Dict
from socialea.models.pe import PositionalEncoding

# Initialize device:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ACInteraction(PredictionEncoder):
    def __init__(self, args: Dict):
        super().__init__()

        self.hidden_size = args['hidden_size']
        self.future_steps = args['op_len']
        self.num_nodes = args['num_modes']

        self.q1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.k1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.v1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.tl_attn = nn.MultiheadAttention(self.hidden_size, num_heads=args['num_heads'])

        self.q2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.k2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.v2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.ta_attn = nn.MultiheadAttention(self.hidden_size, num_heads=args['num_heads'])
        self.mix = nn.Linear(self.hidden_size*3, self.hidden_size)

        # attn
        self.exp_attn = nn.MultiheadAttention(self.hidden_size, num_heads=args['num_heads'])
        self.social_attn = nn.MultiheadAttention(self.hidden_size, num_heads=args['num_heads'])
        self.node_attn = nn.MultiheadAttention(self.hidden_size, num_heads=1)

        # normalization
        self.norm1 = nn.LayerNorm(self.hidden_size)
        self.norm2 = nn.LayerNorm(self.hidden_size)
        self.norm3 = nn.LayerNorm(self.hidden_size)
        self.dropout1 = nn.Dropout(args['dropout'])
        self.dropout2 = nn.Dropout(args['dropout'])
        self.dropout3 = nn.Dropout(args['dropout'])

        # anchor-base queries
        self.anchor_queries = nn.Embedding(self.num_nodes, self.hidden_size)  # (K, D)
        self.pe = PositionalEncoding(self.hidden_size, 10000)

        self.pi = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 1))

    def forward(self, encodings: Dict) -> Dict:

        target_enc = encodings['target_agent_state']  # (1, B, D)
        nbr_enc = encodings['surrounding_agent_encoding'].transpose(0, 1)  # (N, B, D)
        lane_node_enc = encodings['context_encoding'].transpose(0, 1)  # (N, B, D)

        # T-L & T-A attention
        q1 = self.q1(target_enc)  # (1, B, D)
        k1 = self.k1(lane_node_enc)  # (N, B, D)
        v1 = self.v1(lane_node_enc)  # (N, B, D)
        lane_node_attn = self.tl_attn(q1, k1, v1)[0]  # (1, B, D)

        q2 = self.q2(target_enc)  # (1, B, D)
        k2 = self.k2(nbr_enc)  # (N, B, D)
        v2 = self.v2(nbr_enc)  # (N, B, D)
        nbr_attn = self.ta_attn(q2, k2, v2)[0]  # (1, B, D)
        target_enc = self.mix(torch.cat((target_enc, lane_node_attn, nbr_attn), dim=-1))  # (1, B, D)

        target_enc = target_enc.repeat(self.num_nodes, 1, 1)  # (K, B, D)
        target = target_enc.reshape(-1, self.hidden_size).unsqueeze(0)  # [1, K x B, D]

        anchor_queries = self.anchor_queries.weight.unsqueeze(1).repeat(1, target_enc.shape[1], 1)  # (K, B, D)

        # Anchor-based Queries
        mode_query_states = anchor_queries + target_enc  # (K, B, D)
        # mode_query_states = torch.zeros_like(anchor_queries)  # (K, B, D)
        pos = self.pe(mode_query_states)

        # Temporal Experience Attention
        exp_attn = self.exp_attn(mode_query_states, target_enc + pos, target_enc)[0]  # (K, B, D)
        mode_query_states = mode_query_states + self.dropout1(exp_attn)  # (K, B, D)
        mode_query_states = self.norm1(mode_query_states)  # (K, B, D)

        # Lane-node Attention
        lane_node_attn = self.node_attn(mode_query_states, lane_node_enc + pos, lane_node_enc)[0]
        mode_query_states = mode_query_states + self.dropout2(lane_node_attn)  # (K, B, D)
        mode_query_states = self.norm2(mode_query_states)  # (K, B, D)

        # Social Attention
        social_attn = self.social_attn(mode_query_states, nbr_enc + pos, nbr_enc)[0]  # (K, B, D)
        mode_query_states = mode_query_states + self.dropout3(social_attn)  # (K, B, D)
        query_states = self.norm3(mode_query_states)  # (K, B, D)

        mode_query_states = query_states.reshape(-1, self.hidden_size)  # [K x B, D]
        mode_query_states = mode_query_states.expand(self.future_steps, *mode_query_states.shape)  # [H, K x B, D]

        # pi
        pi = self.pi(torch.cat((query_states, target_enc), -1)).squeeze(-1).t()  # [B, K]

        encodings = {'mode_query_states': mode_query_states,
                     'target': target,
                     'pi': pi
                     }
        return encodings

