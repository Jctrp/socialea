from models.decoders.decoder import PredictionDecoder
import torch
import torch.nn as nn
from typing import Dict, Union
import torch.nn.functional as F
from models.ea_net_g import SocialCellGlobal, EAMLP

# Initialize device:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class QueryTr(PredictionDecoder):
    def __init__(self, args):
        super().__init__()

        self.num_modes = args['num_modes']
        self.future_steps = args['op_len']
        self.hidden_size = args['hidden_size']
        self.min_scale = args['min_scale']

        self.decoder = nn.GRU(input_size=self.hidden_size,
                              hidden_size=self.hidden_size,
                              num_layers=1,
                              bias=True,
                              batch_first=False,
                              dropout=0,
                              bidirectional=False)

        # Laplace MDNs
        self.loc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 2))
        self.scale = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 2))

        # k modes interaction
        self.lvm = nn.Sequential(
            nn.Linear(self.hidden_size + 8, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True))
        self.ea_net = SocialCellGlobal(self.hidden_size, self.hidden_size, self.future_steps, self.future_steps)
        self.aggr_embed = nn.Sequential(
            nn.Linear(self.hidden_size*2, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True))

    def forward(self, inputs: Union[Dict, torch.Tensor]) -> Dict:
        mode_query_states = inputs['mode_query_states']  # [H, K x B, D]
        target = inputs['target']  # [1, K x B, D]
        pi = inputs['pi']  # [B, K]

        target_ = target.repeat(self.future_steps, 1, 1)  # [H, K x B, D]
        z = torch.randn(target_.shape[0], target_.shape[1], 8, device=device)
        target_noise = self.lvm(torch.cat((target_, z), dim=-1))  # [H, K x B, D]
        target_noise = self.ea_net(target_noise)  # [H, K x B, D]
        aggr = self.aggr_embed(torch.cat((mode_query_states, target_noise), -1))  # [H, K x B, D]

        # Decoder
        out = self.decoder(aggr, target)[0]  # [H, K x B, D]
        out = out.transpose(0, 1)  # [K x B, H, D]

        # get loc and scale of Laplace Distribution
        loc = self.loc(out)  # [K x B, H, 2]
        scale = F.elu_(self.scale(out), alpha=1.0) + 1.0 + self.min_scale  # [K x B, H, 2]
        loc = loc.view(self.num_modes, -1, self.future_steps, 2)  # [K x B, H, 2]
        scale = scale.view(self.num_modes, -1, self.future_steps, 2)  # [K, B, H, 2]

        loc, scale = loc.transpose(0, 1), scale.transpose(0, 1)  # [B, K, H, 2]
        predictions = {'traj': loc, 'scale': scale, 'probs': pi}

        return predictions





