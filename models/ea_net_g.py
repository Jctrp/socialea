import torch
import torch.nn as nn
from typing import Dict


class SocialCellLocal(nn.Module):
    def __init__(self, spatial_input, spatial_output, temporal_input, temporal_output):
        super(SocialCellLocal, self).__init__()

        self.feat = nn.Conv1d(spatial_input, spatial_output, 3, padding=1)
        self.highway_input = nn.Conv1d(spatial_input, spatial_output, 1, padding=0)
        self.tpcnn = nn.Conv1d(temporal_input, temporal_output, 3, padding=1)
        self.highway = nn.Conv1d(temporal_input, temporal_output, 1, padding=0)
        self.feat_act = nn.LeakyReLU()

    def forward(self, v):  # [H, K x B, D]
        v = v.permute(1, 2, 0)  # [K x B, D, H]
        v = self.feat_act(self.feat(v)) + self.highway_input(v)  # [K x B, D, H]
        v = v.permute(0, 2, 1)  # [K x B, H, D]
        v = self.tpcnn(v) + self.highway(v)  # [K x B, H, D]
        v = v.permute(1, 0, 2)  # [H, K x B, D]
        return v


class SocialCellGlobal(nn.Module):
    def __init__(self,
                 spatial_input, spatial_output, temporal_input, temporal_output):
        super(SocialCellGlobal, self).__init__()

        self.feat = nn.Conv2d(spatial_input, spatial_output, 3, padding=1)
        self.highway_input = nn.Conv2d(spatial_input, spatial_output, 1, padding=0)

        self.global_w = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.local_w = nn.Parameter(torch.zeros(1), requires_grad=True)

        # Local Stream
        self.conv_local = SocialCellLocal(spatial_input=spatial_input,
                                          spatial_output=spatial_output,
                                          temporal_input=temporal_input,
                                          temporal_output=temporal_output)

        self.tpcnn = nn.Conv2d(temporal_input, temporal_output, 3, padding=1)
        self.highway = nn.Conv2d(temporal_input, temporal_output, 1, padding=0)
        self.feat_act = nn.LeakyReLU()

    def forward(self, global_embed):
        v_shape = global_embed.shape  # [H, K x B, D]
        v_local = self.conv_local(global_embed)  # [H, K x B, D]
        v = global_embed.reshape(10, v_shape[2], v_shape[0], -1)  # [K, D, H, B]
        v = self.feat_act(self.feat(v)) + self.highway_input(v)  # [K, D, H, B]
        v = v.permute(0, 2, 1, 3)  # [K, H, D, B]
        v = self.tpcnn(v) + self.highway(v)  # [K, H, D, B]
        v = v.permute(1, 0, 3, 2).reshape(v_shape[0], v_shape[1], v_shape[2])  # [H, K x B, D]
        global_embed_n = v_local * self.local_w + v * self.global_w  # [H, K x B, D]
        return global_embed_n


class EAMLP(nn.Module):
    def __init__(self, dim, num_modes):
        super().__init__()
        self.lin_d = nn.Sequential(nn.Linear(dim, dim),
                                   nn.LayerNorm(dim),
                                   nn.ReLU(inplace=True))
        self.lin_kd = nn.Sequential(nn.Linear(dim*num_modes, dim*num_modes),
                                    nn.LayerNorm(dim * num_modes),
                                    nn.ReLU(inplace=True))
        self.num_modes = num_modes
        self.dim = dim

        self.w1 = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.w2 = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, global_embed):  # [H, K x B, D]
        x_d = self.lin_d(global_embed) + global_embed
        x_kd = global_embed.view(global_embed.shape[0], -1, self.num_modes * self.dim)  # [H, B, K*D]
        x_kd = self.lin_kd(x_kd) + x_kd
        x_kd = x_kd.view(global_embed.shape[0], -1, global_embed.shape[-1])
        x = self.w1 * x_d + self.w2 * x_kd
        return x

