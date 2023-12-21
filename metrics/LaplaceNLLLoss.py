from metrics.metric import Metric
from typing import Dict, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class LaplaceNLLLoss(Metric):
    """
    Negative log likelihood loss for ground truth goal nodes under predicted goal log-probabilities.
    """
    def __init__(self, args: Dict):
        self.name = 'LaplaceNLLLoss'
        self.loss = nn.SmoothL1Loss(reduction='mean')

    def compute(self, predictions: Dict, ground_truth: Union[torch.Tensor, Dict]) -> torch.Tensor:
        out_mu = predictions['traj']  # (32, 10, 12, 2)
        out_sigma = predictions['scale']  # (32, 10, 12, 2)
        gt = ground_truth['traj']  # (32, 12, 2)
        y = gt.repeat(10, 1, 1, 1).transpose(0, 1)
        # (32, 12, 2) -> (10, 32, 12, 2) -> (32, 10, 12, 2)
        out_pi = predictions['probs']  # (32, 10)
        pred = torch.cat((out_mu, out_sigma), dim=-1)  # (32, 10, 12, 4)
        l2_norm = torch.norm(out_mu - y, p=2, dim=-1)  # (32, 10, 12)
        l2_norm = l2_norm.sum(dim=-1)  # (32, 10)
        best_mode = l2_norm.argmin(dim=1)  # (32, 1)
        pred_best = pred[torch.arange(pred.shape[0]), best_mode]  # (32, 12, 4)
        soft_target = F.softmax(-l2_norm / pred.shape[2], dim=1).detach()  # (32, 10)

        loc, scale = pred_best.chunk(2, dim=-1)
        # loc: (32, 12, 2), scale: (32, 12, 2), gt: (32, 12, 2)
        scale = scale.clone()
        with torch.no_grad():
            scale.clamp_(min=1e-6)
        loss = 0
        for b in range(loc.shape[0]):
            # s_l1 = self.loss(loc[b], gt[b])
            nll = torch.log(2 * scale[b]) + torch.abs(gt[b] - loc[b]) / scale[b]  # (12, 2)
            nll_mean = nll.mean()  # (1)

            cross_entropy = torch.sum(-soft_target[b] * F.log_softmax(out_pi[b], dim=-1), dim=-1)  # (1)

            loss += nll_mean + cross_entropy * 0.5
        loss_total = loss / loc.shape[0]

        return loss_total
