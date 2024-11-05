import numpy as np
import torch

def proposal2roi(c_proposals):
    B = 1
    N = c_proposals.shape[0]
    c_proposals = c_proposals.unsqueeze(0)
    batch_ind = torch.arange(0, B).view((B, 1, 1)).to(c_proposals.device)
    batch_ind = batch_ind.repeat(1, N, 1)
    rois_abs = torch.cat((batch_ind, c_proposals), dim=2)
    rois = rois_abs.view((B*N, 3)).float().detach()
    return rois

def refine_proposals(pseudo_boxes):
    base_ratios = [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5]
    shake_ratios = [0.1]
    # >> base ratios
    proposal_list = []
    proposals_valid_list = []
    for i in range(pseudo_boxes.shape[0]):
        pps = []
        base_boxes = pseudo_boxes[i]
        for ratio_w in base_ratios:
            base_boxes_ = interval_xx_to_cxw(base_boxes)
            base_boxes_[:, 1] *= ratio_w
            base_boxes_ = interval_cxw_to_xx(base_boxes_)
            pps.append(base_boxes_.unsqueeze(1))
        pps_old = torch.cat(pps, dim=1)
        if shake_ratios is not None:
            pps_new = []

            pps_new.append(pps_old.reshape(*pps_old.shape[0:2], -1, 2))
            for ratio in shake_ratios:
                pps = interval_xx_to_cxw(pps_old)
                pps_center = pps[:, :, :1]
                pps_wh = pps[:, :, 1:2]
                pps_x_l = pps_center[:, :, 0] - ratio * pps_wh[:, :, 0]
                pps_x_r = pps_center[:, :, 0] + ratio * pps_wh[:, :, 0]
                pps_center = torch.stack([pps_x_l.unsqueeze(-1), pps_x_r.unsqueeze(-1)], dim=2)
                pps_wh = pps_wh.unsqueeze(2).expand(pps_center.shape)
                pps = torch.cat([pps_center, pps_wh], dim=-1)
                pps = pps.reshape(pps.shape[0], -1, 2)
                pps = interval_cxw_to_xx(pps)
                pps_new.append(pps.reshape(*pps_old.shape[0:2], -1, 2))
            pps_new = torch.cat(pps_new, dim=2)

        proposal_list.append(pps_new.reshape(-1, 2))

    return proposal_list

def interval_xx_to_cxw(bbox):
    """Convert interval coordinates from (x1, x2) to (cx, w).

    Args:
        bbox (Tensor): Shape (n, 2) for intervals.

    Returns:
        Tensor: Converted intervals.
    """
    x1, x2 = bbox.split((1, 1), dim=-1)
    bbox_new = [(x1 + x2) / 2, (x2 - x1)]
    return torch.cat(bbox_new, dim=-1)

def interval_cxw_to_xx(bbox):
    """Convert interval coordinates from (cx, w) to (x1, x2).

    Args:
        bbox (Tensor): Shape (n, 2) for intervals.

    Returns:
        Tensor: Converted intervals.
    """
    cx, w = bbox.split((1, 1), dim=-1)
    bbox_new = [(cx - 0.5 * w), (cx + 0.5 * w)]
    return torch.cat(bbox_new, dim=-1)
