import numpy as np
import torch

def get_proposal_bags(seed_proposals, w):
    base_ratios = [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5]
    shake_ratios = [0.1]
    proposal_list = []
    
    for i in range(seed_proposals.shape[0]):
        pps = []
        base_boxes = seed_proposals[i]
        
        # Apply base ratios to width
        for ratio_w in base_ratios:
            base_boxes_ = interval_xx_to_cxw(base_boxes)
            base_boxes_[:, 1] *= ratio_w  # Adjust width by ratio
            base_boxes_ = interval_cxw_to_xx(base_boxes_)
            pps.append(base_boxes_.unsqueeze(1))
        
        # Combine all proposals along the second dimension
        pps_old = torch.cat(pps, dim=1)
        
        if shake_ratios is not None:
            pps_new = [pps_old.reshape(*pps_old.shape[0:2], -1, 2)]
            
            for ratio in shake_ratios:
                # Convert intervals and adjust centers
                pps = interval_xx_to_cxw(pps_old)
                pps_center = pps[:, :, :1]
                pps_wh = pps[:, :, 1:2]
                
                # Shake the center points
                pps_x_l = pps_center[:, :, 0] - ratio * pps_wh[:, :, 0]
                pps_x_r = pps_center[:, :, 0] + ratio * pps_wh[:, :, 0]
                
                # Reshape centers and widths, combine them
                pps_center = torch.stack([pps_x_l.unsqueeze(-1), pps_x_r.unsqueeze(-1)], dim=2)
                pps_wh = pps_wh.unsqueeze(2).expand(pps_center.shape)
                pps = torch.cat([pps_center, pps_wh], dim=-1)
                pps = pps.reshape(pps.shape[0], -1, 2)
                pps = interval_cxw_to_xx(pps)
                
                pps_new.append(pps.reshape(*pps_old.shape[0:2], -1, 2))
            
            pps_new = torch.cat(pps_new, dim=2)
        
        proposal_list.append(pps_new.reshape(-1, 2))
    
    all_refine_nums = 3 * len(shake_ratios) * len(base_ratios)
    pse_iou_labels = []
    for i in range(seed_proposals[0].shape[0]):
        tmp_seed = seed_proposals[0, i, ...].unsqueeze(0)
        tmp_proposal = proposal_list[0][i*all_refine_nums:(i+1)*all_refine_nums, ...]
        pse_iou_label = wrapper_segment_iou_torch(tmp_seed, tmp_proposal)
        pse_iou_labels.append(pse_iou_label[:, 0])

    pse_iou_labels = torch.cat(pse_iou_labels)
    neg_proposals, neg_weights = gen_negative_proposals(proposal_list[0], w)

    return proposal_list[0], neg_proposals, neg_weights, pse_iou_labels


def gen_negative_proposals(aug_generate_proposals, w):
    num_neg_gen = 500
    pos_box = aug_generate_proposals
    x1 = -0.2 * w + torch.rand(num_neg_gen) * (1.2 * w)
    x2 = x1 + torch.rand(num_neg_gen) * (1.2 * w - x1)
    neg_bboxes = torch.stack([x1, x2], dim=1).to(pos_box.device)
    iou = wrapper_segment_iou_torch(pos_box.detach(), neg_bboxes.detach())
    iou = iou.to(pos_box.device)
    neg_weight = ((iou < 0.3).sum(dim=1) == iou.shape[1])

    return neg_bboxes, neg_weight

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

def segment_iou_torch(target_segment, candidate_segments):
    """Compute the temporal intersection over union between a
    target segment and all the test segments.

    Parameters
    ----------
    target_segment : 1d tensor
        Temporal target segment containing [starting, ending] times.
    candidate_segments : 2d tensor
        Temporal candidate segments containing N x [starting, ending] times.

    Outputs
    -------
    tiou : 1d tensor
        Temporal intersection over union score of the N's candidate segments.
    """
    tt1 = torch.max(target_segment[0], candidate_segments[:, 0])
    tt2 = torch.min(target_segment[1], candidate_segments[:, 1])
    # Intersection including Non-negative overlap score.
    segments_intersection = torch.clamp(tt2 - tt1, min=0)
    # Segment union.
    segments_union = (candidate_segments[:, 1] - candidate_segments[:, 0]) + \
                     (target_segment[1] - target_segment[0]) - segments_intersection
    # Compute overlap as the ratio of the intersection
    # over union of two segments.
    tIoU = segments_intersection / segments_union
    return tIoU

def wrapper_segment_iou_torch(target_segments, candidate_segments):
    """Compute intersection over union between segments

    Parameters
    ----------
    target_segments : tensor
        2-dim tensor in format [m x 2:=[init, end]]
    candidate_segments : tensor
        2-dim tensor in format [n x 2:=[init, end]]
    
    Outputs
    -------
    tiou : tensor
        2-dim tensor [n x m] with IOU ratio.
    """
    if candidate_segments.dim() != 2 or target_segments.dim() != 2:
        raise ValueError('Dimension of arguments is incorrect')

    n, m = candidate_segments.size(0), target_segments.size(0)
    tiou = torch.empty((n, m))
    for i in range(m):
        tiou[:, i] = segment_iou_torch(target_segments[i, :], candidate_segments)

    return tiou

