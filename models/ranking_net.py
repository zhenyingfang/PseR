import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from six.moves import map, zip


def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


class RankingNet(nn.Module):
    def __init__(self, args):
        super(RankingNet, self).__init__()
        self.shared_fcs = nn.Sequential(
            nn.Linear(8 * 2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU()
        )
        self.fc_cls = nn.ModuleList()
        for _ in range(args.stage_nums):
            self.fc_cls.append(nn.Linear(1024, 20))

        self.fc_ins = nn.ModuleList()
        for _ in range(args.stage_nums):
            self.fc_ins.append(nn.Linear(1024, 20))
        self.eps = 1e-6

    def forward(self, x, refine_stage):
        shared_features = self.shared_fcs(x)
        cls_score = self.fc_cls[refine_stage](shared_features)
        ins_score = self.fc_ins[refine_stage](shared_features)
        return cls_score, ins_score

    def gfocal_loss(self, p, q, w=1.0):
        l1 = (p - q) ** 2
        l2 = q * (p + self.eps).log() + (1 - q) * (1 - p + self.eps).log()
        return -(l1 * l2 * w).sum(dim=-1)

    def rk_loss_func(self, bag_cls_prob, bag_ins_outs, gt_label):
        gt_label = gt_label[0]
        bag_cls_prob = bag_cls_prob.sigmoid()
        B, N, C = bag_cls_prob.shape
        prob_cls = bag_cls_prob.unsqueeze(dim=-1)  # (B, N, C, 1)
        prob_ins = bag_ins_outs.reshape(B, N, C, -1)  # (B, N, C, 2/1)
        prob_ins = prob_ins.softmax(dim=1)
        prob_ins = F.normalize(prob_ins, dim=1, p=1)
        prob = (prob_cls * prob_ins).sum(dim=1)
        if prob.shape[-1] == 1:
            prob = prob.squeeze(dim=-1)
        prob = prob.clamp(0, 1)
        labels = self._expand_onehot_labels(gt_label, None, C)[0].float()
        loss = self.gfocal_loss(prob, labels.float())
        loss = loss.mean()
        return loss

    def neg_loss_func(self, neg_cls_score, neg_weights):
        num_neg, num_class = neg_cls_score.shape
        neg_cls_score = neg_cls_score.sigmoid()
        neg_labels = torch.full((num_neg, num_class), 0, dtype=torch.float32).to(neg_cls_score.device)
        loss_weights = 0.5
        neg_valid = neg_weights.reshape(num_neg, -1)
        neg_loss = self.gfocal_loss(neg_cls_score, neg_labels, neg_valid.float())
        neg_loss = loss_weights * neg_loss.mean()
        return neg_loss

    def _expand_onehot_labels(self, labels, label_weights, label_channels):
        bin_labels = labels.new_full((labels.size(0), label_channels), 0)
        inds = torch.nonzero(
            (labels >= 0) & (labels < label_channels), as_tuple=False).squeeze()
        if inds.numel() > 0:
            bin_labels[inds, labels[inds]] = 1

        if label_weights is None:
            bin_label_weights = None
        else:
            bin_label_weights = label_weights.view(-1, 1).expand(
                label_weights.size(0), label_channels)

        return bin_labels, bin_label_weights

    def merge_box(self, cls_scores, ins_scores, proposals_list, gt_labels, refine_scores):
        gt_labels = gt_labels[0]
        proposals_list = proposals_list[0]
        proposals_list = proposals_list.reshape(cls_scores.shape[0], cls_scores.shape[1], 2)
        cls_scores = cls_scores.sigmoid()
        ins_scores = ins_scores.softmax(dim=-2)
        ins_scores = F.normalize(ins_scores, dim=1, p=1)
        cls_scores = cls_scores
        dynamic_weight = (cls_scores * ins_scores)
        dynamic_weight = dynamic_weight[torch.arange(len(cls_scores)), :, gt_labels]
        cls_scores = cls_scores[torch.arange(len(cls_scores)), :, gt_labels]
        ins_scores = ins_scores[torch.arange(len(cls_scores)), :, gt_labels]
        # split batch
        batch_gt = [gt_labels.shape[0]]
        cls_scores = torch.split(cls_scores, batch_gt)
        ins_scores = torch.split(ins_scores, batch_gt)
        gt_labels = torch.split(gt_labels, batch_gt)
        dynamic_weight_list = torch.split(dynamic_weight, batch_gt)

        refine_scores = refine_scores.squeeze(2)
        refine_scores = torch.split(refine_scores, batch_gt)

        if not isinstance(proposals_list, list):
            proposals_list = torch.split(proposals_list, batch_gt)

        boxes, filtered_scores = multi_apply(
                                    self.merge_box_single,
                                    cls_scores,
                                    ins_scores,
                                    gt_labels,
                                    dynamic_weight_list,
                                    proposals_list,
                                    refine_scores
                                )

        return boxes[0]

    def merge_box_single(self, cls_score, ins_score, gt_label, dynamic_weight, proposals, refine_score):
        k = 8
        new_dynamic_weight = dynamic_weight * refine_score
        dynamic_weight_, idx = new_dynamic_weight.topk(k=k, dim=1)
        weight = dynamic_weight_.unsqueeze(2).repeat([1, 1, 2])
        weight = weight / (weight.sum(dim=1, keepdim=True) + 1e-8)
        filtered_boxes = proposals[torch.arange(proposals.shape[0]).unsqueeze(1), idx]
        boxes = (filtered_boxes * weight).sum(dim=1)
        filtered_scores = dict(cls_score=cls_score[torch.arange(proposals.shape[0]).unsqueeze(1), idx],
                                ins_score=ins_score[torch.arange(proposals.shape[0]).unsqueeze(1), idx],
                                dynamic_weight=dynamic_weight_)
        return boxes, filtered_scores
