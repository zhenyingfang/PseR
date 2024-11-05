import torch
import torch.nn as nn
import torch.nn.functional as F

class RefineNet(nn.Module):
    def __init__(self):
        super(RefineNet, self).__init__()
        self.shared_fcs = nn.Sequential(
            nn.Linear(8 * 2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU()
        )
        self.fc_iou = nn.Linear(1024, 1)
    
    def forward(self, x):
        shared_features = self.shared_fcs(x)

        B = shared_features.view(-1, 21, 1024)
        C = torch.mean(B, dim=1, keepdim=True)
        C = C.repeat(1, 21, 1)
        bag_features = C.view(-1, 1024)
        shared_features = shared_features + bag_features

        iou_score = self.fc_iou(shared_features)
        iou_score = iou_score.sigmoid()
        return iou_score

    def iou_loss_func(self, preds, gts):
        gts = gts.unsqueeze(1)
        iou_loss = F.smooth_l1_loss(preds, gts)
        return iou_loss
