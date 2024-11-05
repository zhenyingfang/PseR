import torch
import numpy as np
import random

def get_neg_pps(bkg_score, point_label):
    _, bkg_seed = select_seed(bkg_score, point_label)
    bkg_seed = bkg_seed.cpu().detach().numpy()
    segs = grouping(np.where(bkg_seed[0] > 0)[0])
    NP = []

    if len(segs) > 0:
        for seg in segs:
            if len(seg) == 1:
                NP.append([seg[0], seg[0]+1, -1])
            else:
                NP.append([seg[0], seg[-1], -1])

    NP = torch.from_numpy(np.array(NP))[:, :2]
    np_weights = torch.ones((NP.shape[0])).bool()
    return NP, np_weights

def grouping(arr):
    return np.split(arr, np.where(np.diff(arr) != 1)[0] + 1)

def select_seed(bkg_score, point_anno):
    point_anno_agnostic = point_anno.max(dim=2)[0]
    bkg_seed = torch.zeros_like(point_anno_agnostic)
    act_seed = point_anno.clone().detach()

    act_thresh = 0.1  
    bkg_thresh = 0.95

    for b in range(point_anno.shape[0]):
        act_idx = torch.nonzero(point_anno_agnostic[b]).squeeze(1)
        if len(act_idx) == 0:
            continue
        """ most left """
        if act_idx[0] > 0:
            bkg_score_tmp = bkg_score[b, :act_idx[0]]
            idx_tmp = bkg_seed[b, :act_idx[0]]
            idx_tmp[bkg_score_tmp >= bkg_thresh] = 1

            if idx_tmp.sum() >= 1:
                start_index = idx_tmp.nonzero().squeeze(1)[-1]
                idx_tmp[:start_index] = 1
            else:
                max_index = bkg_score_tmp.argmax(dim=0)
                idx_tmp[:max_index + 1] = 1
            """ pseudo action point selection """
            for j in range(act_idx[0] - 1, -1, -1):
                if bkg_score[b][j] <= act_thresh and bkg_seed[b][j] < 1:
                    act_seed[b, j] = act_seed[b, act_idx[0]]
                else:
                    break

        """ most right """
        if act_idx[-1] < (point_anno.shape[1] - 1):
            bkg_score_tmp = bkg_score[b, act_idx[-1] + 1:]
            idx_tmp = bkg_seed[b, act_idx[-1] + 1:]
            idx_tmp[bkg_score_tmp >= bkg_thresh] = 1

            if idx_tmp.sum() >= 1:
                start_index = idx_tmp.nonzero().squeeze(1)[0]
                idx_tmp[start_index:] = 1
            else:
                max_index = bkg_score_tmp.argmax(dim=0)
                idx_tmp[max_index:] = 1
            """ pseudo action point selection """
            for j in range(act_idx[-1] + 1, point_anno.shape[1]):
                if bkg_score[b][j] <= act_thresh and bkg_seed[b][j] < 1:
                    act_seed[b, j] = act_seed[b, act_idx[-1]]
                else:
                    break

        """ between two instances """
        for i in range(len(act_idx) - 1):
            if act_idx[i + 1] - act_idx[i] <= 1:
                continue

            bkg_score_tmp = bkg_score[b, act_idx[i] + 1:act_idx[i + 1]]
            idx_tmp = bkg_seed[b, act_idx[i] + 1:act_idx[i + 1]]
            idx_tmp[bkg_score_tmp >= bkg_thresh] = 1

            if idx_tmp.sum() >= 2:
                start_index = idx_tmp.nonzero().squeeze(1)[0]
                end_index = idx_tmp.nonzero().squeeze(1)[-1]
                idx_tmp[start_index + 1:end_index] = 1
            else:
                max_index = bkg_score_tmp.argmax(dim=0)
                idx_tmp[max_index] = 1
            """ pseudo action point selection """
            for j in range(act_idx[i] + 1, act_idx[i + 1]):
                if bkg_score[b][j] <= act_thresh and bkg_seed[b][j] < 1:
                    act_seed[b, j] = act_seed[b, act_idx[i]]
                else:
                    break
            for j in range(act_idx[i + 1] - 1, act_idx[i], -1):
                if bkg_score[b][j] <= act_thresh and bkg_seed[b][j] < 1:
                    act_seed[b, j] = act_seed[b, act_idx[i + 1]]
                else:
                    break

    return act_seed, bkg_seed
