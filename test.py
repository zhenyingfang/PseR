import torch
from torch.utils import data
import numpy as np
import utils.model_utils as utils
import os
import json

from tqdm import tqdm
from log import log_evaluate

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

@torch.no_grad()
def S_test(net, args, test_loader, logger, step, test_info, subset='test'):
    net.eval()
    snippet_result = {}
    snippet_result['version'] = 'VERSION 1.3'
    snippet_result['results'] = {}
    snippet_result['external_data'] = {'used': True, 'details': 'Features from I3D Network'}
    if subset == 'train':
        snippet_result['bkg_score'] = {}

    num_correct = 0.
    num_total = 0.
    for sample in tqdm(test_loader):
        _data, _vid_label, _vid_name, _vid_len, _vid_duration = sample['data'], sample['vid_label'], sample['vid_name'], sample['vid_len'], sample['vid_duration']
        outputs = net(_data.to(args.device))
        _vid_score, _cas_fuse = outputs['vid_score'], outputs['cas_fuse']
        for b in range(_data.shape[0]):
            vid_name = _vid_name[b]
            vid_len = _vid_len[b].item()
            vid_duration = _vid_duration[b].item()
            # >> caculate video-level prediction
            label_np = _vid_label[b].unsqueeze(0).numpy()
            score_np = _vid_score[b].cpu().numpy()
            pred_np = np.zeros_like(score_np)
            pred_np[np.where(score_np < args.class_thresh)] = 0
            pred_np[np.where(score_np >= args.class_thresh)] = 1
            if pred_np.sum() == 0:
                pred_np[np.argmax(score_np)] = 1
            correct_pred = np.sum(label_np == pred_np, axis=1)
            num_correct += np.sum((correct_pred == args.num_class).astype(np.float32))
            num_total += correct_pred.shape[0]

            # >> post-process
            cas_fuse = _cas_fuse[b]
            num_segments = _data[b].shape[0]
            # class-specific score
            cas_S = cas_fuse[:, :-1]
            pred = np.where(score_np >= args.class_thresh)[0]
            if len(pred) == 0:
                pred = np.array([np.argmax(score_np)])
            cas_pred = cas_S.cpu().numpy()[:, pred]   
            cas_pred = np.reshape(cas_pred, (num_segments, -1, 1))
            cas_pred = utils.upgrade_resolution(cas_pred, args.scale)
            # class-agnostic score
            agnostic_score = 1 - cas_fuse[:, -1].unsqueeze(1)
            agnostic_score = agnostic_score.expand((-1, args.num_class))
            agnostic_score = agnostic_score.cpu().numpy()[:, pred]
            agnostic_score = np.reshape(agnostic_score, (num_segments, -1, 1))
            agnostic_score = utils.upgrade_resolution(agnostic_score, args.scale)

            # >> save output
            if subset == 'train':
                snippet_result['bkg_score'][vid_name] = cas_fuse[:, -1].cpu().numpy()
            
            # >> generate proposals
            proposal_dict = {}
            for i in range(len(args.act_thresh_cas)):
                cas_temp = cas_pred.copy()
                zero_location = np.where(cas_temp[:, :, 0] < args.act_thresh_cas[i])
                cas_temp[zero_location] = 0

                seg_list = []
                for c in range(len(pred)):
                    pos = np.where(cas_temp[:, c, 0] > 0)
                    seg_list.append(pos)
                proposals = utils.get_proposal_oic(args, seg_list, cas_temp, score_np, pred, vid_len, num_segments, vid_duration)
                for i in range(len(proposals)):
                    class_id = proposals[i][0][2]
                    if class_id not in proposal_dict.keys():
                        proposal_dict[class_id] = []
                    proposal_dict[class_id] += proposals[i]

            for i in range(len(args.act_thresh_agnostic)):
                cas_temp = cas_pred.copy()
                agnostic_score_temp = agnostic_score.copy()
                zero_location = np.where(agnostic_score_temp[:, :, 0] < args.act_thresh_agnostic[i])
                agnostic_score_temp[zero_location] = 0

                seg_list = []
                for c in range(len(pred)):
                    pos = np.where(agnostic_score_temp[:, c, 0] > 0)
                    seg_list.append(pos)
                proposals = utils.get_proposal_oic(args, seg_list, cas_temp, score_np, pred, vid_len, num_segments, vid_duration)
                for i in range(len(proposals)):
                    class_id = proposals[i][0][2]
                    if class_id not in proposal_dict.keys():
                        proposal_dict[class_id] = []
                    proposal_dict[class_id] += proposals[i]

            if args.mode == 'train' or args.mode == 'infer':
                final_proposals = utils.post_process(args, vid_name, proposal_dict, test_loader)
            else:
                final_proposals = []
                for class_id in proposal_dict.keys():
                    temp_proposal = proposal_dict[class_id]
                    final_proposals += temp_proposal
                final_proposals = utils.result2json(args, final_proposals)

            snippet_result['results'][vid_name] = final_proposals

    json_path = os.path.join(args.output_path_s1, 'snippet_result_{}.json'.format(subset, args.seed))
    with open(json_path, 'w') as f:
        json.dump(snippet_result, f, cls=NumpyArrayEncoder)
         
    if args.mode == 'train' or args.mode == 'infer':
        test_acc = num_correct / num_total
        print("TEST ACC:{:.4f}".format(test_acc))
        test_map = log_evaluate(args, step, test_acc, logger, json_path, test_info, subset)
        return test_map


@torch.no_grad()
def PseR_test(net, args, train_loader):
    net.eval()
    rk_mil_losss = []
    # >> load gt json
    # your OpenTAD_data_path
    with open('OpenTAD_data_path/thumos-14/annotations/thumos_14_anno.json', 'r') as f:
        gt_json = json.load(f)

    for sample in train_loader:
        data, vid_label, point_label = sample['data'], sample['vid_label'], sample['point_label']
        data = data.to(args.device)
        vid_label = vid_label.to(args.device)
        point_label = point_label.to(args.device)
        seed_seg = sample["seed_seg"]
        seed_label = sample["seed_label"]
        seed_seg = seed_seg.to(args.device)
        seed_label = seed_label.to(args.device)

        outputs = net.forward_pser(
            data,
            vid_label,
            seed_seg,
            seed_label,
            sample
        )

        # >> pred to pse label
        refined_pse_labels = outputs['refined_pse_labels']
        refined_act_labels = outputs['refined_act_labels'][0]
        annos = pred2pse(refined_pse_labels, refined_act_labels, train_loader)
        vid_name = sample['vid_name'][0]
        gt_json['database'][vid_name]['annotations'] = annos

        rk_mil_loss = outputs['rk_mil_loss'].item()
        rk_mil_losss.append(rk_mil_loss)
    
    # your OpenTAD_data_path
    file_path = 'OpenTAD_data_path/thumos-14/annotations/lacp_pser.json'
    if os.path.exists(file_path):
        os.remove(file_path)
    with open(file_path, "w") as json_file:
        json.dump(gt_json, json_file, indent=4)

    return rk_mil_losss


def pred2pse(proposals, action_labels, data_loader):
    name2id = data_loader.dataset.class_idx_dict
    id2name = {value: key for key, value in name2id.items()}
    proposals = proposals.cpu().detach().numpy()

    pse_labels = []
    for si in range(proposals.shape[0]):
        start = proposals[si][0]
        end = proposals[si][1]
        label_idx = int(action_labels[si])
        label_name = id2name[label_idx]
        pse_labels.append(dict(
            label = label_name,
            segment = [round(start, 3), round(end, 3)],
            label_id = label_idx,
        ))

    return pse_labels
