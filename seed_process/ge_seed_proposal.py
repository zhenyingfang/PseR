import os
import json
import argparse
import numpy as np
import pandas as pd


def ge_seed_proposal_single(props, points):
    t_pos = [-1] + [point[0] for point in points] + [1e5]
    RP = {}
    for i, point in enumerate(points):
        t, label = point
        p_match, max_score = None, 0
        # with boundary condition
        if label in props.keys():
            for prop in props[label]:
                inclusive_condition = (prop[0] <= t and prop[1] >= t)
                boundary_condition = (prop[0] > t_pos[i] and prop[1] < t_pos[i+2])
                score_condition = prop[2] > max_score
                if inclusive_condition and boundary_condition and score_condition:
                    p_match, max_score = prop, prop[2]
        # missing
        if p_match is None:
            p_match = [t - 1.0, t + 1.0, 0.2]

        if label not in RP.keys():
            RP[label] = []
        if p_match is not None:
            RP[label].append(p_match)
                    
    return RP

def ge_seed_proposal(json_path, point_path, gt_path, dst_path, segment_frames_num=16, frames_per_sec=25):
    with open(json_path,'r') as f:
        seeds_dict = json.load(f)["results"]
    with open(gt_path, "r") as f:
        gt_dict = json.load(f)["database"]
    point_anno = pd.read_csv(point_path)

    seed_proposal_dict = dict()
    vid_names = list(seeds_dict.keys())
    for vid_name in vid_names:
        vid_fps = gt_dict[vid_name]['fps']
        item_seeds = seeds_dict[vid_name]
        points_df = point_anno[point_anno["video_id"] == vid_name][['point', 'class']].sort_values(by='point', ascending=True)
        points = []
        for key in points_df['point'].keys():
            t_point = points_df['point'][key] / vid_fps                # frame -> time
            points.append([t_point, points_df['class'][key]])
        proposals = dict()
        for item_seed in item_seeds:
            item_key = item_seed['label']
            item_seg = item_seed['segment']
            item_score = item_seed['score']
            if item_key not in proposals.keys():
                proposals[item_key] = [[item_seg[0], item_seg[1], item_score]]
            else:
                proposals[item_key].append([item_seg[0], item_seg[1], item_score])

        seed_item = ge_seed_proposal_single(proposals, points)
        seed_item_keys = list(seed_item.keys())
        seed_res_item = []
        for seed_item_key in seed_item_keys:
            tmp_item = seed_item[seed_item_key]
            for tmp_seg in tmp_item:
                seed_res_item.append(
                    {
                        "label": seed_item_key,
                        "segment": [tmp_seg[0], tmp_seg[1]]
                    }
                )
        seed_proposal_dict[vid_name] = seed_res_item
    
    with open(dst_path, 'w') as f:
        json.dump(seed_proposal_dict, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process video validation')
    parser.add_argument('--file_name', default="hrpro_seed.json", help='Name of the source data file')
    args = parser.parse_args()

    file_name = args.file_name
    base_dir = "dataset/THUMOS14/seed_proposals"
    json_path = os.path.join(base_dir, file_name)

    point_path = "dataset/THUMOS14/point_labels/point_gaussian.csv"
    gt_path = "dataset/THUMOS14/gt_full.json"
    
    dst_path = os.path.join(base_dir, file_name.replace(".json", "_final.json"))

    ge_seed_proposal(json_path, point_path, gt_path, dst_path)

    print('done...')
