import os
import json
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import random


class dataset(Dataset):
    def __init__(self, args, phase="train", sample="random", stage=1):
        self.args = args
        self.phase = phase
        self.sample = sample
        self.stage = stage
        self.num_segments = args.num_segments
        self.class_name_lst = args.class_name_lst
        self.class_idx_dict = {cls: idx for idx, cls in enumerate(self.class_name_lst)}
        self.num_class = args.num_class
        self.t_factor = args.frames_per_sec / args.segment_frames_num
        self.data_path = args.data_path
        self.feature_dir = os.path.join(self.data_path, 'features', self.phase)
        self._prepare_data()
    
    def _prepare_data(self):
        # >> video list
        self.data_list = [item.strip() for item in list(open(os.path.join(self.data_path, "split_{}.txt".format(self.phase))))]
        print("number of {} videos:{}".format(self.phase, len(self.data_list)))
        with open(os.path.join(self.data_path, "gt_full.json")) as f:
            self.gt_dict = json.load(f)["database"]

        # >> video label
        self.vid_labels = {}
        self.seg_labels = {}
        for item_name in self.data_list:
            item_anns_list = self.gt_dict[item_name]["annotations"]
            item_label = np.zeros(self.num_class)
            item_seg_label = []
            for ann in item_anns_list:
                ann_label = ann["label"]
                item_label[self.class_idx_dict[ann_label]] = 1.0
                item_seg_label.append(ann['segment'])
                item_seg_label[-1].append(self.class_idx_dict[ann_label])
            self.vid_labels[item_name] = item_label
            self.seg_labels[item_name] = item_seg_label

        # >> point label
        self.point_anno = pd.read_csv(os.path.join(self.data_path, 'point_labels', 'point_gaussian.csv'))

        if self.phase == "train":
            # >> seed proposal
            with open(os.path.join(self.args.data_path, 'seed_proposals/{}.json'.format(self.args.seed_name)),'r') as f:
                self.seeds_json = json.load(f)
            self.load_seeds()

        # >> ambilist
        if self.args.dataset == "THUMOS14":
            ambilist = './dataset/THUMOS14/Ambiguous_test.txt'
            ambilist = list(open(ambilist, "r"))
            self.ambilist = [a.strip("\n").split(" ") for a in ambilist]
      

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        vid_name = self.data_list[idx]
        vid_feature = np.load(os.path.join(self.feature_dir, vid_name + ".npy"))
        data, vid_len, sample_idx = self.process_feat(vid_feature)
        if self.phase == "train":
            vid_label, point_label, vid_duration, seg_label, vid_fps, seed_seg, seed_label = self.process_label(vid_name, vid_len, sample_idx)
            sample = dict(
                data = data, 
                vid_label = vid_label, 
                point_label = point_label,
                vid_name = vid_name, 
                vid_len = vid_len, 
                vid_duration = vid_duration,
                seg_label = seg_label,
                vid_fps = vid_fps,
                seed_seg = seed_seg,
                seed_label = seed_label,
            )
            return sample
        vid_label, point_label, vid_duration, seg_label, vid_fps = self.process_label(vid_name, vid_len, sample_idx)
        seg_label = np.array(seg_label)
        if self.stage == 1:
            sample = dict(
                data = data, 
                vid_label = vid_label, 
                point_label = point_label,
                vid_name = vid_name, 
                vid_len = vid_len, 
                vid_duration = vid_duration,
                seg_label = seg_label
            )
            return sample

    def process_feat(self, vid_feature):
        vid_len = vid_feature.shape[0]
        if vid_len <= self.num_segments or self.num_segments == -1:
            sample_idx = np.arange(vid_len).astype(int)
        elif self.num_segments > 0 and self.sample == "random":
            sample_idx = np.arange(self.num_segments) * vid_len / self.num_segments
            for i in range(self.num_segments):
                if i < self.num_segments - 1:
                    if int(sample_idx[i]) != int(sample_idx[i + 1]):
                        sample_idx[i] = np.random.choice(range(int(sample_idx[i]), int(sample_idx[i + 1]) + 1))
                    else:
                        sample_idx[i] = int(sample_idx[i])
                else:
                    if int(sample_idx[i]) < vid_len - 1:
                        sample_idx[i] = np.random.choice(range(int(sample_idx[i]), vid_len))
                    else:
                        sample_idx[i] = int(sample_idx[i])
        elif self.num_segments > 0 and self.sample == 'uniform':
            samples = np.arange(self.num_segments) * vid_len / self.num_segments
            samples = np.floor(samples)
            sample_idx =  samples.astype(int)
        else:
            raise AssertionError('Not supported sampling !')
        feature = vid_feature[sample_idx]
        
        return feature, vid_len, sample_idx

    def process_label(self, vid_name, vid_len, sample_idx):
        vid_label = self.vid_labels[vid_name]
        vid_duration, vid_fps = self.gt_dict[vid_name]['duration'], self.gt_dict[vid_name]['fps']
        seg_label = self.seg_labels[vid_name]
        seg_label = np.array(seg_label)
        if self.phase == "train":
            seed_seg = np.array(self.seed_segments[vid_name])
            seed_label = np.array(self.seed_labels[vid_name])

        if self.num_segments == -1:
            self.t_factor_point = self.args.frames_per_sec / (vid_fps * 16)
            temp_anno = np.zeros([vid_len, self.num_class], dtype=np.float32)
            temp_df = self.point_anno[self.point_anno["video_id"] == vid_name][['point', 'class']]
            for key in temp_df['point'].keys():
                point = temp_df['point'][key]
                class_idx = self.class_idx_dict[temp_df['class'][key]]
                temp_anno[int(point * self.t_factor_point)][class_idx] = 1
            point_label = temp_anno[sample_idx, :]
            if self.phase == "train":
                return vid_label, point_label, vid_duration, seg_label, vid_fps, seed_seg, seed_label
            return vid_label, point_label, vid_duration, seg_label, vid_fps
        
        else:
            self.t_factor_point = self.num_segments / (vid_fps * vid_duration)
            temp_anno = np.zeros([self.num_segments, self.num_class], dtype=np.float32)
            temp_df = self.point_anno[self.point_anno["video_id"] == vid_name][['point', 'class']]
            for key in temp_df['point'].keys():
                point = temp_df['point'][key]
                class_idx = self.class_idx_dict[temp_df['class'][key]]
                temp_anno[int(point * self.t_factor_point)][class_idx] = 1
            point_label = temp_anno
            if self.phase == "train":
                return vid_label, point_label, vid_duration, seg_label, vid_fps, seed_seg, seed_label
            return vid_label, point_label, vid_duration, seg_label, vid_fps

    def load_seeds(self):
        self.seed_segments = {}
        self.seed_labels = {}
        vid_names = list(self.seeds_json.keys())
        for vid_name in vid_names:
            item_seed_segment = []
            item_seed_label = []
            item_seeds = self.seeds_json[vid_name]
            for item_seed in item_seeds:
                item_seed_label.append(self.class_idx_dict[item_seed['label']])
                item_seed_segment.append(item_seed['segment'])
            self.seed_segments[vid_name] = item_seed_segment
            self.seed_labels[vid_name] = item_seed_label


    def collate_fn(self, batch):
        """
        Collate function for creating batches of data samples.
        """
        keys = batch[0].keys()
        data = {key: [] for key in keys}
        for sample in batch:
            for key in keys:
                data[key].append(sample[key])
        return data