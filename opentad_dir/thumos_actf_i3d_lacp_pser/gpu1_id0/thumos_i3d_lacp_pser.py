_base_ = [
    "../../_base_/datasets/thumos-14/features_i3d_pad.py",  # dataset config
    "../../_base_/models/actionformer.py",  # model config
]

annotation_path = "data/thumos-14/annotations/lacp_pser.json"

dataset = dict(
    train=dict(
        ann_file=annotation_path,
    ),
    val=dict(
        ann_file=annotation_path,
    ),
    test=dict(
        ann_file=annotation_path,
    ),
)

evaluation = dict(
    type="mAP",
    subset="validation",
    tiou_thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    ground_truth_filename=annotation_path,
)

model = dict(projection=dict(in_channels=2048, input_pdrop=0.2))

solver = dict(
    train=dict(batch_size=2, num_workers=2),
    val=dict(batch_size=1, num_workers=1),
    test=dict(batch_size=1, num_workers=1),
    clip_grad_norm=1,
    ema=True,
)

optimizer = dict(type="AdamW", lr=1e-4, weight_decay=0.05, paramwise=True)
scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=5, max_epoch=35)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
post_processing = dict(
    nms=dict(
        use_soft_nms=True,
        sigma=0.5,
        max_seg_num=2000,
        iou_threshold=0.1,  # does not matter when use soft nms
        min_score=0.001,
        multiclass=True,
        voting_thresh=0.7,  #  set 0 to disable
    ),
    save_dict=True,
)

workflow = dict(
    logging_interval=20,
    checkpoint_interval=1,
    val_loss_interval=1,
    val_eval_interval=1,
    val_start_epoch=30,
)

work_dir = "exps/pser/thumos_actf_i3d_lacp_pser"
