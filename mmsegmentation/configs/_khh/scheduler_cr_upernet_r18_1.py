_base_ = ['./_base_/models/upernet_r50.py']

# 모델 수정
model = dict(
    pretrained='open-mmlab://resnet18_v1c',
    backbone=dict(depth=18),
    decode_head=dict(in_channels=[64, 128, 256, 512], num_classes=11),
    auxiliary_head=dict(in_channels=256, num_classes=11))

# 싹다 수정
dataset_type = 'CustomDataset'
data_root = '/opt/ml/input/data/mmseg'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

classes = [
    'Backgroud', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
    'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing'
]

palette = [[0, 0, 0], [192, 0, 128], [0, 128, 192], [0, 128, 64], [128, 0, 0],
           [64, 0, 128], [64, 0, 192], [192, 128, 64], [192, 192, 128],
           [64, 64, 128], [128, 0, 192]]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(512, 512)),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type='CustomDataset',
        data_root='/opt/ml/input/data/mmseg_tiny',
        img_dir='images/training',
        ann_dir='annotations/training',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='Resize', img_scale=(512, 512)),
            dict(type='RandomFlip', prob=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ],
        classes=[
            'Backgroud', 'General trash', 'Paper', 'Paper pack', 'Metal',
            'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery',
            'Clothing'
        ],
        palette=[[0, 0, 0], [192, 0, 128], [0, 128, 192], [0, 128, 64],
                 [128, 0, 0], [64, 0, 128], [64, 0, 192], [192, 128, 64],
                 [192, 192, 128], [64, 64, 128], [128, 0, 192]]),
    val=dict(
        type='CustomDataset',
        data_root='/opt/ml/input/data/mmseg_tiny',
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=[
            'Backgroud', 'General trash', 'Paper', 'Paper pack', 'Metal',
            'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery',
            'Clothing'
        ],
        palette=[[0, 0, 0], [192, 0, 128], [0, 128, 192], [0, 128, 64],
                 [128, 0, 0], [64, 0, 128], [64, 0, 192], [192, 128, 64],
                 [192, 192, 128], [64, 64, 128], [128, 0, 192]]),
    test=dict(
        type='CustomDataset',
        data_root='/opt/ml/input/data/mmseg_tiny',
        img_dir='test',
        ann_dir='annotations/validation',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=[
            'Backgroud', 'General trash', 'Paper', 'Paper pack', 'Metal',
            'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery',
            'Clothing'
        ],
        palette=[[0, 0, 0], [192, 0, 128], [0, 128, 192], [0, 128, 64],
                 [128, 0, 0], [64, 0, 128], [64, 0, 192], [192, 128, 64],
                 [192, 192, 128], [64, 64, 128], [128, 0, 192]]))

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='dataset_search',
                entity='boostcamp_aitech4_jdp',
                name='fold3'),
            interval=10)
    ])

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
# change work flow for hook validation loss
workflow = [('train', 1), ('val', 1)] 
cudnn_benchmark = True

## 신경쓸것들
optimizer_config = dict()
# 옵티마이저 수정
optimizer = dict(
    # _delete_=True, # 기존게 없으므로 삭제
    type='AdamW',
    lr=5e-5*1, # 6e-5
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))


lr_config = dict(
    policy='CosineRestart',
    warmup='constant',
    by_epoch=True,
    warmup_iters=1,
    warmup_ratio=1,
    periods=[10, 10, 10],
    restart_weights=[1, 0.75, 0.5],
    min_lr=0)

runner = dict(type='EpochBasedRunner', max_epochs=30)
checkpoint_config = dict(interval=25, save_last=True)
evaluation = dict(metric='mIoU', save_best='mIoU')
work_dir = './work_dirs/fcn_r50' # train.py에서 update됨
gpu_ids = [0]
auto_resume = False