_base_ = [
    '../_base_/models/upernet_beit.py', '../_base_/datasets/trash_custom_640x640.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_custom.py'
]

checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/beit/upernet_beit-large_fp16_8x1_640x640_160k_ade20k/upernet_beit-large_fp16_8x1_640x640_160k_ade20k-8fc0dd5d.pth'

model = dict(
    backbone=dict(
        type='BEiT',
        embed_dims=1024,
        num_layers=24,
        num_heads=16,
        mlp_ratio=4,
        qv_bias=True,
        init_values=1e-6,
        drop_path_rate=0.2,
        out_indices=[7, 11, 15, 23]),
    neck=dict(embed_dim=1024, rescales=[4, 2, 1, 0.5]),
    decode_head=dict(
        in_channels=[1024, 1024, 1024, 1024], num_classes=11, channels=1024),
    auxiliary_head=dict(in_channels=1024, num_classes=11),
    test_cfg=dict(mode='slide', crop_size=(640, 640), stride=(426, 426)),
    init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file)
    )

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=2e-5,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=24, layer_decay_rate=0.95))

# scheduler 수정 ※ lr의 변동 없음
# lr_config = dict(policy='poly', power=1, min_lr=0.00006, by_epoch=True)
lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=3000,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

data = dict(samples_per_gpu=2) # batch_size
optimizer_config = dict(
    type='GradientCumulativeFp16OptimizerHook', cumulative_iters=2)

fp16 = dict()

# ⭐️⭐️⭐️ 꼭 지우고 돌리세요~~~~
runner = dict(type='EpochBasedRunner', max_epochs=1)