lr = 0.0001  # max learning rate

# optimizer
optimizer = dict( # ⭐️ 옵티마이저 고정
    # _delete_=True, # 기존게 없으므로 삭제
    type='AdamW',
    lr=0.00006, # 6e-5
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)

# optimizer_config
optimizer_config = dict()
# optimizer_config = dict(grad_clip=None)



# runtime settings
# lr_config = dict(policy="poly", power=0.9, min_lr=1e-4)
# lr_config = dict(
#     policy='CosineAnnealing',
#     warmup='linear',
#     warmup_iters=327,
#     warmup_ratio=0.1,
#     min_lr_ratio=1e-6)

# learning policy
#lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# scheduler 수정 ※ lr의 변동 없음
lr_config = dict(policy='poly', power=1, min_lr=0.00006, by_epoch=True)

# runner
#runner = dict(type='IterBasedRunner', max_iters=80000)
runner = dict(type='EpochBasedRunner', max_epochs=30)

# checkpoint_config
# checkpoint_config = dict(interval=5, save_last=True)
checkpoint_config = dict(max_keep_ckpts=3, interval=3, save_last=True)

evaluation = dict(metric='mIoU', save_best='mIoU')