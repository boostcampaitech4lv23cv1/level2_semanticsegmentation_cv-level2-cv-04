lr = 0.0001  # max learning rate

# optimizer
# optimizer = dict(type='Adam', lr=lr, weight_decay=0.005)
# optimizer = dict(
#     _delete_=True,
#     type="AdamW",
#     lr=0.00006,
#     betas=(0.9, 0.999),
#     weight_decay=0.01,
#     paramwise_cfg=dict(
#         custom_keys={
#             "absolute_pos_embed": dict(decay_mult=0.0),
#             "relative_position_bias_table": dict(decay_mult=0.0),
#             "norm": dict(decay_mult=0.0),
#         }
#     ),
# )

# optimizer_config = dict(grad_clip=None)

# runtime settings
# lr_config = dict(policy="poly", power=0.9, min_lr=1e-4)
# lr_config = dict(
#     policy='CosineAnnealing',
#     warmup='linear',
#     warmup_iters=327,
#     warmup_ratio=0.1,
#     min_lr_ratio=1e-6)

runner = dict(type='EpochBasedRunner', max_epochs=50)
checkpoint_config = dict(interval=5, save_last=True)
#checkpoint_config = dict(max_keep_ckpts=2, interval=1, save_last=True)
evaluation = dict(metric='mIoU', save_best='mIoU')