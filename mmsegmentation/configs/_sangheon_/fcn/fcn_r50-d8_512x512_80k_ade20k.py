_base_ = [
    '../_base_/models/fcn_r50-d8.py', '../_base_/datasets/trash_custom.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_custom.py'
]
model = dict(
    decode_head=dict(num_classes=11), auxiliary_head=dict(num_classes=11))

data = dict(
    samples_per_gpu=8
)