_base_ = ['./upernet_v_vit-base.py']

model = dict(
    pretrained='/opt/ml/mmsegmentation/pretrain/upernet_deit-b16_ln_mln_512x512_160k_ade20k_20210623_153535-8a959c14.pth',
    backbone=dict(drop_path_rate=0.1, final_norm=True))