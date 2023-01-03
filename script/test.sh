
# upernet-beit-adapter-large-ade20k ✅
# python test.py /opt/ml/ViT-Adapter/segmentation/work_dirs/upernet_beit_adapter_large_ade20k/upernet_beit_adapter_large_640_160k_ade20k_ss.py \
# /opt/ml/ViT-Adapter/segmentation/work_dirs/upernet_beit_adapter_large_ade20k/best_mIoU_epoch_30.pth \
# --out out.pkl

# mask2former_beit_adapter_large_640
python test.py /opt/ml/ViT-Adapter/segmentation/work_dirs/mask2former_beit_adapter_large_640/mask2former_beit_adapter_large_640_160k_ade20k_ss.py \
/opt/ml/ViT-Adapter/segmentation/work_dirs/mask2former_beit_adapter_large_640/epoch_24.pth \
--out out.pkl
# --show-dir /opt/ml/ViT-Adapter/segmentation/work_dirs/mask2former_beit_adapter_large_640

