### 통과 ✅

# upernet-beit-large-ade ✅
# python train.py /opt/ml/ViT-Adapter/segmentation/configs/_sangheon_/ade20k/upernet_beit_adapter_large_640_160k_ade20k_ss.py \
# --work-dir ./work_dirs/tmp \
# --project_name SH_find_model \
# --exp_name tmp


### 학습이 안 된 것들 ⛔️

# # upernet-beit-large-pascal ⛔️ 학습이 안됩니다. data.split 이라는 것 때문인 듯?
# python train.py /opt/ml/ViT-Adapter/segmentation/configs/_sangheon_/pascal_context/upernet_beit_adapter_large_480_80k_pascal_context_59_ss.py \
# --work-dir ./work_dirs/upernet_beit_adapter_large \
# --project_name SH_find_model \
# --exp_name upernet_beit_adapter_large


### 1 epoch 실험 대상자 🔥

# # upernet-beit-large-pascal ⛔️ 학습이 안됩니다. data.split 이라는 것 때문인 듯?
# python train.py /opt/ml/ViT-Adapter/segmentation/configs/_sangheon_/pascal_context/upernet_beit_adapter_large_480_80k_pascal_context_59_ss.py \
# --work-dir ./work_dirs/upernet_beit_adapter_large \
# --project_name SH_find_model \
# --exp_name upernet_beit_adapter_large

# mask2former_beitv2_adapter_large_896
# python train.py /opt/ml/ViT-Adapter/segmentation/configs/_sangheon_/ade20k/mask2former_beitv2_adapter_large_896_80k_ade20k_ss.py \
# --work-dir ./work_dirs/mask2former_beitv2_adapter_large_896 \
# --project_name SH_find_model \
# --exp_name mask2former_beitv2_adapter_large_896

# # mask2former_beit_adapter_large_640 -> 저장공간때매 끊겨서 resume 했음
# python train.py /opt/ml/ViT-Adapter/segmentation/configs/_sangheon_/ade20k/mask2former_beit_adapter_large_640_160k_ade20k_ss.py \
# --work-dir ./work_dirs/mask2former_beit_adapter_large_640 \
# --resume-from /opt/ml/ViT-Adapter/segmentation/work_dirs/mask2former_beit_adapter_large_640/best_mIoU_epoch_4.pth \
# --project_name SH_find_model \
# --exp_name mask2former_beit_adapter_large_640

# 위 실험 27epoch에서 train_all.json에 finetuning
# python train.py /opt/ml/ViT-Adapter/segmentation/work_dirs/mask2former_beit_adapter_large_640_finetuning/mask2former_beit_adapter_large_640_160k_ade20k_ss_finetuning.py \
# --work-dir ./work_dirs/mask2former_beit_adapter_large_640_finetuning \
# --project_name SH_find_model \
# --exp_name mask2former_beit_adapter_large_640_finetuning_resume \
# --no-validate

# 위 실험 finetuning 10epoch 더 training
python train.py /opt/ml/ViT-Adapter/segmentation/configs/_sangheon_/ade20k/mask2former_beit_adapter_large_640_160k_ade20k_ss_finetuning_resume.py \
--work-dir ./work_dirs/mask2former_beit_adapter_large_640_finetuning_resume2 \
--project_name SH_find_model \
--exp_name mask2former_beit_adapter_large_640_finetuning_resume \
--no-validate