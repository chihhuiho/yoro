det=$1
epoch=$2
gpu=$3
# change debug to False to run full training
debug=True

# For Copsref dataset, we load the pretrained weight on the gqa imbalanced training set as initial weight
# no oa no pa 
python run.py with data_root=./datasets/arrow/COPSREF num_gpus=$gpu num_nodes=1 per_gpu_batchsize=32 fast_dev_run=$debug task_finetune_copsref_mdetr_coco_noaug reg_input="det_token_with_cls" load_path="result/pretrain/no_oa_no_pa.ckpt" det_token_num=$det max_epoch=$epoch mdetr_use_alignment=False  patch_level_alignment_KL=False

