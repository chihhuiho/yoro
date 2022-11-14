det=$1
epoch=$2
gpu=$3
# change debug to False to run full training
debug=False

# For RefCoco+ dataset, we load the modulated detection pretraining checkpoints as initial weight.
python run.py with data_root=./datasets/arrow/refcocop_mdetr num_gpus=$gpu num_nodes=1 per_gpu_batchsize=32 fast_dev_run=$debug task_finetune_refcocop_mdetr_coco_noaug reg_input="det_token_with_cls" load_path="result/pretrain/best.ckpt" det_token_num=$det max_epoch=$epoch mdetr_use_alignment=True patch_level_alignment=True resume=True

