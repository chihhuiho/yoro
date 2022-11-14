det=$1
epoch=$2
gpu=$3
debug=True

#For Modulated detection pretraining, we start from a mlm-itm pretrained model, such as the vilt pretraining checkpoint.
python run.py with data_root=./datasets/arrow/mdetr_pretrain/maxlen40_maxbox1 num_gpus=$gpu num_nodes=1 per_gpu_batchsize=32 fast_dev_run=$debug task_mdetr_pretrain reg_input="det_token_with_cls" load_path="pretrained_weight/vilt_200k_mlm_itm.ckpt" det_token_num=$det max_epoch=$epoch mdetr_use_alignment=True batch_size=512 patch_level_alignment_KL=True


