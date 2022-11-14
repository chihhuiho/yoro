# running testing without specifying the test set
# use default test set in vilt/datasets/mdetr_pretrain_dataset.py 

debug=False
ckptpath="result/pretrain/best.ckpt"

python run.py with data_root=./datasets/arrow/mdetr_pretrain/maxlen40_maxbox1 num_gpus=1 num_nodes=1 per_gpu_batchsize=32 fast_dev_run=$debug task_mdetr_pretrain reg_input="det_token_with_cls" load_path=$ckptpath det_token_num=5 max_epoch=20 mdetr_use_alignment=True batch_size=512 patch_level_alignment_KL=True patch_level_alignment_weight1=0.5 patch_level_alignment_weight2=0.5  test_only=True

