# change debug to False to run full evaluation
debug=False
# no oa and no pa
#loadpath="result/copsref/no_oa_no_pa.ckpt"
# with oa and no pa
#loadpath="result/copsref/with_oa_no_pa.ckpt"
# with oa and pa
loadpath="result/copsref/best.ckpt"


# running testing without specifying the test set
# use default test set in vilt/datasets/copsref_dataset.py 

#python run.py with data_root=./datasets/arrow/COPSREF num_gpus=1 num_nodes=1 per_gpu_batchsize=32 fast_dev_run=$debug task_finetune_copsref_mdetr_coco_noaug_rec reg_input="det_token_with_cls" load_path="result/finetune_copsref_mdetr_coco_noaug_rec_seed0_from_best/version_3/checkpoints/best.ckpt" det_token_num=5 max_epoch=20 mdetr_use_alignment=True  test_only=True

python run.py with data_root=./datasets/arrow/COPSREF num_gpus=1 num_nodes=1 per_gpu_batchsize=32 fast_dev_run=$debug task_finetune_copsref_mdetr_coco_noaug_rec reg_input="det_token_with_cls" load_path=$loadpath det_token_num=5 max_epoch=20 mdetr_use_alignment=True specified_test_data=[\'copsref_val\']   test_only=True

python run.py with data_root=./datasets/arrow/COPSREF num_gpus=1 num_nodes=1 per_gpu_batchsize=32 fast_dev_run=$debug task_finetune_copsref_mdetr_coco_noaug_rec reg_input="det_token_with_cls" load_path=$loadpath det_token_num=5 max_epoch=20 mdetr_use_alignment=True specified_test_data=[\'copsref_test\']   test_only=True
