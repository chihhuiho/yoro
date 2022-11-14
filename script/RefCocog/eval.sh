# change debug to False to run full evaluation
debug=False
loadpath="result/refcocog/best.ckpt"

# running testing without specifying the test set
# use default test set in vilt/datasets/refcoco_mdetr_dataset.py 
#python run.py with data_root=./datasets/arrow/refcocog_mdetr num_gpus=1 num_nodes=1 per_gpu_batchsize=32 fast_dev_run=$debug task_finetune_refcocog_mdetr_coco_noaug reg_input="det_token_with_cls" load_path=$loadpath det_token_num=5 max_epoch=20 mdetr_use_alignment=False test_only=True

# running testing with specifying the test set
# use refcocog_test
python run.py with data_root=./datasets/arrow/refcocog_mdetr num_gpus=1 num_nodes=1 per_gpu_batchsize=32 fast_dev_run=$debug task_finetune_refcocog_mdetr_coco_noaug reg_input="det_token_with_cls" load_path=$loadpath det_token_num=5 max_epoch=20 mdetr_use_alignment=False test_only=True specified_test_data=[\'refcocog_test\']


# running testing without specifying the test set
# use refcocog_val
python run.py with data_root=./datasets/arrow/refcocog_mdetr num_gpus=1 num_nodes=1 per_gpu_batchsize=32 fast_dev_run=$debug task_finetune_refcocog_mdetr_coco_noaug reg_input="det_token_with_cls" load_path=$loadpath det_token_num=5 max_epoch=20 mdetr_use_alignment=False test_only=True specified_test_data=[\'refcocog_val\']
