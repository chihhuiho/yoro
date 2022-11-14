#change debug to False to run full evaluation
debug=False

loadpath="result/refcocop/best.ckpt"

# running testing without specifying the test set
# use default test set in vilt/datasets/refcocop_mdetr_dataset.py 

#python run.py with data_root=./datasets/arrow/refcocop_mdetr num_gpus=1 num_nodes=1 per_gpu_batchsize=32 fast_dev_run=$debug task_finetune_refcocop_mdetr_coco_noaug reg_input="det_token_with_cls" load_path=$loadpath det_token_num=5 max_epoch=20 mdetr_use_alignment=True test_only=True

# running testing with specifying the test set
# use refcocop_testA
python run.py with data_root=./datasets/arrow/refcocop_mdetr num_gpus=1 num_nodes=1 per_gpu_batchsize=32 fast_dev_run=$debug task_finetune_refcocop_mdetr_coco_noaug reg_input="det_token_with_cls" load_path=$loadpath det_token_num=5 max_epoch=20 mdetr_use_alignment=True specified_test_data=[\'refcoco+_testA\'] test_only=True


# running testing with specifying the test set
# use refcocop_testB
python run.py with data_root=./datasets/arrow/refcocop_mdetr num_gpus=1 num_nodes=1 per_gpu_batchsize=32 fast_dev_run=$debug task_finetune_refcocop_mdetr_coco_noaug reg_input="det_token_with_cls" load_path=$loadpath det_token_num=5 max_epoch=20 mdetr_use_alignment=True specified_test_data=[\'refcoco+_testB\'] test_only=True

# running testing with specifying the test set
# use refcocop_val
python run.py with data_root=./datasets/arrow/refcocop_mdetr num_gpus=1 num_nodes=1 per_gpu_batchsize=32 fast_dev_run=$debug task_finetune_refcocop_mdetr_coco_noaug reg_input="det_token_with_cls" load_path=$loadpath det_token_num=5 max_epoch=20 mdetr_use_alignment=True specified_test_data=[\'refcoco+_val\'] test_only=True
