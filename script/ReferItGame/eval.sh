# change debug to False to run full evaluation
debug=False
# no oa no pa
#loadpath="result/referitgame/no_oa_no_pa.ckpt"
# with oa no pa
#loadpath="result/referitgame/with_oa_no_pa.ckpt"
# with oa with pa
loadpath="result/referitgame/best.ckpt"

# running testing without specifying the test set
# use default test set in vilt/datasets/referitgame_dataset.py 
#python run.py with data_root=./datasets/arrow/referitgame num_gpus=1 num_nodes=1 per_gpu_batchsize=32 fast_dev_run=$debug task_finetune_referitgame_mdetr_coco_noaug reg_input="det_token_with_cls" load_path=$loadpath det_token_num=5 max_epoch=40 mdetr_use_alignment=True test_only=True

python run.py with data_root=./datasets/arrow/referitgame num_gpus=1 num_nodes=1 per_gpu_batchsize=32 fast_dev_run=$debug task_finetune_referitgame_mdetr_coco_noaug reg_input="det_token_with_cls" load_path=$loadpath det_token_num=5 max_epoch=40 mdetr_use_alignment=True test_only=True specified_test_data=[\'referitgame_val\']

python run.py with data_root=./datasets/arrow/referitgame num_gpus=1 num_nodes=1 per_gpu_batchsize=32 fast_dev_run=$debug task_finetune_referitgame_mdetr_coco_noaug reg_input="det_token_with_cls" load_path=$loadpath det_token_num=5 max_epoch=40 mdetr_use_alignment=True test_only=True specified_test_data=[\'referitgame_test\']
