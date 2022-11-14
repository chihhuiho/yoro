from sacred import Experiment

ex = Experiment("ViLT")

def _loss_names(d):
    ret = {
        "itm": 0,
        "mlm": 0,
        "mpp": 0,
        "vqa": 0,
        "nlvr2": 0,
        "irtr": 0,
        "rec": 0,
        "mdetr_pretrain": 0,
        "mdetr_pretrain_ablade": 0,
        "flickr_mdetr": 0,
        "gqa_mdetr": 0,
        "clever_mdetr": 0,
        "snlive": 0,
        "copsref": 0,
    }
    ret.update(d)
    return ret


@ex.config
def config():
    exp_name = "vilt"
    seed = 0
    datasets = ["coco", "vg", "sbu", "gcc"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 4096  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.

    # Image setting
    train_transform_keys = ["pixelbert"]
    val_transform_keys = ["pixelbert"]
    image_size = 384
    max_image_len = -1
    patch_size = 32
    draw_false_image = 1
    image_only = False

    # Text Setting
    vqav2_label_size = 3129
    max_text_len = 40
    tokenizer = "bert-base-uncased"
    vocab_size = 30522
    whole_word_masking = False
    mlm_prob = 0.15
    draw_false_text = 0

    # Transformer Setting
    vit = "vit_base_patch32_384"
    hidden_size = 768
    num_heads = 12
    num_layers = 12
    mlp_ratio = 4
    drop_rate = 0.1

    # Optimizer Setting
    optim_type = "adamw"
    learning_rate = 1e-4
    weight_decay = 0.01
    decay_power = 1
    max_epoch = 100
    max_steps = 25000
    warmup_steps = 2500
    end_lr = 0
    lr_mult = 1  # multiply lr for downstream heads

    # Downstream Setting
    get_recall_metric = False

    # PL Trainer Setting
    resume_from = None
    fast_dev_run = False
    val_check_interval = 1.0
    test_only = False

    # below params varies with the environment
    data_root = ""
    log_dir = "result"
    per_gpu_batchsize = 0  # you should define this manually with per_gpu_batch_size=#
    num_gpus = 1
    num_nodes = 1
    load_path = ""
    num_workers = 8
    precision = 16

    load_dict_path = ""

    # for REC
    reg_input = "det_token_with_cls" # [det_token, det_token_with_cls, det_token_concat_cls]
    matchingloss_bbox_weight = 5
    matchingloss_giou_weight = 2
    matchingloss_cls_weight = 1

    
    mdetr_use_alignment = False # for using contrastive_alignment in MDETR
    contrastive_align_loss_weight = 1
    patch_level_alignment = False # for using patch alignment
    patch_level_alignment_weight1 = 1 
    patch_level_alignment_weight2 = 1 
    patch_level_alignment_KL = False

    # flickr mdetr downstream task
    flickr_mdetr_GT_type = "separate"

    # gqa mdetr
    gqa_mdetr_split_type = "balanced"
    gqa_mdetr_test_data = "testdev"
    qa_loss_coef = 1
    qa_loss_only = False

    # for det token
    add_det_token = False
    det_token_num = 100
    matcher_multioutput = False
    matcher_multioutput_mix = False
    truncated_det_token_num = None
    use_det_fuser = False
    det_token_reinit = False
    load_yolos_pretrain_det = False
    yolos_pretrain_det_path = "pretrained_weight/yolos_det_weight/"

    # Load specific training, validation and testing data
    specified_train_data=None
    specified_val_data=None
    specified_test_data=None

    # resume the model
    resume = False
    resume_qa = False

    # for visualization
    vis_sample = 50 # save 50 correct samples and 50 incorrect samples
    save_sample = False
    compute_loss = True

    
# Named configs for "environment" which define gpus and nodes, and paths
@ex.named_config
def env_dandelin():
    data_root = "/data2/dsets/dataset"
    log_dir = "/data2/vilt/result"
    num_gpus = 8
    num_nodes = 1


# Named configs for "task" which define datasets, loss_names and desired batch_size, warmup_steps, epochs, and exp_name

# Vilt pretraining task
@ex.named_config
def task_mlm_itm():
    exp_name = "mlm_itm"
    #datasets = ["coco", "vg", "sbu", "gcc"]
    datasets = ["coco"]#, "vg", "sbu", "gcc"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 4096
    max_epoch = 10
    max_image_len = 200


# Vilt pretraining task with image augmentation
@ex.named_config
def task_mlm_itm_randaug():
    exp_name = "mlm_itm_randaug"
    datasets = ["coco", "vg", "sbu", "gcc"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 4096
    max_epoch = 10
    max_image_len = 200

# Vilt pretraining task with masked patch prediction
@ex.named_config
def task_mlm_itm_mpp():
    exp_name = "mlm_itm_mpp"
    datasets = ["coco", "vg", "sbu", "gcc"]
    loss_names = _loss_names({"itm": 1, "mlm": 1, "mpp": 1})
    batch_size = 4096
    max_epoch = 10
    max_image_len = 200


# Vilt pretraining task on mdetr dataset
@ex.named_config
def task_mdetr_mlm_itm():
    exp_name = "mdetr_mlm_itm"
    datasets = ["mdetr_pretrain"]
    train_transform_keys = ["coco"]
    val_transform_keys = ["coco"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 512

# mdetr pretraining with modulated detection dataset with no augmentation
@ex.named_config
def task_mdetr_pretrain():
    exp_name = "mdetr_pretrain"
    datasets = ["mdetr_pretrain"]
    train_transform_keys = ["coco"]
    val_transform_keys = ["coco"]
    loss_names = _loss_names({"mdetr_pretrain": 1})
    batch_size = 512
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4
    val_check_interval = 0.1
    lr_mult = 10

    # for det token
    add_det_token = True
    det_token_num = 1
    
# mdetr pretraining with modulated detection dataset with no augmentation
@ex.named_config
def task_mdetr_pretrain_ablade():
    exp_name = "mdetr_pretrain_ablade"
    datasets = ["mdetr_pretrain"]
    train_transform_keys = ["coco"]
    val_transform_keys = ["coco"]
    loss_names = _loss_names({"mdetr_pretrain_ablade": 1})
    batch_size = 512
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4
    val_check_interval = 0.1
    lr_mult = 10

    # for det token
    add_det_token = False

# mdetr pretraining with modulated detection dataset with no augmentation
@ex.named_config
def task_mdetr_pretrain_flickr_ref():
    exp_name = "mdetr_pretrain_flickr_ref"
    datasets = ["mdetr_pretrain"]
    train_transform_keys = ["coco"]
    val_transform_keys = ["coco"]
    loss_names = _loss_names({"mdetr_pretrain": 1})
    batch_size = 512
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4
    val_check_interval = 0.1
    lr_mult = 10

    # for det token
    add_det_token = True
    det_token_num = 1



# Finetune for downstream task on nlvr2
@ex.named_config
def task_finetune_nlvr2():
    exp_name = "finetune_nlvr2"
    datasets = ["nlvr2"]
    loss_names = _loss_names({"nlvr2": 1})
    batch_size = 128
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4


# Finetune for downstream task on nlvr2 with random augmentation
@ex.named_config
def task_finetune_nlvr2_randaug():
    exp_name = "finetune_nlvr2_randaug"
    datasets = ["nlvr2"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"nlvr2": 1})
    batch_size = 128
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4


# Finetune for downstream task on vqa
@ex.named_config
def task_finetune_vqa():
    exp_name = "finetune_vqa"
    datasets = ["vqa"]
    loss_names = _loss_names({"vqa": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    raw_false_image = 0
    learning_rate = 1e-4
    val_check_interval = 0.1
    lr_mult = 10


# Finetune for downstream task on vqa with random augmentation
@ex.named_config
def task_finetune_vqa_randaug():
    exp_name = "finetune_vqa_randaug"
    datasets = ["vqa"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"vqa": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4
    val_check_interval = 0.1
    lr_mult = 10


# Finetune for downstream task on image retrieval on coco
@ex.named_config
def task_finetune_irtr_coco():
    exp_name = "finetune_irtr_coco"
    datasets = ["coco"]
    loss_names = _loss_names({"itm": 0.5, "irtr": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    draw_false_text = 15
    learning_rate = 1e-4


# Finetune for downstream task on image retrieval on coco with random augmentation
@ex.named_config
def task_finetune_irtr_coco_randaug():
    exp_name = "finetune_irtr_coco_randaug"
    datasets = ["coco"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"itm": 0.5, "irtr": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    draw_false_text = 15
    learning_rate = 1e-4


# Finetune for downstream task on image/text retrieval on flickr30k
@ex.named_config
def task_finetune_irtr_f30k():
    exp_name = "finetune_irtr_f30k"
    datasets = ["f30k"]
    loss_names = _loss_names({"itm": 0.5, "irtr": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    draw_false_text = 15
    learning_rate = 1e-4


# Finetune for downstream task on image/text retrieval on flickr30k with random augmentation
@ex.named_config
def task_finetune_irtr_f30k_randaug():
    exp_name = "finetune_irtr_f30k_randaug"
    datasets = ["f30k"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"itm": 0.5, "irtr": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    draw_false_text = 15
    learning_rate = 1e-4

# Finetune for downstream task on image/text retrieval on flickr30k
@ex.named_config
def task_finetune_irtr_f30k_coco_noaug():
    exp_name = "finetune_irtr_f30k_coco_noaug"
    datasets = ["f30k"]
    train_transform_keys = ["coco"]
    val_transform_keys = ["coco"]
    loss_names = _loss_names({"itm": 0.5, "irtr": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    draw_false_text = 15
    learning_rate = 1e-4


# Finetune for downstream REC task on refcoco+ with mdetr annotation with no augmentation
@ex.named_config
def task_finetune_refcocop_mdetr_coco_noaug():
    exp_name = "finetune_refcocop_mdetr_coco_noaug"
    datasets = ["refcocop_mdetr"]
    train_transform_keys = ["coco"]
    val_transform_keys = ["coco"]
    loss_names = _loss_names({"rec": 1})
    batch_size = 512
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4
    val_check_interval = 0.1
    lr_mult = 10

    # for det token
    add_det_token = True
    det_token_num = 1

# Finetune for downstream REC task on refcocop with mdetr annotation with no augmentation
@ex.named_config
def task_finetune_refcocop_mdetr_coco_noaug_ablade():
    exp_name = "finetune_refcocop_mdetr_coco_noaug_ablade"
    datasets = ["refcocop_mdetr"]
    train_transform_keys = ["coco"]
    val_transform_keys = ["coco"]
    loss_names = _loss_names({"mdetr_pretrain_ablade": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4
    val_check_interval = 0.1
    lr_mult = 10

    # for det token
    add_det_token = True
    det_token_num = 1



# Finetune for downstream REC task on refcoco with mdetr annotation with no augmentation
@ex.named_config
def task_finetune_refcoco_mdetr_coco_noaug():
    exp_name = "finetune_refcoco_mdetr_coco_noaug"
    datasets = ["refcoco_mdetr"]
    train_transform_keys = ["coco"]
    val_transform_keys = ["coco"]
    loss_names = _loss_names({"rec": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4
    val_check_interval = 0.1
    lr_mult = 10

    # for det token
    add_det_token = True
    det_token_num = 1

# Finetune for downstream REC task on refcoco with mdetr annotation with no augmentation
@ex.named_config
def task_finetune_refcoco_mdetr_coco_noaug_ablade():
    exp_name = "finetune_refcoco_mdetr_coco_noaug_ablade"
    datasets = ["refcoco_mdetr"]
    train_transform_keys = ["coco"]
    val_transform_keys = ["coco"]
    loss_names = _loss_names({"mdetr_pretrain_ablade": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4
    val_check_interval = 0.1
    lr_mult = 10

    # for det token
    add_det_token = True
    det_token_num = 1



# Finetune for downstream REC task on refcocog with mdetr annotation with no augmentation
@ex.named_config
def task_finetune_refcocog_mdetr_coco_noaug():
    exp_name = "finetune_refcocog_mdetr_coco_noaug"
    datasets = ["refcocog_mdetr"]
    train_transform_keys = ["coco"]
    val_transform_keys = ["coco"]
    loss_names = _loss_names({"rec": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4
    val_check_interval = 0.1
    lr_mult = 10

    # for det token
    add_det_token = True
    det_token_num = 1

# Finetune for downstream REC task on refcocog with mdetr annotation with no augmentation
@ex.named_config
def task_finetune_refcocog_mdetr_coco_noaug_ablade():
    exp_name = "finetune_refcocog_mdetr_coco_noaug_ablade"
    datasets = ["refcocog_mdetr"]
    train_transform_keys = ["coco"]
    val_transform_keys = ["coco"]
    loss_names = _loss_names({"mdetr_pretrain_ablade": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4
    val_check_interval = 0.1
    lr_mult = 10

    # for det token
    add_det_token = True
    det_token_num = 1


# Finetune for phrase grounding downstream task on Flickr30k entities
@ex.named_config
def task_finetune_flickr_mdetr_coco_noaug():
    exp_name = "finetune_flickr_mdetr_coco_noaug"
    datasets = ["flickr_mdetr"]
    train_transform_keys = ["coco"]
    val_transform_keys = ["coco"]
    loss_names = _loss_names({"flickr_mdetr": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4
    val_check_interval = 0.1
    lr_mult = 10

    # for det token
    add_det_token = True
    det_token_num = 1

# Finetune for phrase grounding downstream task on Flickr30k entities transvg
@ex.named_config
def task_finetune_flickr_entity_transvg_coco_noaug():
    exp_name = "finetune_flickr_entity_transvg_coco_noaug"
    datasets = ["flickr_entity"]
    train_transform_keys = ["coco"]
    val_transform_keys = ["coco"]
    loss_names = _loss_names({"rec": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4
    val_check_interval = 0.1
    lr_mult = 10

    # for det token
    add_det_token = True
    det_token_num = 1
    specified_train_data=['train']
    specified_val_data=['test']
    specified_test_data=['test']



# Finetune for phrase grounding downstream task on Flickr30k entities merged
@ex.named_config
def task_finetune_flickr_entity_merged_coco_noaug():
    exp_name = "finetune_flickr_entity_merged_coco_noaug"
    datasets = ["flickr_entity"]
    train_transform_keys = ["coco"]
    val_transform_keys = ["coco"]
    loss_names = _loss_names({"rec": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4
    val_check_interval = 0.1
    lr_mult = 10

    # for det token
    add_det_token = True
    det_token_num = 1
    specified_train_data=['train_merged']
    specified_val_data=['val_merged']
    specified_test_data=['test_merged']



# Finetune for phrase grounding downstream task on Flickr30k entities seperate
@ex.named_config
def task_finetune_flickr_entity_seperate_coco_noaug():
    exp_name = "finetune_flickr_entity_seperate_coco_noaug"
    datasets = ["flickr_entity"]
    train_transform_keys = ["coco"]
    val_transform_keys = ["coco"]
    loss_names = _loss_names({"rec": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4
    val_check_interval = 0.1
    lr_mult = 10

    # for det token
    add_det_token = True
    det_token_num = 1
    specified_train_data=['train_seperate']
    specified_val_data=['val_seperate']
    specified_test_data=['test_seperate']
 


# Finetune for question answering (QA) downstream task on GQA
@ex.named_config
def task_finetune_gqa_mdetr_coco_noaug():
    exp_name = "finetune_gqa_mdetr_coco_noaug"
    datasets = ["gqa_mdetr"]
    train_transform_keys = ["coco"]
    val_transform_keys = ["coco"]
    loss_names = _loss_names({"gqa_mdetr": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4
    val_check_interval = 0.1
    lr_mult = 10

    # for det token
    add_det_token = True
    det_token_num = 1


# Finetune for question answering (QA) downstream task on CLEVER
@ex.named_config
def task_finetune_clever_mdetr_coco_noaug():
    exp_name = "finetune_clever_mdetr_coco_noaug"
    datasets = ["clever_mdetr"]
    train_transform_keys = ["coco"]
    val_transform_keys = ["coco"]
    loss_names = _loss_names({"clever_mdetr": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4
    val_check_interval = 0.1
    lr_mult = 10

    # for det token
    add_det_token = True
    det_token_num = 1


# Finetune for visual entailment downstream task on snlive
@ex.named_config
def task_finetune_snlive_coco_noaug():
    exp_name = "finetune_snlive_coco_noaug"
    datasets = ["snlive"]
    train_transform_keys = ["coco"]
    val_transform_keys = ["coco"]
    loss_names = _loss_names({"snlive": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4
    val_check_interval = 0.1
    lr_mult = 10

    # for det token
    add_det_token = True
    det_token_num = 1


# Finetune for REC downstream task on copsref
@ex.named_config
def task_finetune_copsref_mdetr_coco_noaug():
    exp_name = "finetune_copsref_mdetr_coco_noaug"
    datasets = ["copsref"]
    train_transform_keys = ["coco"]
    val_transform_keys = ["coco"]
    loss_names = _loss_names({"copsref": 1})
    batch_size = 512
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4
    val_check_interval = 0.1
    lr_mult = 10

    # for det token
    add_det_token = True
    det_token_num = 1

# Finetune for REC downstream task on copsref
@ex.named_config
def task_finetune_copsref_mdetr_coco_noaug_rec():
    exp_name = "finetune_copsref_mdetr_coco_noaug_rec"
    datasets = ["copsref"]
    train_transform_keys = ["coco"]
    val_transform_keys = ["coco"]
    loss_names = _loss_names({"rec": 1})
    batch_size = 512
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4
    val_check_interval = 0.1
    lr_mult = 10

    # for det token
    add_det_token = True
    det_token_num = 1



# Finetune for REC downstream task on referitgame
@ex.named_config
def task_finetune_referitgame_mdetr_coco_noaug():
    exp_name = "finetune_referitgame_mdetr_coco_noaug"
    datasets = ["referitgame"]
    train_transform_keys = ["coco"]
    val_transform_keys = ["coco"]
    loss_names = _loss_names({"rec": 1})
    batch_size = 512
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4
    val_check_interval = 0.1
    lr_mult = 10

    # for det token
    add_det_token = True
    det_token_num = 1



# Named configs for "etc" which are orthogonal to "env" and "task", need to be added at the end


@ex.named_config
def step25k():
    max_epoch = 100
    max_steps = 25000


@ex.named_config
def step50k():
    max_epoch = 100
    max_steps = 50000


@ex.named_config
def step100k():
    max_epoch = 100
    max_steps = 100000


@ex.named_config
def step200k():
    max_epoch = 200
    max_steps = 200000


@ex.named_config
def vit32_base():
    vit = "vit_base_patch32_384"
    patch_size = 32
    hidden_size = 768
    num_heads = 12
    num_layers = 12
