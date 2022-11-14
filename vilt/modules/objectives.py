import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import json
import tqdm
import functools

from torch.utils.data.distributed import DistributedSampler
from einops import rearrange

from vilt.modules.dist_utils import all_gather
from vilt.gadgets.flickr_eval import FlickrEvaluator
from vilt.transforms.coco import box_cxcywh_to_xyxy, generalized_box_iou

def cost_matrix_cosine(x, y, eps=1e-5):
    """Compute cosine distnace across every pairs of x, y (batched)
    [B, L_x, D] [B, L_y, D] -> [B, Lx, Ly]"""
    assert x.dim() == y.dim()
    assert x.size(0) == y.size(0)
    assert x.size(2) == y.size(2)
    x_norm = F.normalize(x, p=2, dim=-1, eps=eps)
    y_norm = F.normalize(y, p=2, dim=-1, eps=eps)
    cosine_sim = x_norm.matmul(y_norm.transpose(1, 2))
    cosine_dist = 1 - cosine_sim
    return cosine_dist


def trace(x):
    """ compute trace of input tensor (batched) """
    b, m, n = x.size()
    assert m == n
    mask = torch.eye(n, dtype=torch.bool, device=x.device).unsqueeze(0).expand_as(x)
    trace = x.masked_select(mask).contiguous().view(b, n).sum(dim=-1, keepdim=False)
    return trace


@torch.no_grad()
def ipot(C, x_len, x_pad, y_len, y_pad, joint_pad, beta, iteration, k):
    """ [B, M, N], [B], [B, M], [B], [B, N], [B, M, N]"""
    b, m, n = C.size()
    sigma = torch.ones(b, m, dtype=C.dtype, device=C.device) / x_len.unsqueeze(1)
    T = torch.ones(b, n, m, dtype=C.dtype, device=C.device)
    A = torch.exp(-C.transpose(1, 2) / beta)

    # mask padded positions
    sigma.masked_fill_(x_pad, 0)
    joint_pad = joint_pad.transpose(1, 2)
    T.masked_fill_(joint_pad, 0)
    A.masked_fill_(joint_pad, 0)

    # broadcastable lengths
    x_len = x_len.unsqueeze(1).unsqueeze(2)
    y_len = y_len.unsqueeze(1).unsqueeze(2)

    # mask to zero out padding in delta and sigma
    x_mask = (x_pad.to(C.dtype) * 1e4).unsqueeze(1)
    y_mask = (y_pad.to(C.dtype) * 1e4).unsqueeze(1)

    for _ in range(iteration):
        Q = A * T  # bs * n * m
        sigma = sigma.view(b, m, 1)
        for _ in range(k):
            delta = 1 / (y_len * Q.matmul(sigma).view(b, 1, n) + y_mask)
            sigma = 1 / (x_len * delta.matmul(Q) + x_mask)
        T = delta.view(b, n, 1) * Q * sigma
    T.masked_fill_(joint_pad, 0)
    return T


def optimal_transport_dist(
    txt_emb, img_emb, txt_pad, img_pad, beta=0.5, iteration=50, k=1
):
    """ [B, M, D], [B, N, D], [B, M], [B, N]"""
    cost = cost_matrix_cosine(txt_emb, img_emb)
    # mask the padded inputs
    joint_pad = txt_pad.unsqueeze(-1) | img_pad.unsqueeze(-2)
    cost.masked_fill_(joint_pad, 0)

    txt_len = (txt_pad.size(1) - txt_pad.sum(dim=1, keepdim=False)).to(dtype=cost.dtype)
    img_len = (img_pad.size(1) - img_pad.sum(dim=1, keepdim=False)).to(dtype=cost.dtype)

    T = ipot(
        cost.detach(), txt_len, txt_pad, img_len, img_pad, joint_pad, beta, iteration, k
    )
    distance = trace(cost.matmul(T.detach()))
    return distance


def compute_mlm(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=True, mask_image=False)
    mlm_logits = pl_module.mlm_score(infer["text_feats"])
    mlm_labels = infer["text_labels"]

    mlm_loss = F.cross_entropy(
        mlm_logits.view(-1, pl_module.hparams.config["vocab_size"]),
        mlm_labels.view(-1),
        ignore_index=-100,
    )

    ret = {
        "mlm_loss": mlm_loss,
        "mlm_logits": mlm_logits,
        "mlm_labels": mlm_labels,
        "mlm_ids": infer["text_ids"],
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mlm_loss")(ret["mlm_loss"])
    acc = getattr(pl_module, f"{phase}_mlm_accuracy")(
        ret["mlm_logits"], ret["mlm_labels"]
    )
    pl_module.log(f"mlm/{phase}/loss", loss)
    pl_module.log(f"mlm/{phase}/accuracy", acc)

    return ret


def compute_mpp(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=True)
    mpp_logits = pl_module.mpp_score(infer["image_feats"])
    mpp_logits = torch.stack(
        [
            mpp_logits[:, :, 0:256],
            mpp_logits[:, :, 256:512],
            mpp_logits[:, :, 512:768],
        ],
        dim=2,
    )
    mpp_labels = infer["image_labels"]

    mpp_loss = F.cross_entropy(
        mpp_logits.view(-1, 256),
        mpp_labels.view(-1),
        ignore_index=-100,
    )

    ret = {
        "mpp_loss": mpp_loss,
        "mpp_logits": mpp_logits,
        "mpp_labels": mpp_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mpp_loss")(ret["mpp_loss"])
    acc = getattr(pl_module, f"{phase}_mpp_accuracy")(
        ret["mpp_logits"], ret["mpp_labels"]
    )
    pl_module.log(f"mpp/{phase}/loss", loss)
    pl_module.log(f"mpp/{phase}/accuracy", acc)

    return ret


def compute_mppd(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=True)
    mppd_logits = pl_module.mppd_score(infer["image_feats"])
    mppd_labels = infer["image_labels_mppd"]
    filter_to_train = infer["image_labels"].float().mean(dim=-1) != -100

    labels = mppd_labels[filter_to_train]
    logits = mppd_logits[filter_to_train]
    mppd_loss = F.mse_loss(logits, labels)

    ret = {
        "mppd_loss": mppd_loss,
        "mppd_logits": mppd_logits,
        "mppd_labels": mppd_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mppd_loss")(ret["mppd_loss"])
    pl_module.log(f"mppd/{phase}/loss", loss)

    return ret


def compute_mpfr(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=True)
    mpfr_logits = pl_module.mpfr_score(infer["image_feats"])
    mpfr_labels = infer["image_labels_mpfr"]
    filter_to_train = infer["image_labels"].float().mean(dim=-1) != -100

    labels = mpfr_labels[filter_to_train]
    logits = mpfr_logits[filter_to_train]
    mpfr_loss = F.mse_loss(logits, labels)

    ret = {
        "mpfr_loss": mpfr_loss,
        "mpfr_logits": mpfr_logits,
        "mpfr_labels": mpfr_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mpfr_loss")(ret["mpfr_loss"])
    pl_module.log(f"mpfr/{phase}/loss", loss)

    return ret


def compute_itm_wpa(pl_module, batch):
    pos_len = len(batch["text"]) // 2
    neg_len = len(batch["text"]) - pos_len
    itm_labels = torch.cat([torch.ones(pos_len), torch.zeros(neg_len)]).to(
        pl_module.device
    )
    itm_labels = itm_labels[torch.randperm(itm_labels.size(0))]

    itm_images = [
        torch.stack(
            [
                ti if itm_labels[i] == 1 else fi
                for i, (ti, fi) in enumerate(zip(bti, bfi))
            ]
        )
        for bti, bfi in zip(batch["image"], batch["false_image_0"])
    ]

    batch = {k: v for k, v in batch.items()}
    batch["image"] = itm_images

    infer = pl_module.infer(batch, mask_text=False, mask_image=False)

    with torch.cuda.amp.autocast(enabled=False):
        txt_emb, img_emb = infer["text_feats"], infer["image_feats"]
        txt_mask, img_mask = infer["text_masks"].bool(), infer["image_masks"].bool()
        for i, _len in enumerate(txt_mask.sum(dim=1)):
            txt_mask[i, _len - 1] = False
        txt_mask[:, 0] = False
        img_mask[:, 0] = False
        if "deit" in pl_module.hparams.config["vit"]:
            img_mask[:, 1] = False
        txt_pad, img_pad = ~txt_mask, ~img_mask

        cost = cost_matrix_cosine(txt_emb.float(), img_emb.float())
        joint_pad = txt_pad.unsqueeze(-1) | img_pad.unsqueeze(-2)
        cost.masked_fill_(joint_pad, 0)

        txt_len = (txt_pad.size(1) - txt_pad.sum(dim=1, keepdim=False)).to(
            dtype=cost.dtype
        )
        img_len = (img_pad.size(1) - img_pad.sum(dim=1, keepdim=False)).to(
            dtype=cost.dtype
        )
        T = ipot(
            cost.detach(), txt_len, txt_pad, img_len, img_pad, joint_pad, 0.5, 50, 1
        )
        distance = trace(cost.matmul(T.detach()))

    dist_pos = distance.masked_select(itm_labels == 1)
    dist_neg = distance.masked_select(itm_labels == 0)
    ot_loss = (dist_pos.sum() - dist_neg.sum()) / (dist_pos.size(0) + dist_neg.size(0))

    itm_logits = pl_module.itm_score(infer["cls_feats"])
    itm_loss = F.cross_entropy(itm_logits, itm_labels.long())

    ret = {
        "itm_loss": itm_loss,
        "itm_wpa_loss": 0.1 * ot_loss,
        "itm_logits": itm_logits,
        "itm_labels": itm_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_itm_loss")(ret["itm_loss"])
    wpa_loss = getattr(pl_module, f"{phase}_itm_wpa_loss")(ret["itm_wpa_loss"])
    acc = getattr(pl_module, f"{phase}_itm_accuracy")(
        ret["itm_logits"], ret["itm_labels"]
    )
    pl_module.log(f"itm/{phase}/loss", loss)
    pl_module.log(f"itm/{phase}/wpa_loss", wpa_loss)
    pl_module.log(f"itm/{phase}/accuracy", acc)

    return ret


def compute_imgcls(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=False)
    imgcls_logits = pl_module.img_classifier(infer["cls_feats"])
    imgcls_labels = batch["label"]
    imgcls_labels = torch.tensor(imgcls_labels).to(pl_module.device).long()
    imgcls_loss = F.cross_entropy(imgcls_logits, imgcls_labels)

    ret = {
        "imgcls_loss": imgcls_loss,
        "imgcls_logits": imgcls_logits,
        "imgcls_labels": imgcls_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_imgcls_loss")(ret["imgcls_loss"])
    acc = getattr(pl_module, f"{phase}_imgcls_accuracy")(
        ret["imgcls_logits"], ret["imgcls_labels"]
    )
    pl_module.log(f"imgcls/{phase}/loss", loss)
    pl_module.log(f"imgcls/{phase}/accuracy", acc)

    return ret


def compute_vqa(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=False)
    vqa_logits = pl_module.vqa_classifier(infer["cls_feats"])
    vqa_targets = torch.zeros(
        len(vqa_logits), pl_module.hparams.config["vqav2_label_size"]
    ).to(pl_module.device)

    vqa_labels = batch["vqa_labels"]
    vqa_scores = batch["vqa_scores"]

    for i, (_label, _score) in enumerate(zip(vqa_labels, vqa_scores)):
        for l, s in zip(_label, _score):
            vqa_targets[i, l] = s

    vqa_loss = (
        F.binary_cross_entropy_with_logits(vqa_logits, vqa_targets)
        * vqa_targets.shape[1]
    )  # https://github.com/jnhwkim/ban-vqa/blob/master/train.py#L19

    ret = {
        "vqa_loss": vqa_loss,
        "vqa_logits": vqa_logits,
        "vqa_targets": vqa_targets,
        "vqa_labels": vqa_labels,
        "vqa_scores": vqa_scores,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_vqa_loss")(ret["vqa_loss"])
    score = getattr(pl_module, f"{phase}_vqa_score")(
        ret["vqa_logits"], ret["vqa_targets"]
    )
    pl_module.log(f"vqa/{phase}/loss", loss)
    pl_module.log(f"vqa/{phase}/score", score)

    return ret


def compute_nlvr2(pl_module, batch):
    infer1 = pl_module.infer(
        batch, mask_text=False, mask_image=False, image_token_type_idx=1
    )
    infer2 = pl_module.infer(
        batch, mask_text=False, mask_image=False, image_token_type_idx=2
    )

    cls_feats = torch.cat([infer1["cls_feats"], infer2["cls_feats"]], dim=-1)
    nlvr2_logits = pl_module.nlvr2_classifier(cls_feats)

    nlvr2_labels = batch["answers"]
    nlvr2_labels = torch.tensor(nlvr2_labels).to(pl_module.device).long()
    nlvr2_loss = F.cross_entropy(nlvr2_logits, nlvr2_labels)

    ret = {
        "nlvr2_loss": nlvr2_loss,
        "nlvr2_logits": nlvr2_logits,
        "nlvr2_labels": nlvr2_labels,
    }

    phase = "train" if pl_module.training else "val"

    if phase == "train":
        loss = getattr(pl_module, f"{phase}_nlvr2_loss")(ret["nlvr2_loss"])
        acc = getattr(pl_module, f"{phase}_nlvr2_accuracy")(
            ret["nlvr2_logits"], ret["nlvr2_labels"]
        )
        pl_module.log(f"nlvr2/{phase}/loss", loss)
        pl_module.log(f"nlvr2/{phase}/accuracy", acc)
    else:
        dev_batches = [i for i, n in enumerate(batch["table_name"]) if "dev" in n]
        test_batches = [i for i, n in enumerate(batch["table_name"]) if "test" in n]

        if dev_batches:
            dev_loss = getattr(pl_module, f"dev_nlvr2_loss")(
                F.cross_entropy(
                    ret["nlvr2_logits"][dev_batches], ret["nlvr2_labels"][dev_batches]
                )
            )
            dev_acc = getattr(pl_module, f"dev_nlvr2_accuracy")(
                ret["nlvr2_logits"][dev_batches], ret["nlvr2_labels"][dev_batches]
            )
            pl_module.log(f"nlvr2/dev/loss", dev_loss)
            pl_module.log(f"nlvr2/dev/accuracy", dev_acc)
        if test_batches:
            test_loss = getattr(pl_module, f"test_nlvr2_loss")(
                F.cross_entropy(
                    ret["nlvr2_logits"][test_batches], ret["nlvr2_labels"][test_batches]
                )
            )
            test_acc = getattr(pl_module, f"test_nlvr2_accuracy")(
                ret["nlvr2_logits"][test_batches], ret["nlvr2_labels"][test_batches]
            )
            pl_module.log(f"nlvr2/test/loss", test_loss)
            pl_module.log(f"nlvr2/test/accuracy", test_acc)

    return ret

def compute_mdetr_pretrain(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=False)
    num_classes = pl_module.num_classes
    text_feats = infer["text_feats"] 
    image_feats = infer["image_feats"] 
    cls_feats = infer["cls_feats"]
    bs, queries = infer["det_token_feats"].shape[:2]
    hidden_size = pl_module.hparams.config["hidden_size"]

    # input for the bbox regression
    if pl_module.hparams.config["reg_input"] == "det_token":
        pred_cls = pl_module.classifier(infer["det_token_feats"].contiguous().view(-1,hidden_size)).view(-1, queries, num_classes+1)
        pred_bbox = pl_module.bbox_regressor(infer["det_token_feats"].contiguous().view(-1,hidden_size)).view(-1, queries, 4)
    elif pl_module.hparams.config["reg_input"] == "det_token_with_cls":
            pred_cls = pl_module.classifier(infer["det_token_feats"].contiguous().view(-1,hidden_size) + cls_feats.repeat_interleave(queries, dim=0)).view(-1, queries, num_classes+1)
            pred_bbox = pl_module.bbox_regressor(infer["det_token_feats"].contiguous().view(-1,hidden_size) + cls_feats.repeat_interleave(queries, dim=0)).view(-1, queries, 4)
    elif pl_module.hparams.config["reg_input"] == "det_token_concat_cls":
            pred_cls = pl_module.classifier(torch.cat((infer["det_token_feats"].contiguous().view(-1,hidden_size) , cls_feats.repeat_interleave(queries, dim=0)), dim=1)).view(-1, queries, num_classes+1)
            pred_bbox = pl_module.bbox_regressor(torch.cat((infer["det_token_feats"].contiguous().view(-1,hidden_size) , cls_feats.repeat_interleave(queries, dim=0)), dim=1 )).view(-1, queries, 4)

    
    # use contrastive alignment from mdetr
    if pl_module.hparams.config["mdetr_use_alignment"]:
        proj_tokens = F.normalize(pl_module.contrastive_align_projection_text(text_feats), p=2, dim=-1 )
        proj_queries = F.normalize(pl_module.contrastive_align_projection_image(infer["det_token_feats"]), p=2, dim=-1)

    gt_bbox = [torch.FloatTensor(batch["gt_bbox"][i]).to(pl_module.device) for i in range(len(batch["gt_bbox"]))]

    positive_map = torch.cat([batch["positive_map"][i] for i in range(len(batch["positive_map"]))], dim=0)

    # patch_level_alignment
    if pl_module.hparams.config["patch_level_alignment"]:
        temperature = 0.07 
        image_feats_no_cls = image_feats[:,1:,:]
        patch_length = image_feats_no_cls.shape[1]
        patch_img_feat = F.normalize(pl_module.patch_contrastive_align_projection_image(image_feats_no_cls), dim=2)

        text_feat = F.normalize(pl_module.patch_contrastive_align_projection_text(text_feats), dim=2)
        patch_text_exp_logits = torch.exp(torch.matmul(text_feat, patch_img_feat.permute(0,2,1))/temperature)
        patch_text_label = torch.zeros(bs, pl_module.hparams.config["max_text_len"], patch_length).to(pl_module.device)
        for b in range(bs):
            for box in range(batch["positive_map"][b].shape[0]):
                bbox_patch_label_idx = (batch["bbox_patch_label"][b].permute(1,0).bool()[box] != 0).nonzero().view(-1)
                pos_map_idx = (batch["positive_map"][b][box, :pl_module.hparams.config["max_text_len"]] != 0).nonzero().view(-1)
                patch_text_label[b, pos_map_idx.repeat_interleave(len(bbox_patch_label_idx)),
                        bbox_patch_label_idx.repeat(len(pos_map_idx))] = 1

        # Compute pos patch and neg patch for a token
        neg_patches_feats = patch_text_exp_logits*(1-patch_text_label)
        pos_patches_feats = patch_text_exp_logits*patch_text_label
        denominator = pos_patches_feats + torch.sum(neg_patches_feats, dim=2).view(bs,-1,1).expand(-1,-1,neg_patches_feats.shape[2])

        # (1-patch_text_label) to aviod nan by torch.log
        token_patch_alignment_loss = -torch.sum(torch.log((pos_patches_feats + (1-patch_text_label))/denominator)*patch_text_label)/torch.sum(patch_text_label)  
      
        # Compute pos token and neg token for a patch
        patch_text_label_T = patch_text_label.permute(0,2,1)
        patch_text_exp_logits_T = patch_text_exp_logits.permute(0,2,1)
        neg_patches_feats_T = patch_text_exp_logits_T*(1-patch_text_label_T)
        pos_patches_feats_T = patch_text_exp_logits_T*patch_text_label_T
        denominator_T = pos_patches_feats_T + torch.sum(neg_patches_feats_T, dim=2).view(bs,-1,1).expand(-1,-1,neg_patches_feats_T.shape[2])

        # (1-patch_text_label) to aviod nan by torch.log
        patch_token_alignment_loss = -torch.sum(torch.log((pos_patches_feats_T + (1-patch_text_label_T))/denominator_T)*patch_text_label_T)/torch.sum(patch_text_label_T)  
        tot_loss_patch_text_alignment = (pl_module.hparams.config["patch_level_alignment_weight1"]*token_patch_alignment_loss + pl_module.hparams.config["patch_level_alignment_weight2"]*patch_token_alignment_loss)/2

    elif pl_module.hparams.config["patch_level_alignment_KL"]:
        temperature = 0.07 
        image_feats_no_cls = image_feats[:,1:,:]
        patch_length = image_feats_no_cls.shape[1]
        patch_img_feat = F.normalize(pl_module.patch_contrastive_align_projection_image(image_feats_no_cls), dim=2)

        text_feat = F.normalize(pl_module.patch_contrastive_align_projection_text(text_feats), dim=2)
        patch_text_logits = torch.matmul(text_feat, patch_img_feat.permute(0,2,1))/temperature

        patch_text_label = torch.zeros(bs, pl_module.hparams.config["max_text_len"], patch_length).to(pl_module.device)
        for b in range(bs):
            for box in range(batch["positive_map"][b].shape[0]):
                bbox_patch_label_idx = (batch["bbox_patch_label"][b].permute(1,0).bool()[box] != 0).nonzero().view(-1)
                pos_map_idx = (batch["positive_map"][b][box, :pl_module.hparams.config["max_text_len"]] != 0).nonzero().view(-1)
                patch_text_label[b, pos_map_idx.repeat_interleave(len(bbox_patch_label_idx)),
                bbox_patch_label_idx.repeat(len(pos_map_idx))] = 1

        # Compute pos patch and neg patch for a token
        pred_log_token_patch_prob = F.log_softmax(patch_text_logits ,dim=2)
        # clamping to avoid nan
        gt_token_patch_prob = patch_text_label/torch.clamp(torch.sum(patch_text_label, dim=2, keepdim=True), min =1).repeat(1,1,patch_length)
        token_patch_alignment_loss = F.kl_div(pred_log_token_patch_prob, gt_token_patch_prob, reduction="sum")/torch.sum(gt_token_patch_prob)

        # Compute pos token and neg token for a patch
        pred_log_patch_token_prob = F.log_softmax(patch_text_logits ,dim=1)
        gt_patch_token_prob = patch_text_label/torch.clamp(torch.sum(patch_text_label, dim=1, keepdim=True), min =1).repeat(1, pl_module.hparams.config["max_text_len"] ,1)
        patch_token_alignment_loss = F.kl_div(pred_log_patch_token_prob, gt_patch_token_prob, reduction="sum")/torch.sum(gt_patch_token_prob)

        tot_loss_patch_text_alignment = pl_module.hparams.config["patch_level_alignment_weight1"]*token_patch_alignment_loss + pl_module.hparams.config["patch_level_alignment_weight2"]*patch_token_alignment_loss
 

    gt_label = [torch.FloatTensor(batch["category_id"][i]).to(pl_module.device) for i in range(len(batch["category_id"]))]
    pred_bbox = pred_bbox.sigmoid() # ensure w, h >= 0
    outputs = {"pred_boxes": pred_bbox,  "pred_logits": pred_cls }
    if pl_module.hparams.config["mdetr_use_alignment"]:
        outputs["proj_queries"] = proj_queries
        outputs["proj_tokens"] = proj_tokens
        tokenized = pl_module.tokenizer.batch_encode_plus(batch["text"], padding="longest", return_tensors="pt").to(pl_module.device)
        outputs["tokenized"] = tokenized
    

    tokens_positive = batch["tokens_positive"]
    targets = [{"boxes":gt_bbox[i], "tokens_positive":tokens_positive[i], "labels":gt_label[i]} for i in range(len(gt_bbox))]
    num_boxes = sum(len(t["boxes"]) for t in targets)

    # compute losses
    losses_dict = pl_module.criterion(outputs, targets, positive_map)

    loss = sum(losses_dict[k] * pl_module.matcher.weight_dict[k] for k in losses_dict.keys() if k in pl_module.matcher.weight_dict)
    if pl_module.hparams.config["patch_level_alignment"] or pl_module.hparams.config["patch_level_alignment_KL"]:
        loss += tot_loss_patch_text_alignment
 
    ret = {
                "loss": loss,
                "loss_bbox":losses_dict['loss_bbox'],
                "loss_giou":losses_dict['loss_giou'],
                "loss_class":losses_dict['loss_ce'],
                "pred_bbox": pred_bbox,
                "pred_bbox_all": outputs["pred_boxes"],
                "pred_logits": outputs["pred_logits"],
                "gt_bbox": gt_bbox,
                "label": gt_label,
            }

    if pl_module.hparams.config["patch_level_alignment"] or pl_module.hparams.config["patch_level_alignment_KL"]:
        ret["patch_text_alignment_loss"] = tot_loss_patch_text_alignment

    # Logging
    loss_name = "mdetr_pretrain"
    phase = "train" if pl_module.training else "val"
    if phase != "train":
        det_acc = getattr(pl_module, f"{phase}_{loss_name}_detacc")(outputs["pred_boxes"], ret["gt_bbox"], batch["width"], batch["height"] , _pred_logits=outputs["pred_logits"])
        pl_module.log(f"{loss_name}/{phase}/detacc", det_acc)

    pl_module.log(f"{loss_name}/{phase}/loss", loss)
    pl_module.log(f"{loss_name}/{phase}/loss_ce", losses_dict['loss_ce'])
    pl_module.log(f"{loss_name}/{phase}/loss_bbox", losses_dict['loss_bbox'])
    pl_module.log(f"{loss_name}/{phase}/loss_giou", losses_dict['loss_giou'])
    if pl_module.hparams.config["mdetr_use_alignment"]:
        pl_module.log(f"{loss_name}/{phase}/loss_contrastive_align", losses_dict['loss_contrastive_align'])
    if pl_module.hparams.config["patch_level_alignment"] or pl_module.hparams.config["patch_level_alignment_KL"]:
        pl_module.log(f"{loss_name}/{phase}/patch_text_alignment_loss", tot_loss_patch_text_alignment)
 
  
    return ret


def compute_mdetr_pretrain_ablade(pl_module, batch):
    
    infer = pl_module.infer(batch, mask_text=False, mask_image=False)
    text_feats = infer["text_feats"] 
    image_feats = infer["image_feats"] 
    cls_feats = infer["cls_feats"]
    hidden_size = pl_module.hparams.config["hidden_size"]

    # input for the bbox regression
    if pl_module.hparams.config["reg_input"] == "image":
        pred_bbox = pl_module.bbox_regressor(pl_module.image_pooler(infer["image_feats"]))
    elif pl_module.hparams.config["reg_input"] == "text":
        pred_bbox = pl_module.bbox_regressor(pl_module.text_pooler(infer["text_feats"]))
    elif  pl_module.hparams.config["reg_input"] == "cls":
        pred_bbox = pl_module.bbox_regressor(infer["cls_feats"])
    
    pred_bbox = pred_bbox.sigmoid() # ensure w, h >= 0
    gt_bbox = [torch.FloatTensor(batch["gt_bbox"][i]).to(pl_module.device) for i in range(len(batch["gt_bbox"]))]
    gt_label = [torch.FloatTensor(batch["category_id"][i]).to(pl_module.device) for i in range(len(batch["category_id"]))]
    tokens_positive = batch["tokens_positive"]

    targets = [{"boxes":gt_bbox[i], "tokens_positive":tokens_positive[i], "labels":gt_label[i]} for i in range(len(gt_bbox))]
    num_boxes = sum(len(t["boxes"]) for t in targets)

    # compute losses
    src_boxes = pred_bbox
    target_boxes = torch.cat(gt_bbox, dim=0)
    loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")  
    losses_dict = {}
    losses_dict["loss_bbox"] = loss_bbox.sum() / num_boxes
    losses_dict["loss_giou"] = 0 
    '''
    loss_giou = 1 - torch.diag(
        generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
    )
    losses_dict["loss_giou"] = loss_giou.sum() / num_boxes    
    loss = losses_dict["loss_bbox"]*pl_module.hparams.config["matchingloss_bbox_weight"] + losses_dict["loss_giou"]*pl_module.hparams.config["matchingloss_giou_weight"]
    '''
    loss = losses_dict["loss_bbox"]
    
    ret = {
                "loss": loss,
                "loss_bbox":losses_dict['loss_bbox'],
                "loss_giou":losses_dict['loss_giou'],
                "pred_bbox": src_boxes,
                "gt_bbox": target_boxes,
                "label": gt_label,
            }


    # Logging
    loss_name = "mdetr_pretrain_ablade"
    phase = "train" if pl_module.training else "val"
    if phase != "train":
        det_acc = getattr(pl_module, f"{phase}_{loss_name}_detacc")(src_boxes, target_boxes, batch["width"], batch["height"])
        pl_module.log(f"{loss_name}/{phase}/detacc", det_acc)

    pl_module.log(f"{loss_name}/{phase}/loss", loss)
    pl_module.log(f"{loss_name}/{phase}/loss_bbox", losses_dict['loss_bbox'])
    pl_module.log(f"{loss_name}/{phase}/loss_giou", losses_dict['loss_giou'])
  
    return ret


# for referring expression comprehension task
def compute_rec(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=False)
    num_classes = pl_module.num_classes
    text_feats = infer["text_feats"] 
    image_feats = infer["image_feats"] 
    cls_feats = infer["cls_feats"]
    bs, queries = infer["det_token_feats"].shape[:2]
    hidden_size = pl_module.hparams.config["hidden_size"]

    # input for the bbox regression
    if pl_module.hparams.config["reg_input"] == "det_token":
        pred_cls = pl_module.classifier(infer["det_token_feats"].contiguous().view(-1,hidden_size)).view(-1, queries, num_classes+1)
        pred_bbox = pl_module.bbox_regressor(infer["det_token_feats"].contiguous().view(-1,hidden_size)).view(-1, queries, 4)
    elif pl_module.hparams.config["reg_input"] == "det_token_with_cls":
            pred_cls = pl_module.classifier(infer["det_token_feats"].contiguous().view(-1,hidden_size) + cls_feats.repeat_interleave(queries, dim=0)).view(-1, queries, num_classes+1)
            pred_bbox = pl_module.bbox_regressor(infer["det_token_feats"].contiguous().view(-1,hidden_size) + cls_feats.repeat_interleave(queries, dim=0)).view(-1, queries, 4)
    elif pl_module.hparams.config["reg_input"] == "det_token_concat_cls":
            pred_cls = pl_module.classifier(torch.cat((infer["det_token_feats"].contiguous().view(-1,hidden_size) , cls_feats.repeat_interleave(queries, dim=0)), dim=1)).view(-1, queries, num_classes+1)
            pred_bbox = pl_module.bbox_regressor(torch.cat((infer["det_token_feats"].contiguous().view(-1,hidden_size) , cls_feats.repeat_interleave(queries, dim=0)), dim=1 )).view(-1, queries, 4)
    # use contrastive alignment from mdetr  
    if pl_module.hparams.config["mdetr_use_alignment"]:
        proj_tokens = F.normalize(pl_module.contrastive_align_projection_text(text_feats), p=2, dim=-1 )
        proj_queries = F.normalize(pl_module.contrastive_align_projection_image(infer["det_token_feats"]), p=2, dim=-1)

    gt_bbox = [torch.FloatTensor(batch["gt_bbox"][i]).to(pl_module.device) for i in range(len(batch["gt_bbox"]))]

    positive_map = torch.cat([batch["positive_map"][i] for i in range(len(batch["positive_map"]))], dim=0)

    # patch_level_alignment
    if pl_module.hparams.config["patch_level_alignment"]:
        temperature = 0.07 
        image_feats_no_cls = image_feats[:,1:,:]
        patch_length = image_feats_no_cls.shape[1]
        patch_img_feat = F.normalize(pl_module.patch_contrastive_align_projection_image(image_feats_no_cls), dim=2)

        text_feat = F.normalize(pl_module.patch_contrastive_align_projection_text(text_feats), dim=2)
        patch_text_exp_logits = torch.exp(torch.matmul(text_feat, patch_img_feat.permute(0,2,1))/temperature)
        patch_text_label = torch.zeros(bs, pl_module.hparams.config["max_text_len"], patch_length).to(pl_module.device)

        for b in range(bs):
            for box in range(batch["positive_map"][b].shape[0]):
                ''' For pytorch version 1.5 '''
                #bbox_patch_label_idx = (batch["bbox_patch_label"][b].permute(1,0).bool()[box] != 0).nonzero().view(-1)
                bbox_patch_label_idx = torch.nonzero((batch["bbox_patch_label"][b].permute(1,0).bool()[box] != 0), as_tuple=False).view(-1)
                #pos_map_idx = (batch["positive_map"][b][box, :pl_module.hparams.config["max_text_len"]] != 0).nonzero().view(-1)
                pos_map_idx = torch.nonzero((batch["positive_map"][b][box, :pl_module.hparams.config["max_text_len"]] != 0), as_tuple=False).view(-1)

                patch_text_label[b, pos_map_idx.repeat_interleave(len(bbox_patch_label_idx)),
                        bbox_patch_label_idx.repeat(len(pos_map_idx))] = 1

        # Compute pos patch and neg patch for a token
        neg_patches_feats = patch_text_exp_logits*(1-patch_text_label)
        pos_patches_feats = patch_text_exp_logits*patch_text_label
        denominator = pos_patches_feats + torch.sum(neg_patches_feats, dim=2).view(bs,-1,1).expand(-1,-1,neg_patches_feats.shape[2])

        # (1-patch_text_label) to aviod nan by torch.log
        token_patch_alignment_loss = -torch.sum(torch.log((pos_patches_feats + (1-patch_text_label))/denominator)*patch_text_label)/torch.sum(patch_text_label)  
      
        # Compute pos token and neg token for a patch
        patch_text_label_T = patch_text_label.permute(0,2,1)
        patch_text_exp_logits_T = patch_text_exp_logits.permute(0,2,1)
        neg_patches_feats_T = patch_text_exp_logits_T*(1-patch_text_label_T)
        pos_patches_feats_T = patch_text_exp_logits_T*patch_text_label_T
        denominator_T = pos_patches_feats_T + torch.sum(neg_patches_feats_T, dim=2).view(bs,-1,1).expand(-1,-1,neg_patches_feats_T.shape[2])

        # (1-patch_text_label) to aviod nan by torch.log
        patch_token_alignment_loss = -torch.sum(torch.log((pos_patches_feats_T + (1-patch_text_label_T))/denominator_T)*patch_text_label_T)/torch.sum(patch_text_label_T)  
        tot_loss_patch_text_alignment = (pl_module.hparams.config["patch_level_alignment_weight1"]*token_patch_alignment_loss + pl_module.hparams.config["patch_level_alignment_weight2"]*patch_token_alignment_loss)/2

    elif pl_module.hparams.config["patch_level_alignment_KL"]:
        temperature = 0.07 
        image_feats_no_cls = image_feats[:,1:,:]
        patch_length = image_feats_no_cls.shape[1]
        patch_img_feat = F.normalize(pl_module.patch_contrastive_align_projection_image(image_feats_no_cls), dim=2)

        text_feat = F.normalize(pl_module.patch_contrastive_align_projection_text(text_feats), dim=2)
        patch_text_logits = torch.matmul(text_feat, patch_img_feat.permute(0,2,1))/temperature

        patch_text_label = torch.zeros(bs, pl_module.hparams.config["max_text_len"], patch_length).to(pl_module.device)
        for b in range(bs):
            for box in range(batch["positive_map"][b].shape[0]):
                bbox_patch_label_idx = (batch["bbox_patch_label"][b].permute(1,0).bool()[box] != 0).nonzero().view(-1)
                pos_map_idx = (batch["positive_map"][b][box, :pl_module.hparams.config["max_text_len"]] != 0).nonzero().view(-1)
                patch_text_label[b, pos_map_idx.repeat_interleave(len(bbox_patch_label_idx)),
                bbox_patch_label_idx.repeat(len(pos_map_idx))] = 1

        # Compute pos patch and neg patch for a token
        pred_log_token_patch_prob = F.log_softmax(patch_text_logits ,dim=2)
        # clamping to avoid nan
        gt_token_patch_prob = patch_text_label/torch.clamp(torch.sum(patch_text_label, dim=2, keepdim=True), min =1).repeat(1,1,patch_length)
        token_patch_alignment_loss = F.kl_div(pred_log_token_patch_prob, gt_token_patch_prob, reduction="sum")/torch.sum(gt_token_patch_prob)

        # Compute pos token and neg token for a patch
        pred_log_patch_token_prob = F.log_softmax(patch_text_logits ,dim=1)
        gt_patch_token_prob = patch_text_label/torch.clamp(torch.sum(patch_text_label, dim=1, keepdim=True), min =1).repeat(1, pl_module.hparams.config["max_text_len"] ,1)
        patch_token_alignment_loss = F.kl_div(pred_log_patch_token_prob, gt_patch_token_prob, reduction="sum")/torch.sum(gt_patch_token_prob)

        tot_loss_patch_text_alignment = pl_module.hparams.config["patch_level_alignment_weight1"]*token_patch_alignment_loss + pl_module.hparams.config["patch_level_alignment_weight2"]*patch_token_alignment_loss
 

    gt_label = [torch.FloatTensor(batch["category_id"][i]).to(pl_module.device) for i in range(len(batch["category_id"]))]
    pred_bbox = pred_bbox.sigmoid() # ensure w, h >= 0
    outputs = {"pred_boxes": pred_bbox,  "pred_logits": pred_cls }
    if pl_module.hparams.config["mdetr_use_alignment"]:
        outputs["proj_queries"] = proj_queries
        outputs["proj_tokens"] = proj_tokens
        tokenized = pl_module.tokenizer.batch_encode_plus(batch["text"], padding="longest", return_tensors="pt").to(pl_module.device)
        outputs["tokenized"] = tokenized
    

    tokens_positive = batch["tokens_positive"]
    targets = [{"boxes":gt_bbox[i], "tokens_positive":tokens_positive[i], "labels":gt_label[i]} for i in range(len(gt_bbox))]
    num_boxes = sum(len(t["boxes"]) for t in targets)

    # compute losses
    
    if not pl_module.hparams.config["compute_loss"]:
        ret = {
                "loss": 0,
                "loss_bbox": 0,
                "loss_giou":0,
                "loss_class":0,
                "pred_bbox": pred_bbox,
                "pred_bbox_all": outputs["pred_boxes"],
                "pred_logits": outputs["pred_logits"],
                "gt_bbox": gt_bbox,
                "label": gt_label
            }
        return ret


    
    losses_dict = pl_module.criterion(outputs, targets, positive_map)

    loss = sum(losses_dict[k] * pl_module.matcher.weight_dict[k] for k in losses_dict.keys() if k in pl_module.matcher.weight_dict)
    if pl_module.hparams.config["patch_level_alignment"] or pl_module.hparams.config["patch_level_alignment_KL"]:
        loss += tot_loss_patch_text_alignment
 
    ret = {
                "loss": loss,
               "loss_bbox":losses_dict['loss_bbox'],
                "loss_giou":losses_dict['loss_giou'],
                "loss_class":losses_dict['loss_ce'],
                "pred_bbox": pred_bbox,
                "pred_bbox_all": outputs["pred_boxes"],
                "pred_logits": outputs["pred_logits"],
                "gt_bbox": gt_bbox,
                "label": gt_label,
                "bbox_patch_label":batch["bbox_patch_label"] if "bbox_patch_label" in batch else None
            }

    if pl_module.hparams.config["patch_level_alignment"] or pl_module.hparams.config["patch_level_alignment_KL"]:
        ret["patch_text_alignment_loss"] = tot_loss_patch_text_alignment

    # logging
    loss_name = "rec"
 
    phase = "train" if pl_module.training else "val"
 
    #print(pl_module.hparams.config["datasets"])
  
    det_acc = getattr(pl_module, f"{phase}_{loss_name}_detacc")(outputs["pred_boxes"], ret["gt_bbox"], batch["width"], batch["height"] , _pred_logits=outputs["pred_logits"])

    pl_module.log(f"{loss_name}/{phase}/loss", loss)
    pl_module.log(f"{loss_name}/{phase}/loss_ce", losses_dict['loss_ce'])
    pl_module.log(f"{loss_name}/{phase}/loss_bbox", losses_dict['loss_bbox'])
    pl_module.log(f"{loss_name}/{phase}/loss_giou", losses_dict['loss_giou'])
    if pl_module.hparams.config["mdetr_use_alignment"]:
        pl_module.log(f"{loss_name}/{phase}/loss_contrastive_align", losses_dict['loss_contrastive_align'])
    pl_module.log(f"{loss_name}/{phase}/detacc", det_acc)

    return ret


def compute_irtr(pl_module, batch):
    is_training_phase = pl_module.training

    _bs, _c, _h, _w = batch["image"][0].shape
    false_len = pl_module.hparams.config["draw_false_text"]
    text_ids = torch.stack(
        [batch[f"false_text_{i}_ids"] for i in range(false_len)], dim=1
    )
    text_masks = torch.stack(
        [batch[f"false_text_{i}_masks"] for i in range(false_len)], dim=1
    )
    text_labels = torch.stack(
        [batch[f"false_text_{i}_labels"] for i in range(false_len)], dim=1
    )

    text_ids = torch.cat([batch["text_ids"].unsqueeze(1), text_ids], dim=1)
    text_masks = torch.cat([batch["text_masks"].unsqueeze(1), text_masks], dim=1)
    text_labels = torch.cat([batch["text_labels"].unsqueeze(1), text_labels], dim=1)
    images = batch["image"][0].unsqueeze(1).expand(_bs, false_len + 1, _c, _h, _w)

    infer = pl_module.infer(
        {
            "image": [rearrange(images, "bs fs c h w -> (bs fs) c h w")],
            "text_ids": rearrange(text_ids, "bs fs tl -> (bs fs) tl"),
            "text_masks": rearrange(text_masks, "bs fs tl -> (bs fs) tl"),
            "text_labels": rearrange(text_labels, "bs fs tl -> (bs fs) tl"),
        }
    )
    score = pl_module.rank_output(infer["cls_feats"])[:, 0]
    score = rearrange(score, "(bs fs) -> bs fs", bs=_bs, fs=false_len + 1)
    answer = torch.zeros(_bs).to(score).long()
    irtr_loss = F.cross_entropy(score, answer)

    ret = {
        "irtr_loss": irtr_loss,
    }

    phase = "train" if pl_module.training else "val"
    irtr_loss = getattr(pl_module, f"{phase}_irtr_loss")(ret["irtr_loss"])

    pl_module.log(f"irtr/{phase}/irtr_loss", irtr_loss)

    return ret

def compute_gqa_mdetr(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=False)
    num_classes = pl_module.num_classes
    text_feats = infer["text_feats"] 
    image_feats = infer["image_feats"] 
    cls_feats = infer["cls_feats"]
    bs, queries = infer["det_token_feats"].shape[:2]
    hidden_size = pl_module.hparams.config["hidden_size"]

    # for gqa the answer token is 6
    gqa_ans_det_num = 6
    queries = queries-gqa_ans_det_num # Detection queries token 
    obj_det_feat = infer["det_token_feats"][:,:queries,:]
    answer_feat = infer["det_token_feats"][:,queries:,:]
    pred_answer = {}
    
    # input for the bbox regression
    if pl_module.hparams.config["reg_input"] == "det_token":
        pred_cls = pl_module.classifier(obj_det_feat.contiguous().view(-1,hidden_size)).view(-1, queries, num_classes+1)
        pred_bbox = pl_module.bbox_regressor(obj_det_feat.contiguous().view(-1,hidden_size)).view(-1, queries, 4)
        pred_answer["pred_answer_type"] = pl_module.answer_type_head(answer_feat[:,0,:])
        pred_answer["pred_answer_obj"] = pl_module.answer_obj_head(answer_feat[:,1,:])
        pred_answer["pred_answer_rel"] = pl_module.answer_rel_head(answer_feat[:,2,:])
        pred_answer["pred_answer_attr"] = pl_module.answer_attr_head(answer_feat[:,3,:])
        pred_answer["pred_answer_cat"] = pl_module.answer_cat_head(answer_feat[:,4,:])
        pred_answer["pred_answer_global"] = pl_module.answer_global_head(answer_feat[:,5,:])

    elif pl_module.hparams.config["reg_input"] == "det_token_with_cls":
        pred_cls = pl_module.classifier(obj_det_feat.contiguous().view(-1,hidden_size) + cls_feats.repeat_interleave(queries, dim=0)).view(-1, queries, num_classes+1)
        pred_bbox = pl_module.bbox_regressor(obj_det_feat.contiguous().view(-1,hidden_size) + cls_feats.repeat_interleave(queries, dim=0)).view(-1, queries, 4)
        pred_answer["pred_answer_type"] = pl_module.answer_type_head(answer_feat[:,0,:] + cls_feats)
        pred_answer["pred_answer_obj"] = pl_module.answer_obj_head(answer_feat[:,1,:] + cls_feats)
        pred_answer["pred_answer_rel"] = pl_module.answer_rel_head(answer_feat[:,2,:] + cls_feats)
        pred_answer["pred_answer_attr"] = pl_module.answer_attr_head(answer_feat[:,3,:] + cls_feats)
        pred_answer["pred_answer_cat"] = pl_module.answer_cat_head(answer_feat[:,4,:] + cls_feats)
        pred_answer["pred_answer_global"] = pl_module.answer_global_head(answer_feat[:,5,:] + cls_feats)

    elif pl_module.hparams.config["reg_input"] == "det_token_concat_cls":
        pred_cls = pl_module.classifier(torch.cat((obj_det_feat.contiguous().view(-1,hidden_size) , cls_feats.repeat_interleave(queries, dim=0)), dim=1)).view(-1, queries, num_classes+1)
        pred_bbox = pl_module.bbox_regressor(torch.cat((obj_det_feat.contiguous().view(-1,hidden_size) , cls_feats.repeat_interleave(queries, dim=0)), dim=1 )).view(-1, queries, 4)

        pred_answer["pred_answer_type"] = pl_module.answer_type_head(torch.cat((answer_feat[:,0,:], cls_feats), dim=1))
        pred_answer["pred_answer_obj"] = pl_module.answer_obj_head(torch.cat((answer_feat[:,1,:], cls_feats), dim=1))
        pred_answer["pred_answer_rel"] = pl_module.answer_rel_head(torch.cat((answer_feat[:,2,:], cls_feats), dim=1))
        pred_answer["pred_answer_attr"] = pl_module.answer_attr_head(torch.cat((answer_feat[:,3,:], cls_feats), dim=1))
        pred_answer["pred_answer_cat"] = pl_module.answer_cat_head(torch.cat((answer_feat[:,4,:], cls_feats), dim=1))
        pred_answer["pred_answer_global"] = pl_module.answer_global_head(torch.cat((answer_feat[:,5,:], cls_feats), dim=1))

    # use contrastive alignment from mdetr
    if pl_module.hparams.config["mdetr_use_alignment"]:
        proj_tokens = F.normalize(pl_module.contrastive_align_projection_text(text_feats), p=2, dim=-1 )
        proj_queries = F.normalize(pl_module.contrastive_align_projection_image(infer["det_token_feats"]), p=2, dim=-1)

    # Patch level alignment
    if pl_module.hparams.config["patch_level_alignment_KL"] and pl_module.training:
        temperature = 0.07 
        image_feats_no_cls = image_feats[:,1:,:]
        patch_length = image_feats_no_cls.shape[1]
        patch_img_feat = F.normalize(pl_module.patch_contrastive_align_projection_image(image_feats_no_cls), dim=2)

        text_feat = F.normalize(pl_module.patch_contrastive_align_projection_text(text_feats), dim=2)
        patch_text_logits = torch.matmul(text_feat, patch_img_feat.permute(0,2,1))/temperature

        patch_text_label = torch.zeros(bs, pl_module.hparams.config["max_text_len"], patch_length).to(pl_module.device)
        for b in range(bs):
            for box in range(batch["positive_map"][b].shape[0]):
                bbox_patch_label_idx = (batch["bbox_patch_label"][b].permute(1,0).bool()[box] != 0).nonzero().view(-1)
                pos_map_idx = (batch["bbox_patch_label"][b][box, :pl_module.hparams.config["max_text_len"]] != 0).nonzero().view(-1)
                patch_text_label[b, pos_map_idx.repeat_interleave(len(bbox_patch_label_idx)),
                                    bbox_patch_label_idx.repeat(len(pos_map_idx))] = 1

        # Compute pos patch and neg patch for a token
        pred_log_token_patch_prob = F.log_softmax(patch_text_logits ,dim=2)
        # clamping to avoid nan
        gt_token_patch_prob = patch_text_label/torch.clamp(torch.sum(patch_text_label, dim=2, keepdim=True), min =1).repeat(1,1,patch_length)
        token_patch_alignment_loss = F.kl_div(pred_log_token_patch_prob, gt_token_patch_prob, reduction="sum")/torch.sum(gt_token_patch_prob)

        # Compute pos token and neg token for a patch
        pred_log_patch_token_prob = F.log_softmax(patch_text_logits ,dim=1)
        gt_patch_token_prob = patch_text_label/torch.clamp(torch.sum(patch_text_label, dim=1, keepdim=True), min =1).repeat(1, pl_module.hparams.config["max_text_len"] ,1)
        patch_token_alignment_loss = F.kl_div(pred_log_patch_token_prob, gt_patch_token_prob, reduction="sum")/torch.sum(gt_patch_token_prob)

        tot_loss_patch_text_alignment = pl_module.hparams.config["patch_level_alignment_weight1"]*token_patch_alignment_loss + pl_module.hparams.config["patch_level_alignment_weight2"]*patch_token_alignment_loss
 

    gt_bbox = [torch.FloatTensor(batch["gt_bbox"][i]).to(pl_module.device) for i in range(len(batch["gt_bbox"]))]

    positive_map = torch.cat([batch["positive_map"][i] for i in range(len(batch["positive_map"]))], dim=0)

    gt_label = [torch.FloatTensor(batch["category_id"][i]).to(pl_module.device) for i in range(len(batch["category_id"]))]
    #pred_bbox = pred_bbox.clamp(0.0) # ensure w, h >= 0
    pred_bbox = pred_bbox.sigmoid() # ensure w, h >= 0
    outputs = {"pred_boxes": pred_bbox,  "pred_logits": pred_cls }
    if pl_module.hparams.config["mdetr_use_alignment"]:
        outputs["proj_queries"] = proj_queries
        outputs["proj_tokens"] = proj_tokens
        tokenized = pl_module.tokenizer.batch_encode_plus(batch["text"], padding="longest", return_tensors="pt").to(pl_module.device)
        outputs["tokenized"] = tokenized
    
    tokens_positive = batch["tokens_positive"]
    targets = [{"boxes":gt_bbox[i], "tokens_positive":tokens_positive[i], "labels":gt_label[i]} for i in range(len(gt_bbox))]
    num_boxes = sum(len(t["boxes"]) for t in targets)
 
    # compute losses
    losses_dict = {}
    if not pl_module.hparams.config["test_only"]:
        losses_dict = pl_module.criterion(outputs, targets, positive_map)
    answer = {}
    for i in ["answer_type", "answer_obj", "answer_rel", "answer_attr", "answer_cat", "answer_global"]:
        answer[i] = torch.LongTensor(batch[i]).to(pl_module.device)
    answer_loss_dict = pl_module.answer_criterion(pred_answer, answer)
    answer_loss = sum(answer_loss_dict[k] * pl_module.matcher.weight_dict[k] for k in answer_loss_dict.keys() if k in pl_module.matcher.weight_dict)
    losses_dict.update(answer_loss_dict)
 
    loss = sum(losses_dict[k] * pl_module.matcher.weight_dict[k] for k in losses_dict.keys() if k in pl_module.matcher.weight_dict)
    if pl_module.hparams.config["patch_level_alignment_KL"] and pl_module.training:
        loss += tot_loss_patch_text_alignment

    # for finetuning on the question head with cross entropy loss on the QA 
    # no bbox related losses
    if pl_module.hparams.config["qa_loss_only"]:
        loss = answer_loss
        losses_dict['loss_bbox'] = 0
        losses_dict['loss_giou'] = 0
        losses_dict['loss_ce'] = 0

    if not pl_module.hparams.config["test_only"]:
        ret = {
                "loss": loss,
                "loss_bbox":losses_dict['loss_bbox'],
                "loss_giou":losses_dict['loss_giou'],
                "loss_class":losses_dict['loss_ce'],
                "pred_bbox": pred_bbox,
                "pred_bbox_all": outputs["pred_boxes"],
                "pred_logits": outputs["pred_logits"],
                "gt_bbox": gt_bbox,
                "gt_label": gt_label,
                "pred_answer": pred_answer
            }
        if pl_module.hparams.config["patch_level_alignment_KL"] and pl_module.training:
            ret["patch_text_alignment_loss"] = tot_loss_patch_text_alignment
 
    else:
        ret = {"pred_answer": pred_answer, "pred_bbox_all": outputs["pred_boxes"]}
    phase = "train" if pl_module.training else "val"

    # logging
    loss_name = "gqa_mdetr"
    ans_acc = getattr(pl_module, f"{phase}_{loss_name}_accuracy_answer_total")(answer_loss_dict["accuracy_answer_total"]*answer["answer_type"].numel(), answer["answer_type"].numel())


    if not pl_module.hparams.config["test_only"]:
        pl_module.log(f"{loss_name}/{phase}/loss", loss)
        pl_module.log(f"{loss_name}/{phase}/loss_ce", losses_dict['loss_ce'])
        pl_module.log(f"{loss_name}/{phase}/loss_bbox", losses_dict['loss_bbox'])
        pl_module.log(f"{loss_name}/{phase}/loss_giou", losses_dict['loss_giou'])
        if pl_module.hparams.config["mdetr_use_alignment"]:
            pl_module.log(f"{loss_name}/{phase}/loss_contrastive_align", losses_dict['loss_contrastive_align'])
        if pl_module.hparams.config["patch_level_alignment_KL"] and pl_module.training:
            pl_module.log(f"{loss_name}/{phase}/patch_text_alignment_loss", tot_loss_patch_text_alignment)
 

    pl_module.log(f"{loss_name}/{phase}/loss_answer", answer_loss)
    pl_module.log(f"{loss_name}/{phase}/ans_acc", ans_acc)
 
    return ret

def compute_copsref(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=False)
    num_classes = pl_module.num_classes
    text_feats = infer["text_feats"] 
    image_feats = infer["image_feats"] 
    cls_feats = infer["cls_feats"]
    bs, queries = infer["det_token_feats"].shape[:2]
    hidden_size = pl_module.hparams.config["hidden_size"]

    # for gqa the answer token is 6
    gqa_ans_det_num = 6
    queries = queries-gqa_ans_det_num 
    obj_det_feat = infer["det_token_feats"][:,:queries,:]
    answer_feat = infer["det_token_feats"][:,queries:,:]
    pred_answer = {}

    # input for the bbox regression
    if pl_module.hparams.config["reg_input"] == "det_token":
        pred_cls = pl_module.classifier(obj_det_feat.contiguous().view(-1,hidden_size)).view(-1, queries, num_classes+1)
        pred_bbox = pl_module.bbox_regressor(obj_det_feat.contiguous().view(-1,hidden_size)).view(-1, queries, 4)
        pred_answer["pred_answer_type"] = pl_module.answer_type_head(answer_feat[:,0,:])
        pred_answer["pred_answer_obj"] = pl_module.answer_obj_head(answer_feat[:,1,:])
        pred_answer["pred_answer_rel"] = pl_module.answer_rel_head(answer_feat[:,2,:])
        pred_answer["pred_answer_attr"] = pl_module.answer_attr_head(answer_feat[:,3,:])
        pred_answer["pred_answer_cat"] = pl_module.answer_cat_head(answer_feat[:,4,:])
        pred_answer["pred_answer_global"] = pl_module.answer_global_head(answer_feat[:,5,:])

    elif pl_module.hparams.config["reg_input"] == "det_token_with_cls":
        pred_cls = pl_module.classifier(obj_det_feat.contiguous().view(-1,hidden_size) + cls_feats.repeat_interleave(queries, dim=0)).view(-1, queries, num_classes+1)
        pred_bbox = pl_module.bbox_regressor(obj_det_feat.contiguous().view(-1,hidden_size) + cls_feats.repeat_interleave(queries, dim=0)).view(-1, queries, 4)
        pred_answer["pred_answer_type"] = pl_module.answer_type_head(answer_feat[:,0,:] + cls_feats)
        pred_answer["pred_answer_obj"] = pl_module.answer_obj_head(answer_feat[:,1,:] + cls_feats)
        pred_answer["pred_answer_rel"] = pl_module.answer_rel_head(answer_feat[:,2,:] + cls_feats)
        pred_answer["pred_answer_attr"] = pl_module.answer_attr_head(answer_feat[:,3,:] + cls_feats)
        pred_answer["pred_answer_cat"] = pl_module.answer_cat_head(answer_feat[:,4,:] + cls_feats)
        pred_answer["pred_answer_global"] = pl_module.answer_global_head(answer_feat[:,5,:] + cls_feats)

    elif pl_module.hparams.config["reg_input"] == "det_token_concat_cls":
        pred_cls = pl_module.classifier(torch.cat((obj_det_feat.contiguous().view(-1,hidden_size) , cls_feats.repeat_interleave(queries, dim=0)), dim=1)).view(-1, queries, num_classes+1)
        pred_bbox = pl_module.bbox_regressor(torch.cat((obj_det_feat.contiguous().view(-1,hidden_size) , cls_feats.repeat_interleave(queries, dim=0)), dim=1 )).view(-1, queries, 4)

        pred_answer["pred_answer_type"] = pl_module.answer_type_head(torch.cat((answer_feat[:,0,:], cls_feats), dim=1))
        pred_answer["pred_answer_obj"] = pl_module.answer_obj_head(torch.cat((answer_feat[:,1,:], cls_feats), dim=1))
        pred_answer["pred_answer_rel"] = pl_module.answer_rel_head(torch.cat((answer_feat[:,2,:], cls_feats), dim=1))
        pred_answer["pred_answer_attr"] = pl_module.answer_attr_head(torch.cat((answer_feat[:,3,:], cls_feats), dim=1))
        pred_answer["pred_answer_cat"] = pl_module.answer_cat_head(torch.cat((answer_feat[:,4,:], cls_feats), dim=1))
        pred_answer["pred_answer_global"] = pl_module.answer_global_head(torch.cat((answer_feat[:,5,:], cls_feats), dim=1))

    # use contrastive alignment from mdetr
    if pl_module.hparams.config["mdetr_use_alignment"]:
        proj_tokens = F.normalize(pl_module.contrastive_align_projection_text(text_feats), p=2, dim=-1 )
        proj_queries = F.normalize(pl_module.contrastive_align_projection_image(infer["det_token_feats"]), p=2, dim=-1)

    gt_bbox = [torch.FloatTensor(batch["gt_bbox"][i]).to(pl_module.device) for i in range(len(batch["gt_bbox"]))]

    positive_map = torch.cat([batch["positive_map"][i] for i in range(len(batch["positive_map"]))], dim=0)

    # patch_level_alignment
    if pl_module.hparams.config["patch_level_alignment"]:
        temperature = 0.07 
        image_feats_no_cls = image_feats[:,1:,:]
        patch_length = image_feats_no_cls.shape[1]
        patch_img_feat = F.normalize(pl_module.patch_contrastive_align_projection_image(image_feats_no_cls), dim=2)

        text_feat = F.normalize(pl_module.patch_contrastive_align_projection_text(text_feats), dim=2)

        patch_text_exp_logits = torch.exp(torch.matmul(text_feat, patch_img_feat.permute(0,2,1))/temperature)

        patch_text_label = torch.zeros(bs, pl_module.hparams.config["max_text_len"], patch_length).to(pl_module.device)

        for b in range(bs):
            for box in range(batch["positive_map"][b].shape[0]):
                bbox_patch_label_idx = (batch["bbox_patch_label"][b].permute(1,0).bool()[box] != 0).nonzero().view(-1)
                pos_map_idx = (batch["positive_map"][b][box, :pl_module.hparams.config["max_text_len"]] != 0).nonzero().view(-1)

                patch_text_label[b, pos_map_idx.repeat_interleave(len(bbox_patch_label_idx)),
                        bbox_patch_label_idx.repeat(len(pos_map_idx))] = 1

        # Compute pos patch and neg patch for a token
        neg_patches_feats = patch_text_exp_logits*(1-patch_text_label)
        pos_patches_feats = patch_text_exp_logits*patch_text_label
        denominator = pos_patches_feats + torch.sum(neg_patches_feats, dim=2).view(bs,-1,1).expand(-1,-1,neg_patches_feats.shape[2])

        # (1-patch_text_label) to aviod nan by torch.log
        token_patch_alignment_loss = -torch.sum(torch.log((pos_patches_feats + (1-patch_text_label))/denominator)*patch_text_label)/torch.sum(patch_text_label)  
      
        # Compute pos token and neg token for a patch
        patch_text_label_T = patch_text_label.permute(0,2,1)
        patch_text_exp_logits_T = patch_text_exp_logits.permute(0,2,1)
        neg_patches_feats_T = patch_text_exp_logits_T*(1-patch_text_label_T)
        pos_patches_feats_T = patch_text_exp_logits_T*patch_text_label_T
        denominator_T = pos_patches_feats_T + torch.sum(neg_patches_feats_T, dim=2).view(bs,-1,1).expand(-1,-1,neg_patches_feats_T.shape[2])

        # (1-patch_text_label) to aviod nan by torch.log
        patch_token_alignment_loss = -torch.sum(torch.log((pos_patches_feats_T + (1-patch_text_label_T))/denominator_T)*patch_text_label_T)/torch.sum(patch_text_label_T)  
        tot_loss_patch_text_alignment = (pl_module.hparams.config["patch_level_alignment_weight1"]*token_patch_alignment_loss + pl_module.hparams.config["patch_level_alignment_weight2"]*patch_token_alignment_loss)/2

    elif pl_module.hparams.config["patch_level_alignment_KL"]:
        temperature = 0.07 
        image_feats_no_cls = image_feats[:,1:,:]
        patch_length = image_feats_no_cls.shape[1]
        patch_img_feat = F.normalize(pl_module.patch_contrastive_align_projection_image(image_feats_no_cls), dim=2)

        text_feat = F.normalize(pl_module.patch_contrastive_align_projection_text(text_feats), dim=2)
        patch_text_logits = torch.matmul(text_feat, patch_img_feat.permute(0,2,1))/temperature

        patch_text_label = torch.zeros(bs, pl_module.hparams.config["max_text_len"], patch_length).to(pl_module.device)
        for b in range(bs):
            for box in range(batch["positive_map"][b].shape[0]):
                bbox_patch_label_idx = (batch["bbox_patch_label"][b].permute(1,0).bool()[box] != 0).nonzero().view(-1)
                pos_map_idx = (batch["positive_map"][b][box, :pl_module.hparams.config["max_text_len"]] != 0).nonzero().view(-1)
                patch_text_label[b, pos_map_idx.repeat_interleave(len(bbox_patch_label_idx)),
                bbox_patch_label_idx.repeat(len(pos_map_idx))] = 1

        # Compute pos patch and neg patch for a token
        pred_log_token_patch_prob = F.log_softmax(patch_text_logits ,dim=2)
        # clamping to avoid nan
        gt_token_patch_prob = patch_text_label/torch.clamp(torch.sum(patch_text_label, dim=2, keepdim=True), min =1).repeat(1,1,patch_length)
        token_patch_alignment_loss = F.kl_div(pred_log_token_patch_prob, gt_token_patch_prob, reduction="sum")/torch.sum(gt_token_patch_prob)

        # Compute pos token and neg token for a patch
        pred_log_patch_token_prob = F.log_softmax(patch_text_logits ,dim=1)
        gt_patch_token_prob = patch_text_label/torch.clamp(torch.sum(patch_text_label, dim=1, keepdim=True), min =1).repeat(1, pl_module.hparams.config["max_text_len"] ,1)
        patch_token_alignment_loss = F.kl_div(pred_log_patch_token_prob, gt_patch_token_prob, reduction="sum")/torch.sum(gt_patch_token_prob)

        tot_loss_patch_text_alignment = pl_module.hparams.config["patch_level_alignment_weight1"]*token_patch_alignment_loss + pl_module.hparams.config["patch_level_alignment_weight2"]*patch_token_alignment_loss
 

    gt_label = [torch.FloatTensor(batch["category_id"][i]).to(pl_module.device) for i in range(len(batch["category_id"]))]
    pred_bbox = pred_bbox.sigmoid() # ensure w, h >= 0
    outputs = {"pred_boxes": pred_bbox,  "pred_logits": pred_cls }
    if pl_module.hparams.config["mdetr_use_alignment"]:
        outputs["proj_queries"] = proj_queries
        outputs["proj_tokens"] = proj_tokens
        tokenized = pl_module.tokenizer.batch_encode_plus(batch["text"], padding="longest", return_tensors="pt").to(pl_module.device)
        outputs["tokenized"] = tokenized
    

    tokens_positive = batch["tokens_positive"]
    targets = [{"boxes":gt_bbox[i], "tokens_positive":tokens_positive[i], "labels":gt_label[i]} for i in range(len(gt_bbox))]
    num_boxes = sum(len(t["boxes"]) for t in targets)

    # compute losses
    losses_dict = pl_module.criterion(outputs, targets, positive_map)

    loss = sum(losses_dict[k] * pl_module.matcher.weight_dict[k] for k in losses_dict.keys() if k in pl_module.matcher.weight_dict)
    if pl_module.hparams.config["patch_level_alignment"] or pl_module.hparams.config["patch_level_alignment_KL"]:
        loss += tot_loss_patch_text_alignment
 
    ret = {
                "loss": loss,
                "loss_bbox":losses_dict['loss_bbox'],
                "loss_giou":losses_dict['loss_giou'],
                "loss_class":losses_dict['loss_ce'],
                "pred_bbox": pred_bbox,
                "pred_bbox_all": outputs["pred_boxes"],
                "pred_logits": outputs["pred_logits"],
                "gt_bbox": gt_bbox,
                "gt_label": gt_label,
            }

    if pl_module.hparams.config["patch_level_alignment"] or pl_module.hparams.config["patch_level_alignment_KL"]:
        ret["patch_text_alignment_loss"] = tot_loss_patch_text_alignment

    # logging
    phase = "train" if pl_module.training else "val"
    loss_name = "copsref"
    if phase != "train":
        det_acc = getattr(pl_module, f"{phase}_{loss_name}_detacc")(outputs["pred_boxes"], ret["gt_bbox"], batch["width"], batch["height"] , _pred_logits=outputs["pred_logits"])
        pl_module.log(f"{loss_name}/{phase}/detacc", det_acc)

    pl_module.log(f"{loss_name}/{phase}/loss", loss)
    pl_module.log(f"{loss_name}/{phase}/loss_ce", losses_dict['loss_ce'])
    pl_module.log(f"{loss_name}/{phase}/loss_bbox", losses_dict['loss_bbox'])
    pl_module.log(f"{loss_name}/{phase}/loss_giou", losses_dict['loss_giou'])
    if pl_module.hparams.config["mdetr_use_alignment"]:
        pl_module.log(f"{loss_name}/{phase}/loss_contrastive_align", losses_dict['loss_contrastive_align'])
    if pl_module.hparams.config["patch_level_alignment"] or pl_module.hparams.config["patch_level_alignment_KL"]:
        pl_module.log(f"{loss_name}/{phase}/patch_text_alignment_loss", tot_loss_patch_text_alignment)
 
    return ret




def compute_clever_mdetr(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=False)
    num_classes = pl_module.num_classes
    text_feats = infer["text_feats"] 
    image_feats = infer["image_feats"] 
    cls_feats = infer["cls_feats"]
    bs, queries = infer["det_token_feats"].shape[:2]
    hidden_size = pl_module.hparams.config["hidden_size"]

    # for clever the answer token is 4
    clever_ans_det_num = 4
    queries = queries-clever_ans_det_num 
    obj_det_feat = infer["det_token_feats"][:,:queries,:]
    answer_feat = infer["det_token_feats"][:,queries:,:]
    pred_answer = {}

    # input for the bbox regression
    if pl_module.hparams.config["reg_input"] == "det_token":
        pred_cls = pl_module.classifier(obj_det_feat.contiguous().view(-1,hidden_size)).view(-1, queries, num_classes+1)
        pred_bbox = pl_module.bbox_regressor(obj_det_feat.contiguous().view(-1,hidden_size)).view(-1, queries, 4)
        pred_answer["pred_answer_type"] = pl_module.answer_type_head(answer_feat[:,0,:])
        pred_answer["pred_answer_binary"] = pl_module.answer_binary_head(answer_feat[:,1,:]).view(bs)
        pred_answer["pred_answer_reg"] = pl_module.answer_reg_head(answer_feat[:,2,:])
        pred_answer["pred_answer_attr"] = pl_module.answer_attr_head(answer_feat[:,3,:])

    elif pl_module.hparams.config["reg_input"] == "det_token_with_cls":
        pred_cls = pl_module.classifier(obj_det_feat.contiguous().view(-1,hidden_size) + cls_feats.repeat_interleave(queries, dim=0)).view(-1, queries, num_classes+1)
        pred_bbox = pl_module.bbox_regressor(obj_det_feat.contiguous().view(-1,hidden_size) + cls_feats.repeat_interleave(queries, dim=0)).view(-1, queries, 4)
        pred_answer["pred_answer_type"] = pl_module.answer_type_head(answer_feat[:,0,:] + cls_feats)
        pred_answer["pred_answer_binary"] = pl_module.answer_binary_head(answer_feat[:,1,:] + cls_feats).view(bs)
        pred_answer["pred_answer_reg"] = pl_module.answer_reg_head(answer_feat[:,2,:] + cls_feats)
        pred_answer["pred_answer_attr"] = pl_module.answer_attr_head(answer_feat[:,3,:] + cls_feats)

    elif pl_module.hparams.config["reg_input"] == "det_token_concat_cls":
        pred_cls = pl_module.classifier(torch.cat((obj_det_feat.contiguous().view(-1,hidden_size) , cls_feats.repeat_interleave(queries, dim=0)), dim=1)).view(-1, queries, num_classes+1)
        pred_bbox = pl_module.bbox_regressor(torch.cat((obj_det_feat.contiguous().view(-1,hidden_size) , cls_feats.repeat_interleave(queries, dim=0)), dim=1 )).view(-1, queries, 4)

        pred_answer["pred_answer_type"] = pl_module.answer_type_head(torch.cat((answer_feat[:,0,:], cls_feats), dim=1))
        pred_answer["pred_answer_binary"] = pl_module.answer_binary_head(torch.cat((answer_feat[:,1,:], cls_feats), dim=1)).view(bs)
        pred_answer["pred_answer_reg"] = pl_module.answer_reg_head(torch.cat((answer_feat[:,2,:], cls_feats), dim=1))
        pred_answer["pred_answer_attr"] = pl_module.answer_attr_head(torch.cat((answer_feat[:,3,:], cls_feats), dim=1))

    # use contrastive alignment from mdetr
    if pl_module.hparams.config["mdetr_use_alignment"]:
        proj_tokens = F.normalize(pl_module.contrastive_align_projection_text(text_feats), p=2, dim=-1 )
        proj_queries = F.normalize(pl_module.contrastive_align_projection_image(infer["det_token_feats"]), p=2, dim=-1)

    # Patch level alignment
    if pl_module.hparams.config["patch_level_alignment_KL"] and pl_module.training:
        temperature = 0.07 
        image_feats_no_cls = image_feats[:,1:,:]
        patch_length = image_feats_no_cls.shape[1]
        patch_img_feat = F.normalize(pl_module.patch_contrastive_align_projection_image(image_feats_no_cls), dim=2)

        text_feat = F.normalize(pl_module.patch_contrastive_align_projection_text(text_feats), dim=2)
        patch_text_logits = torch.matmul(text_feat, patch_img_feat.permute(0,2,1))/temperature

        patch_text_label = torch.zeros(bs, pl_module.hparams.config["max_text_len"], patch_length).to(pl_module.device)
        for b in range(bs):
            for box in range(batch["positive_map"][b].shape[0]):
                bbox_patch_label_idx = (batch["bbox_patch_label"][b].permute(1,0).bool()[box] != 0).nonzero().view(-1)
                pos_map_idx = (batch["bbox_patch_label"][b][box, :pl_module.hparams.config["max_text_len"]] != 0).nonzero().view(-1)
                patch_text_label[b, pos_map_idx.repeat_interleave(len(bbox_patch_label_idx)),
                                    bbox_patch_label_idx.repeat(len(pos_map_idx))] = 1

        # Compute pos patch and neg patch for a token
        pred_log_token_patch_prob = F.log_softmax(patch_text_logits ,dim=2)
        # clamping to avoid nan
        gt_token_patch_prob = patch_text_label/torch.clamp(torch.sum(patch_text_label, dim=2, keepdim=True), min =1).repeat(1,1,patch_length)
        token_patch_alignment_loss = F.kl_div(pred_log_token_patch_prob, gt_token_patch_prob, reduction="sum")/torch.sum(gt_token_patch_prob)

        # Compute pos token and neg token for a patch
        pred_log_patch_token_prob = F.log_softmax(patch_text_logits ,dim=1)
        gt_patch_token_prob = patch_text_label/torch.clamp(torch.sum(patch_text_label, dim=1, keepdim=True), min =1).repeat(1, pl_module.hparams.config["max_text_len"] ,1)
        patch_token_alignment_loss = F.kl_div(pred_log_patch_token_prob, gt_patch_token_prob, reduction="sum")/torch.sum(gt_patch_token_prob)

        tot_loss_patch_text_alignment = pl_module.hparams.config["patch_level_alignment_weight1"]*token_patch_alignment_loss + pl_module.hparams.config["patch_level_alignment_weight2"]*patch_token_alignment_loss
 

    gt_bbox = [torch.FloatTensor(batch["gt_bbox"][i]).to(pl_module.device) for i in range(len(batch["gt_bbox"]))]

    positive_map = torch.cat([batch["positive_map"][i] for i in range(len(batch["positive_map"]))], dim=0)

    gt_label = [torch.FloatTensor(batch["category_id"][i]).to(pl_module.device) for i in range(len(batch["category_id"]))]
    #pred_bbox = pred_bbox.clamp(0.0) # ensure w, h >= 0
    pred_bbox = pred_bbox.sigmoid() # ensure w, h >= 0
    outputs = {"pred_boxes": pred_bbox,  "pred_logits": pred_cls }
    if pl_module.hparams.config["mdetr_use_alignment"]:
        outputs["proj_queries"] = proj_queries
        outputs["proj_tokens"] = proj_tokens
        tokenized = pl_module.tokenizer.batch_encode_plus(batch["text"], padding="longest", return_tensors="pt").to(pl_module.device)
        outputs["tokenized"] = tokenized
    
    tokens_positive = batch["tokens_positive"]
    targets = [{"boxes":gt_bbox[i], "tokens_positive":tokens_positive[i], "labels":gt_label[i]} for i in range(len(gt_bbox))]
    num_boxes = sum(len(t["boxes"]) for t in targets)
 
    # compute losses
    losses_dict = {}
    if not pl_module.hparams.config["test_only"]:
        losses_dict = pl_module.criterion(outputs, targets, positive_map)
    answer = {}
    for i in ["answer_type", "answer_binary", "answer_reg", "answer_attr"]:
        if i == "answer_binary":
            answer[i] = torch.FloatTensor(batch[i]).to(pl_module.device)
        else:
            answer[i] = torch.LongTensor(batch[i]).to(pl_module.device)
    answer_loss_dict = pl_module.answer_criterion(pred_answer, answer)
    answer_loss = sum(answer_loss_dict[k] * pl_module.matcher.weight_dict[k] for k in answer_loss_dict.keys() if k in pl_module.matcher.weight_dict)
    losses_dict.update(answer_loss_dict)
 
    loss = sum(losses_dict[k] * pl_module.matcher.weight_dict[k] for k in losses_dict.keys() if k in pl_module.matcher.weight_dict)
    if pl_module.hparams.config["patch_level_alignment_KL"] and pl_module.training:
        loss += tot_loss_patch_text_alignment
 
    if not pl_module.hparams.config["test_only"]:
        ret = {
                "loss": loss,
                "loss_bbox":losses_dict['loss_bbox'],
                "loss_giou":losses_dict['loss_giou'],
                "loss_class":losses_dict['loss_ce'],
                "pred_bbox": pred_bbox,
                "pred_bbox_all": outputs["pred_boxes"],
                "pred_logits": outputs["pred_logits"],
                "gt_bbox": gt_bbox,
                "gt_label": gt_label,
                "pred_answer": pred_answer
            }
        if pl_module.hparams.config["patch_level_alignment_KL"] and pl_module.training:
            ret["patch_text_alignment_loss"] = tot_loss_patch_text_alignment
 
    else:
        ret = {"pred_answer": pred_answer, "pred_bbox_all": outputs["pred_boxes"]}
    phase = "train" if pl_module.training else "val"

    ans_acc = getattr(pl_module, f"{phase}_clever_mdetr_accuracy_answer_total")(answer_loss_dict["accuracy_answer_total"]*answer["answer_type"].numel(), answer["answer_type"].numel())


    if not pl_module.hparams.config["test_only"]:
        pl_module.log(f"clever_mdetr/{phase}/loss", loss)
        pl_module.log(f"clever_mdetr/{phase}/loss_ce", losses_dict['loss_ce'])
        pl_module.log(f"clever_mdetr/{phase}/loss_bbox", losses_dict['loss_bbox'])
        pl_module.log(f"clever_mdetr/{phase}/loss_giou", losses_dict['loss_giou'])
        if pl_module.hparams.config["mdetr_use_alignment"]:
            pl_module.log(f"clever_mdetr/{phase}/loss_contrastive_align", losses_dict['loss_contrastive_align'])
        if pl_module.hparams.config["patch_level_alignment_KL"] and pl_module.training:
            pl_module.log(f"clever_mdetr/{phase}/patch_text_alignment_loss", tot_loss_patch_text_alignment)
 

    pl_module.log(f"clever_mdetr/{phase}/loss_answer", answer_loss)
    pl_module.log(f"clever_mdetr/{phase}/ans_acc", ans_acc)
 
    return ret




def compute_flickr_mdetr(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=False)
    num_classes = pl_module.num_classes
    text_feats = infer["text_feats"] 
    image_feats = infer["image_feats"] 
    cls_feats = infer["cls_feats"]
    bs, queries = infer["det_token_feats"].shape[:2]
    hidden_size = pl_module.hparams.config["hidden_size"]
    
    # input for the bbox regression
    if pl_module.hparams.config["reg_input"] == "det_token":
        pred_cls = pl_module.classifier(infer["det_token_feats"].contiguous().view(-1,hidden_size)).view(-1, queries, num_classes+1)
        pred_bbox = pl_module.bbox_regressor(infer["det_token_feats"].contiguous().view(-1,hidden_size)).view(-1, queries, 4)
    elif pl_module.hparams.config["reg_input"] == "det_token_with_cls":
            pred_cls = pl_module.classifier(infer["det_token_feats"].contiguous().view(-1,hidden_size) + cls_feats.repeat_interleave(queries, dim=0)).view(-1, queries, num_classes+1)
            pred_bbox = pl_module.bbox_regressor(infer["det_token_feats"].contiguous().view(-1,hidden_size) + cls_feats.repeat_interleave(queries, dim=0)).view(-1, queries, 4)
    elif pl_module.hparams.config["reg_input"] == "det_token_concat_cls":
            pred_cls = pl_module.classifier(torch.cat((infer["det_token_feats"].contiguous().view(-1,hidden_size) , cls_feats.repeat_interleave(queries, dim=0)), dim=1)).view(-1, queries, num_classes+1)
            pred_bbox = pl_module.bbox_regressor(torch.cat((infer["det_token_feats"].contiguous().view(-1,hidden_size) , cls_feats.repeat_interleave(queries, dim=0)), dim=1 )).view(-1, queries, 4)
      
    # use contrastive alignment from mdetr
    if pl_module.hparams.config["mdetr_use_alignment"]:
        proj_tokens = F.normalize(pl_module.contrastive_align_projection_text(text_feats), p=2, dim=-1 )
        proj_queries = F.normalize(pl_module.contrastive_align_projection_image(infer["det_token_feats"]), p=2, dim=-1)

    gt_bbox = [torch.FloatTensor(batch["gt_bbox"][i]).to(pl_module.device) for i in range(len(batch["gt_bbox"]))]

    positive_map = torch.cat([batch["positive_map"][i] for i in range(len(batch["positive_map"]))], dim=0)

    gt_label = [torch.FloatTensor(batch["category_id"][i]).to(pl_module.device) for i in range(len(batch["category_id"]))]
    pred_bbox = pred_bbox.sigmoid() # ensure w, h >= 0
    outputs = {"pred_boxes": pred_bbox,  "pred_logits": pred_cls }
    if pl_module.hparams.config["mdetr_use_alignment"]:
        outputs["proj_queries"] = proj_queries
        outputs["proj_tokens"] = proj_tokens
        tokenized = pl_module.tokenizer.batch_encode_plus(batch["text"], padding="longest", return_tensors="pt").to(pl_module.device)
        outputs["tokenized"] = tokenized
    

    tokens_positive = batch["tokens_positive"]
    targets = [{"boxes":gt_bbox[i], "tokens_positive":tokens_positive[i], "labels":gt_label[i]} for i in range(len(gt_bbox))]
    num_boxes = sum(len(t["boxes"]) for t in targets)

    # compute losses
    losses_dict = pl_module.criterion(outputs, targets, positive_map)

    loss = sum(losses_dict[k] * pl_module.matcher.weight_dict[k] for k in losses_dict.keys() if k in pl_module.matcher.weight_dict)
    ret = {
                "loss": loss,
                "loss_bbox":losses_dict['loss_bbox'],
                "loss_giou":losses_dict['loss_giou'],
                "loss_class":losses_dict['loss_ce'],
                "pred_bbox": pred_bbox,
                "pred_bbox_all": outputs["pred_boxes"],
                "pred_logits": outputs["pred_logits"],
                "gt_bbox": gt_bbox,
                "gt_label": gt_label,
                "outputs": outputs
            }
 
    # logging
    phase = "train" if pl_module.training else "val"

    pl_module.log(f"flickr_mdetr/{phase}/loss", loss)
    pl_module.log(f"flickr_mdetr/{phase}/loss_ce", losses_dict['loss_ce'])
    pl_module.log(f"flickr_mdetr/{phase}/loss_bbox", losses_dict['loss_bbox'])
    pl_module.log(f"flickr_mdetr/{phase}/loss_giou", losses_dict['loss_giou'])
    if pl_module.hparams.config["mdetr_use_alignment"]:
        pl_module.log(f"flickr_mdetr/{phase}/loss_contrastive_align", losses_dict['loss_contrastive_align'])

  
    return ret


def compute_snlive(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=False)
    num_classes = pl_module.num_classes
    text_feats = infer["text_feats"] 
    image_feats = infer["image_feats"] 
    cls_feats = infer["cls_feats"]
    bs, queries = infer["det_token_feats"].shape[:2]
    hidden_size = pl_module.hparams.config["hidden_size"]

    # for snlive the answer token is 1
    # Softmax output for [entailment, contradict, neutral]
    snlive_ans_det_num = 1
    queries = queries-snlive_ans_det_num 
    obj_det_feat = infer["det_token_feats"][:,:queries,:]
    answer_feat = infer["det_token_feats"][:,queries:,:]
    pred_answer = {}

    # input for the bbox regression
    if pl_module.hparams.config["reg_input"] == "det_token":
        pred_cls = pl_module.classifier(obj_det_feat.contiguous().view(-1,hidden_size)).view(-1, queries, num_classes+1)
        pred_bbox = pl_module.bbox_regressor(obj_det_feat.contiguous().view(-1,hidden_size)).view(-1, queries, 4)
        pred_answer = pl_module.answer_head(answer_feat[:,0,:])

    elif pl_module.hparams.config["reg_input"] == "det_token_with_cls":
        pred_cls = pl_module.classifier(obj_det_feat.contiguous().view(-1,hidden_size) + cls_feats.repeat_interleave(queries, dim=0)).view(-1, queries, num_classes+1)
        pred_bbox = pl_module.bbox_regressor(obj_det_feat.contiguous().view(-1,hidden_size) + cls_feats.repeat_interleave(queries, dim=0)).view(-1, queries, 4)
        pred_answer = pl_module.answer_head(answer_feat[:,0,:] + cls_feats)

    elif pl_module.hparams.config["reg_input"] == "det_token_concat_cls":
        pred_cls = pl_module.classifier(torch.cat((obj_det_feat.contiguous().view(-1,hidden_size) , cls_feats.repeat_interleave(queries, dim=0)), dim=1)).view(-1, queries, num_classes+1)
        pred_bbox = pl_module.bbox_regressor(torch.cat((obj_det_feat.contiguous().view(-1,hidden_size) , cls_feats.repeat_interleave(queries, dim=0)), dim=1 )).view(-1, queries, 4)

        pred_answer = pl_module.answer_head(torch.cat((answer_feat[:,0,:], cls_feats), dim=1))


    if pl_module.hparams.config["mdetr_use_alignment"]:
        proj_tokens = F.normalize(pl_module.contrastive_align_projection_text(text_feats), p=2, dim=-1 )
        proj_queries = F.normalize(pl_module.contrastive_align_projection_image(infer["det_token_feats"]), p=2, dim=-1)

    # Patch level alignment
    if pl_module.hparams.config["patch_level_alignment_KL"] and pl_module.training:
        temperature = 0.07 
        image_feats_no_cls = image_feats[:,1:,:]
        patch_length = image_feats_no_cls.shape[1]
        patch_img_feat = F.normalize(pl_module.patch_contrastive_align_projection_image(image_feats_no_cls), dim=2)

        text_feat = F.normalize(pl_module.patch_contrastive_align_projection_text(text_feats), dim=2)
        patch_text_logits = torch.matmul(text_feat, patch_img_feat.permute(0,2,1))/temperature

        patch_text_label = torch.zeros(bs, pl_module.hparams.config["max_text_len"], patch_length).to(pl_module.device)
        for b in range(bs):
            for box in range(batch["positive_map"][b].shape[0]):
                bbox_patch_label_idx = (batch["bbox_patch_label"][b].permute(1,0).bool()[box] != 0).nonzero().view(-1)
                pos_map_idx = (batch["bbox_patch_label"][b][box, :pl_module.hparams.config["max_text_len"]] != 0).nonzero().view(-1)
                patch_text_label[b, pos_map_idx.repeat_interleave(len(bbox_patch_label_idx)),
                                    bbox_patch_label_idx.repeat(len(pos_map_idx))] = 1

        # Compute pos patch and neg patch for a token
        pred_log_token_patch_prob = F.log_softmax(patch_text_logits ,dim=2)
        # clamping to avoid nan
        gt_token_patch_prob = patch_text_label/torch.clamp(torch.sum(patch_text_label, dim=2, keepdim=True), min =1).repeat(1,1,patch_length)
        token_patch_alignment_loss = F.kl_div(pred_log_token_patch_prob, gt_token_patch_prob, reduction="sum")/torch.sum(gt_token_patch_prob)

        # Compute pos token and neg token for a patch
        pred_log_patch_token_prob = F.log_softmax(patch_text_logits ,dim=1)
        gt_patch_token_prob = patch_text_label/torch.clamp(torch.sum(patch_text_label, dim=1, keepdim=True), min =1).repeat(1, pl_module.hparams.config["max_text_len"] ,1)
        patch_token_alignment_loss = F.kl_div(pred_log_patch_token_prob, gt_patch_token_prob, reduction="sum")/torch.sum(gt_patch_token_prob)

        tot_loss_patch_text_alignment = pl_module.hparams.config["patch_level_alignment_weight1"]*token_patch_alignment_loss + pl_module.hparams.config["patch_level_alignment_weight2"]*patch_token_alignment_loss

    has_bbox = torch.Tensor(batch["has_bbox"]).to(pl_module.device)
    has_bbox_idx = (has_bbox == 1).nonzero().view(-1)
    loss = 0
    pred_bbox = pred_bbox.sigmoid() # ensure w, h >= 0
    if len(has_bbox_idx) > 0:
        gt_bbox = [torch.FloatTensor(batch["gt_bbox"][i]).to(pl_module.device) for i in range(len(batch["gt_bbox"])) if batch["has_bbox"][i] == 1]
        assert len(gt_bbox) == len(has_bbox_idx)

        positive_map = torch.cat([batch["positive_map"][i] for i in range(len(batch["positive_map"])) if batch["has_bbox"][i] == 1 ], dim=0)

        gt_label = [torch.FloatTensor(batch["category_id"][i]).to(pl_module.device) for i in range(len(batch["category_id"])) if batch["has_bbox"][i] == 1]


        assert len(gt_label) == len(gt_bbox) == len(has_bbox_idx)  # ensure all snlive are removed, only flicker mdetr annotation are left
        pred_bbox = pred_bbox[has_bbox_idx]
        pred_cls = pred_cls[has_bbox_idx]
        outputs = {"pred_boxes": pred_bbox,  "pred_logits": pred_cls }
        if pl_module.hparams.config["mdetr_use_alignment"]:
            outputs["proj_queries"] = proj_queries[has_bbox_idx]
            outputs["proj_tokens"] = proj_tokens[has_bbox_idx]
            batch_text = [batch["text"][i] for i in range(len(batch["text"])) if batch["has_bbox"][i] == 1]
            tokenized = pl_module.tokenizer.batch_encode_plus(batch_text, padding="longest", return_tensors="pt").to(pl_module.device)
            outputs["tokenized"] = tokenized
    
        tokens_positive = [batch["tokens_positive"][i] for i in range(len(batch["tokens_positive"])) if batch["has_bbox"][i] == 1]

        targets = [{"boxes":gt_bbox[i], "tokens_positive":tokens_positive[i], "labels":gt_label[i]} for i in range(len(gt_bbox))]
        num_boxes = sum(len(t["boxes"]) for t in targets)
 
        losses_dict = {}
        if not pl_module.hparams.config["test_only"]:
            losses_dict = pl_module.criterion(outputs, targets, positive_map)
 
        loss = sum(losses_dict[k] * pl_module.matcher.weight_dict[k] for k in losses_dict.keys() if k in pl_module.matcher.weight_dict)
        if pl_module.hparams.config["patch_level_alignment_KL"] and pl_module.training:
            loss += tot_loss_patch_text_alignment
 
    
    answer = torch.LongTensor(batch["answer"]).to(pl_module.device)
    loss_answer = pl_module.answer_criterion(pred_answer, answer)
    loss += loss_answer
    ret = {
                "loss": loss,
                "answer_loss": loss_answer,
                "pred_bbox_all": pred_bbox,
                "pred_answer": pred_answer
    }
    phase = "train" if pl_module.training else "val"

    ans_acc = getattr(pl_module, f"{phase}_snlive_accuracy")(pred_answer, answer)

    pl_module.log(f"snlive/{phase}/loss", loss)
    pl_module.log(f"snlive/{phase}/loss_answer", loss_answer)
    pl_module.log(f"snlive/{phase}/ans_acc", ans_acc)
 
    return ret



# compute the recall for flickr phrase grounding
# Following https://github.com/ashkamath/mdetr/blob/main/datasets/flickr_eval.py
# [NEED FIX], the recall value is low, might be some bugs
@torch.no_grad()
def compute_flickr_phrase_grounding_recall(pl_module):
    
    subset = "test" if pl_module.hparams.config["test_only"] else "val"
    evaluator = FlickrEvaluator("./datasets/raw/F30K/flickr30k_entities", subset=subset, merge_boxes=pl_module.hparams.config["flickr_mdetr_GT_type"]=="merged")
   
    if pl_module.hparams.config["test_only"]:
        dset = pl_module.trainer.datamodule.dms[0].test_dataset #make_no_false_val_dset()
    else:
        dset = pl_module.trainer.datamodule.dms[0].val_dataset #make_no_false_val_dset()
    #dist_sampler = DistributedSampler(dset, shuffle=False)
    dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=pl_module.hparams.config["per_gpu_batchsize"],
        num_workers=pl_module.hparams.config["num_workers"],
        #sampler=dist_sampler,
        pin_memory=True,
        collate_fn=functools.partial(
            dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )

    flickr_res = [] if "flickr_bbox" in pl_module.postprocessors.keys() else None

    for _, batch in tqdm.tqdm(enumerate(iter(loader)), desc="Computing Flickr Recall"):
        bs = len(batch["positive_map"])
        batch["text_ids"]  = batch["text_ids"].to(pl_module.device)
        batch["image"] = [b.to(pl_module.device) for b in batch["image"]]
        batch['text_labels'] = batch['text_labels'].cuda()
        batch['text_ids_mlm'] = batch['text_ids_mlm'].cuda()
        batch['text_labels_mlm'] = batch['text_labels_mlm'].cuda()
        batch['text_masks'] = batch['text_masks'].cuda()
        batch["positive_map"] = [b.to(pl_module.device) for b in batch["positive_map"]]
        ret = pl_module(batch) 
        outputs = ret["outputs"]
        orig_target_sizes = torch.stack([torch.FloatTensor([batch["height"][i],batch["width"][i]])  for i in range(bs)], dim=0).to(pl_module.device)
        results = pl_module.postprocessors["bbox"](outputs, orig_target_sizes)
        flickr_res = [] if "flickr_bbox" in pl_module.postprocessors.keys() else None
        
        original_img_ids = [batch["original_img_id"][i]  for i in range(bs)]
        sentence_ids = [batch["sentence_id"][i] for i in range(bs)]
        items_per_batch_element = [batch["nb_eval"][i] for i in range(bs)]
        positive_map_eval = torch.cat([batch["positive_map_eval"][i] for i in range(len(batch["positive_map_eval"]))], dim=0).to(pl_module.device)
        flickr_results = pl_module.postprocessors["flickr_bbox"](
                    outputs, orig_target_sizes, positive_map_eval, items_per_batch_element
                )
        assert len(flickr_results) == len(original_img_ids) == len(sentence_ids)
        for im_id, sent_id, output in zip(original_img_ids, sentence_ids, flickr_results):
            flickr_res.append({"image_id": im_id, "sentence_id": sent_id, "boxes": output})
        #pl_module.evaluator.update(flickr_res)
        evaluator.update(flickr_res)

    #torch.distributed.barrier()
    
    #pl_module.evaluator.synchronize_between_processes()
    #flickr_res = pl_module.evaluator.summarize()
    flickr_res = evaluator.summarize()

    return flickr_res



@torch.no_grad()
def compute_irtr_recall(pl_module):
    text_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset()
    text_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    text_loader = torch.utils.data.DataLoader(
        text_dset,
        batch_size=64,
        num_workers=pl_module.hparams.config["num_workers"],
        pin_memory=True,
        collate_fn=functools.partial(
            text_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )

    image_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset(
        image_only=True
    )
    image_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    dist_sampler = DistributedSampler(image_dset, shuffle=False)
    image_loader = torch.utils.data.DataLoader(
        image_dset,
        batch_size=1,
        num_workers=pl_module.hparams.config["num_workers"],
        sampler=dist_sampler,
        pin_memory=True,
        collate_fn=functools.partial(
            image_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )

    text_preload = list()
    for _b in tqdm.tqdm(text_loader, desc="text prefetch loop"):
        text_preload.append(
            {
                "text_ids": _b["text_ids"].to(pl_module.device),
                "text_masks": _b["text_masks"].to(pl_module.device),
                "text_labels": _b["text_labels"].to(pl_module.device),
                "img_index": _b["img_index"],
            }
        )

    tiids = list()
    for pre in text_preload:
        tiids += pre["img_index"]
    tiids = torch.tensor(tiids)

    image_preload = list()
    for _b in tqdm.tqdm(image_loader, desc="image prefetch loop"):
        (ie, im, _, _, _) = pl_module.transformer.visual_embed(
            _b["image"][0].to(pl_module.device),
            max_image_len=pl_module.hparams.config["max_image_len"],
            mask_it=False, bbox_patch_label = None
        )
        image_preload.append((ie, im, _b["img_index"][0]))

    rank_scores = list()
    rank_iids = list()

    for img_batch in tqdm.tqdm(image_preload, desc="rank loop"):
        _ie, _im, _iid = img_batch
        _, l, c = _ie.shape

        img_batch_score = list()
        for txt_batch in text_preload:
            fblen = len(txt_batch["text_ids"])
            ie = _ie.expand(fblen, l, c)
            im = _im.expand(fblen, l)

            with torch.cuda.amp.autocast():
                score = pl_module.rank_output(
                    pl_module.infer(
                        {
                            "text_ids": txt_batch["text_ids"],
                            "text_masks": txt_batch["text_masks"],
                            "text_labels": txt_batch["text_labels"],
                        },
                        image_embeds=ie,
                        image_masks=im,
                    )["cls_feats"]
                )[:, 0]

            img_batch_score.append(score)

        img_batch_score = torch.cat(img_batch_score)
        rank_scores.append(img_batch_score.cpu().tolist())
        rank_iids.append(_iid)

    torch.distributed.barrier()
    gather_rank_scores = all_gather(rank_scores)
    gather_rank_iids = all_gather(rank_iids)

    iids = torch.tensor(gather_rank_iids)
    iids = iids.view(-1)
    scores = torch.tensor(gather_rank_scores)
    scores = scores.view(len(iids), -1)

    topk10 = scores.topk(10, dim=1)
    topk5 = scores.topk(5, dim=1)
    topk1 = scores.topk(1, dim=1)
    topk10_iids = tiids[topk10.indices]
    topk5_iids = tiids[topk5.indices]
    topk1_iids = tiids[topk1.indices]

    tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
    tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
    tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()

    topk10 = scores.topk(10, dim=0)
    topk5 = scores.topk(5, dim=0)
    topk1 = scores.topk(1, dim=0)
    topk10_iids = iids[topk10.indices]
    topk5_iids = iids[topk5.indices]
    topk1_iids = iids[topk1.indices]

    ir_r10 = (tiids.unsqueeze(0) == topk10_iids).float().max(dim=0)[0].mean()
    ir_r5 = (tiids.unsqueeze(0) == topk5_iids).float().max(dim=0)[0].mean()
    ir_r1 = (tiids.unsqueeze(0) == topk1_iids).float().max(dim=0)[0].mean()

    return (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10)


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


def vqa_test_step(pl_module, batch, output):
    id2answer = (
        pl_module.trainer.datamodule.dm_dicts["vqa_trainval"].id2answer
        if "vqa_trainval" in pl_module.trainer.datamodule.dm_dicts
        else pl_module.trainer.datamodule.dm_dicts["vqa"].id2answer
    )
    vqa_logits = output["vqa_logits"]
    vqa_preds = vqa_logits.argmax(dim=-1)
    vqa_preds = [id2answer[pred.item()] for pred in vqa_preds]
    questions = batch["text"]
    qids = batch["qid"]
    return {"qids": qids, "preds": vqa_preds}


def arc_test_step(pl_module, batch, output):
    return output


def vqa_test_wrapup(outs, model_name):
    rank = torch.distributed.get_rank()
    qids, preds = list(), list()
    for out in outs:
        qids += out["qids"]
        preds += out["preds"]

    rets = list()
    for qid, pred in zip(qids, preds):
        rets.append({"question_id": qid, "answer": pred})
    with open(f"vqa_submit_{rank}.json", "w") as fp:
        json.dump(rets, fp, indent=4)

    torch.distributed.barrier()

    if rank == 0:
        jsons = list()
        paths = list(glob.glob("vqa_submit_*.json"))
        for path in paths:
            with open(path, "r") as fp:
                jsons += json.load(fp)
        os.makedirs("result", exist_ok=True)
        with open(f"result/vqa_submit_{model_name}.json", "w") as fp:
            json.dump(jsons, fp, indent=4)

    torch.distributed.barrier()
    os.remove(f"vqa_submit_{rank}.json")


def arc_test_wrapup(outs, caplen, model_name):
    rank = torch.distributed.get_rank()
    iids, captions = list(), list()
    for out in outs:
        iids += out["iid"]
        captions += out["captions"]

    rets = list()
    for iid, caption in zip(iids, captions):
        rets.append({"image_id": iid, "caption": caption})
    with open(f"coco_cap_len{caplen}_{rank}.json", "w") as fp:
        json.dump(rets, fp, indent=4)

    torch.distributed.barrier()

    if rank == 0:
        jsons = list()
        paths = list(glob.glob(f"coco_cap_len{caplen}_*.json"))
        for path in paths:
            with open(path, "r") as fp:
                jsons += json.load(fp)
        os.makedirs("result/arc", exist_ok=True)
        jsons = sorted(jsons, key=lambda x: x["image_id"])
        with open(f"result/arc/coco_cap_{model_name}_len{caplen}.json", "w") as fp:
            json.dump(jsons, fp, indent=4)

    torch.distributed.barrier()
    os.remove(f"coco_cap_len{caplen}_{rank}.json")
