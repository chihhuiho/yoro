import torch
import torch.nn as nn
import pytorch_lightning as pl
from vilt.modules import vision_transformer as vit

from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from transformers.models.roberta.modeling_roberta import RobertaConfig, RobertaEmbeddings
from vilt.modules import heads, objectives, vilt_utils
from vilt.modules.refcoco_matcher import HungarianMatcherRefCoco
from vilt.modules.mdetr_matcher import HungarianMatcher as MdetrMatcher
from vilt.modules.mdetr_matcher import SetCriterion
import torch.nn.functional as F
from vilt.datamodules.datamodule_base import get_pretrained_tokenizer, get_pretrained_roberta_tokenizer 
import copy
from vilt.gadgets.postprocessors import build_postprocessors
from torchvision.ops.boxes import box_area
from vilt.transforms.coco import box_cxcywh_to_xyxy
from vilt.gadgets.gqa_eval import QACriterionGQA
from vilt.gadgets.clever_eval import QACriterionClever
import time

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x



class ViLTransformerSS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        self.config = config
        if "roberta" in config["tokenizer"]:
            roberta_config = RobertaConfig(
                vocab_size=config["vocab_size"], # 50265
                hidden_size=config["hidden_size"],
                num_hidden_layers=config["num_layers"],
                num_attention_heads=config["num_heads"],
                intermediate_size=config["hidden_size"] * config["mlp_ratio"],
                #max_position_embeddings=config["max_text_len"], # will cause error if this line is not comment out
                hidden_dropout_prob=config["drop_rate"],
                attention_probs_dropout_prob=config["drop_rate"],
            )
            self.text_embeddings = RobertaEmbeddings(roberta_config)
        else:
            bert_config = BertConfig(
                vocab_size=config["vocab_size"],
                hidden_size=config["hidden_size"],
                num_hidden_layers=config["num_layers"],
                num_attention_heads=config["num_heads"],
                intermediate_size=config["hidden_size"] * config["mlp_ratio"],
                max_position_embeddings=config["max_text_len"],
                hidden_dropout_prob=config["drop_rate"],
                attention_probs_dropout_prob=config["drop_rate"],
            )

            self.text_embeddings = BertEmbeddings(bert_config)
 
        self.text_embeddings.apply(objectives.init_weights)

        self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])
        self.token_type_embeddings.apply(objectives.init_weights)

        if self.hparams.config["load_path"] == "":
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=True, config=self.hparams.config
            )
        else:
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=False, config=self.hparams.config
            )

        self.pooler = heads.Pooler(config["hidden_size"])
        self.pooler.apply(objectives.init_weights)

        if self.hparams.config["add_det_token"] == True:
            self.transformer.det_init(det_token_num = self.hparams.config["det_token_num"])

        if config["loss_names"]["mlm"] > 0:
            if "roberta" in config["tokenizer"]:
                self.mlm_score = heads.MLMHead(roberta_config)
            else:
                self.mlm_score = heads.MLMHead(bert_config)
            self.mlm_score.apply(objectives.init_weights)

        if config["loss_names"]["itm"] > 0:
            self.itm_score = heads.ITMHead(config["hidden_size"])
            self.itm_score.apply(objectives.init_weights)

        if config["loss_names"]["mpp"] > 0:
            self.mpp_score = heads.MPPHead(bert_config)
            self.mpp_score.apply(objectives.init_weights)

        # ===================== Downstream ===================== #
        if (
            self.hparams.config["load_path"] != ""
            and not self.hparams.config["test_only"]
            and not self.hparams.config["resume_qa"]
        ):
            print("Loading " + self.hparams.config["load_path"])
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)

        hs = self.hparams.config["hidden_size"]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            vs = self.hparams.config["vqav2_label_size"]
            self.vqa_classifier = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, vs),
            )
            self.vqa_classifier.apply(objectives.init_weights)

        if self.hparams.config["loss_names"]["rec"] > 0 or self.hparams.config["loss_names"]["mdetr_pretrain"] > 0 or self.hparams.config["loss_names"]["mdetr_pretrain_ablade"] > 0 or self.hparams.config["loss_names"]["flickr_mdetr"] or self.hparams.config["loss_names"]["gqa_mdetr"] > 0 or self.hparams.config["loss_names"]["clever_mdetr"] > 0 or self.hparams.config["loss_names"]["snlive"] > 0 or self.hparams.config["loss_names"]["copsref"] > 0:
            self.num_classes = 255

            tokenizer = self.hparams.config["tokenizer"]
            if "roberta" in tokenizer:
                self.tokenizer = get_pretrained_roberta_tokenizer(tokenizer)
            else:
                self.tokenizer = get_pretrained_tokenizer(tokenizer)
 
            self.classifier = nn.Linear(hs, self.num_classes + 1)
            self.classifier.apply(objectives.init_weights)
                
            self.bbox_regressor = MLP(hs, hs, 4, 3)
            self.bbox_regressor.apply(objectives.init_weights)

            self.contrastive_align_projection_image = nn.Linear(hs, 64)
            self.contrastive_align_projection_image.apply(objectives.init_weights)

            self.contrastive_align_projection_text = nn.Linear(hs, 64)
            self.contrastive_align_projection_text.apply(objectives.init_weights)

            if self.hparams.config["patch_level_alignment"] or self.hparams.config["patch_level_alignment_KL"]:
                self.patch_contrastive_align_projection_image = nn.Linear(hs, 64)
                self.patch_contrastive_align_projection_image.apply(objectives.init_weights)

                self.patch_contrastive_align_projection_text = nn.Linear(hs, 64)
                self.patch_contrastive_align_projection_text.apply(objectives.init_weights)


            self.image_pooler = heads.Pooler(config["hidden_size"])
            self.text_pooler = heads.Pooler(config["hidden_size"])
            use_gqa=True if self.hparams.config["loss_names"]["gqa_mdetr"] > 0 else False
            use_clever=True if self.hparams.config["loss_names"]["clever_mdetr"] > 0 else False
            self.matcher = MdetrMatcher(cost_bbox=self.hparams.config["matchingloss_bbox_weight"], cost_giou=self.hparams.config["matchingloss_giou_weight"], cost_class=self.hparams.config["matchingloss_cls_weight"], cost_align=self.hparams.config["contrastive_align_loss_weight"], use_gqa=use_gqa, use_clever=use_clever, qa_loss_coef=self.hparams.config["qa_loss_coef"]) 
 
            if self.hparams.config["mdetr_use_alignment"]:
                loss_lst = ['labels', 'boxes', 'cardinality', 'contrastive_align']
            else:
                loss_lst = ['labels', 'boxes', 'cardinality']
            self.criterion = criterion = SetCriterion(
                   self.num_classes,
                   matcher=self.matcher,
                   eos_coef=0.1,
                   losses=loss_lst,
                   temperature=0.07,
            )
            if self.hparams.config["loss_names"]["flickr_mdetr"] > 0:
                self.postprocessors = build_postprocessors(dataset_name = "flickr")
 
            # initialize the answer tokens for gqa related tasks
            if self.hparams.config["loss_names"]["gqa_mdetr"] > 0 or self.hparams.config["loss_names"]["copsref"] > 0:
                self.answer_type_head = nn.Linear(hs, 5)
                self.answer_rel_head = nn.Linear(hs, 1594)
                self.answer_obj_head = nn.Linear(hs, 3)
                self.answer_global_head = nn.Linear(hs, 111)
                self.answer_attr_head = nn.Linear(hs, 403)
                self.answer_cat_head = nn.Linear(hs, 678)
                self.answer_criterion = QACriterionGQA()

                self.transformer.gqa_ans_token_init()

            # initialize the answer tokens for clever
            if self.hparams.config["loss_names"]["clever_mdetr"] > 0:
                self.answer_type_head = nn.Linear(hs, 3)
                self.answer_binary_head = nn.Linear(hs, 1)
                self.answer_attr_head = nn.Linear(hs, 15)
                self.answer_reg_head = MLP(hs, hs, 20, 3)
                self.answer_criterion = QACriterionClever()
                self.transformer.clever_ans_token_init()

            # initialize the answer tokens for snlive
            if self.hparams.config["loss_names"]["snlive"] > 0:
                self.answer_head = nn.Linear(hs, 3)
                self.transformer.snlive_ans_token_init()
                self.answer_criterion = nn.CrossEntropyLoss()


        if self.hparams.config["loss_names"]["nlvr2"] > 0:
            self.nlvr2_classifier = nn.Sequential(
                nn.Linear(hs * 2, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, 2),
            )
            self.nlvr2_classifier.apply(objectives.init_weights)
            emb_data = self.token_type_embeddings.weight.data
            self.token_type_embeddings = nn.Embedding(3, hs)
            self.token_type_embeddings.apply(objectives.init_weights)
            self.token_type_embeddings.weight.data[0, :] = emb_data[0, :]
            self.token_type_embeddings.weight.data[1, :] = emb_data[1, :]
            self.token_type_embeddings.weight.data[2, :] = emb_data[1, :]



        if self.hparams.config["loss_names"]["irtr"] > 0:
            self.rank_output = nn.Linear(hs, 1)
            self.rank_output.weight.data = self.itm_score.fc.weight.data[1:, :]
            self.rank_output.bias.data = self.itm_score.fc.bias.data[1:]
            self.margin = 0.2
            for p in self.itm_score.parameters():
                p.requires_grad = False

        vilt_utils.set_metrics(self)
        self.current_tasks = list()

        # ===================== load downstream (test_only) ======================

        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)
        # ===================== load downstream (resume) ======================

        if self.hparams.config["load_path"] != "" and (self.hparams.config["resume"] or self.hparams.config["resume_qa"]):
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)

        # load yolos_pretrain det tokens
        if self.hparams.config["add_det_token"] and self.hparams.config["load_yolos_pretrain_det"] and not self.hparams.config["test_only"]:
            yolo_det_token = torch.load(self.hparams.config["yolos_pretrain_det_path"] + "base_det_token.pt")[:,:self.hparams.config["det_token_num"], :]
            yolo_det_pos_embed = torch.load(self.hparams.config["yolos_pretrain_det_path"] + "base_det_pos_embed.pt")[:,:self.hparams.config["det_token_num"], :]
            self.transformer.det_token.data = copy.deepcopy(yolo_det_token.data)
            self.transformer.det_pos_embed.data = copy.deepcopy(yolo_det_pos_embed.data)



        # truncate/fuse/reinit the det token after the pretrain model
        if self.hparams.config["add_det_token"] == True and self.hparams.config["truncated_det_token_num"] and self.hparams.config["use_det_fuser"] == False:
            assert self.hparams.config["truncated_det_token_num"] <= self.hparams.config["det_token_num"]
            self.transformer.truncate_det_token(desired_token = self.hparams.config["truncated_det_token_num"])
            self.hparams.config["det_token_num"] = self.hparams.config["truncated_det_token_num"]
        elif self.hparams.config["add_det_token"] == True and self.hparams.config["truncated_det_token_num"] and self.hparams.config["use_det_fuser"] == True:
            self.transformer.fuse_tokens(self.hparams.config["det_token_num"] , self.hparams.config["truncated_det_token_num"])
        if self.hparams.config["add_det_token"] == True and self.hparams.config["det_token_reinit"] and self.hparams.config["truncated_det_token_num"]:
            self.transformer.det_init(det_token_num = self.hparams.config["truncated_det_token_num"])


    def infer(
        self,
        batch,
        mask_text=False,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
    ):

        if f"image_{image_token_type_idx - 1}" in batch:
            imgkey = f"image_{image_token_type_idx - 1}"
        else:
            imgkey = "image"

        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch[f"text_masks"]
        text_embeds = self.text_embeddings(text_ids)

        if image_embeds is None and image_masks is None:
            # curate supervision for bbox patch label
            bbox_patch_label_lst = None
            if self.hparams.config["patch_level_alignment"] or self.hparams.config["patch_level_alignment_KL"]:
                bbox_patch_label_lst = []
                grid = torch.stack(
                    torch.meshgrid(torch.arange(batch['image'][0].shape[3]/self.hparams.config["patch_size"]), torch.arange(batch['image'][0].shape[2]/self.hparams.config["patch_size"])), dim=-1, ).permute(2,0,1)
                grid = grid.flatten(1)*self.hparams.config["patch_size"]
                grid = torch.cat([grid, grid + self.hparams.config["patch_size"]], dim=0)
                grid = grid.cuda().permute(1,0)
                grid[:, 0] = grid[:, 0]/batch['image'][0].shape[2]
                grid[:, 1] = grid[:, 1]/batch['image'][0].shape[3]
                grid[:, 2] = grid[:, 2]/batch['image'][0].shape[2]
                grid[:, 3] = grid[:, 3]/batch['image'][0].shape[3]
                area_per_grid = box_area(grid)

                for b in range(batch['image'][0].shape[0]):
                    gt_bboxes_xyxy = box_cxcywh_to_xyxy(torch.cat([torch.FloatTensor(batch["gt_bbox"][b]).cuda()], dim=0))
                    gt_bboxes_xyxy[:,0] = gt_bboxes_xyxy[:,0]*batch["width"][b]/batch['image'][0].shape[2] 
                    gt_bboxes_xyxy[:,1] = gt_bboxes_xyxy[:,1]*batch["height"][b]/batch['image'][0].shape[3] 
                    gt_bboxes_xyxy[:,2] = gt_bboxes_xyxy[:,2]*batch["width"][b]/batch['image'][0].shape[2] 
                    gt_bboxes_xyxy[:,3] = gt_bboxes_xyxy[:,3]*batch["height"][b]/batch['image'][0].shape[3]

                    lt = torch.max(grid[:, None, :2], gt_bboxes_xyxy[:, :2])  # [N,M,2]
                    rb = torch.min(grid[:, None, 2:], gt_bboxes_xyxy[:, 2:])  # [N,M,2]
                    wh = (rb - lt).clamp(min=0)  # [N,M,2]
                    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
                    iou = inter/area_per_grid.view(-1,1).repeat(1, gt_bboxes_xyxy.shape[0])
                    bbox_patch_label = (iou > 0.5)
                    assert gt_bboxes_xyxy.shape[0] == len(batch["gt_bbox"][b])
                    bbox_patch_label_lst.append(bbox_patch_label)
                batch["bbox_patch_label"] = bbox_patch_label_lst



            img = batch[imgkey][0]

            (
                image_embeds,
                image_masks,
                patch_index,
                image_labels,
                bbox_patch_label
            ) = self.transformer.visual_embed(
                img,
                max_image_len=self.hparams.config["max_image_len"],
                mask_it=mask_image,
                use_det_fuser=self.hparams.config["use_det_fuser"],
                bbox_patch_label=batch["bbox_patch_label"] if "bbox_patch_label" in batch else None
            )
        else:
            bbox_patch_label  = None
            patch_index, image_labels = (
                None,
                None,
            )



        text_embeds, image_embeds = (
            text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
            image_embeds
            + self.token_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx)
            ),
        )


        co_embeds = torch.cat([text_embeds, image_embeds], dim=1)
        co_masks = torch.cat([text_masks, image_masks], dim=1)


        x = co_embeds


        for i, blk in enumerate(self.transformer.blocks):
            x, _attn = blk(x, mask=co_masks)

        x = self.transformer.norm(x)



        if self.hparams.config["add_det_token"]==True:
            text_feats, image_feats, det_token_feats = (
                x[:, : text_embeds.shape[1]],
                x[:, text_embeds.shape[1] : -self.transformer.det_token_num],
                x[:, -self.transformer.det_token_num :],
            )
        else:
            text_feats, image_feats = (
                x[:, : text_embeds.shape[1]],
                x[:, text_embeds.shape[1] :],
            )
 
        if self.hparams.config["add_det_token"] == True:
            cls_feats = self.pooler(torch.cat((text_feats, image_feats), dim=1))
            ret = {
                "det_token_feats":det_token_feats,
                "text_feats": text_feats,
                "image_feats": image_feats,
                "cls_feats": cls_feats,
                "raw_cls_feats": x[:, 0],
                "image_labels": image_labels,
                "image_masks": image_masks,
                "text_labels": text_labels,
                "text_ids": text_ids,
                "text_masks": text_masks,
                "patch_index": patch_index,
                "bbox_patch_label":bbox_patch_label,
                "attn": _attn
            }
        else:
            cls_feats = self.pooler(torch.cat((text_feats, image_feats), dim=1))
            ret = {
                "text_feats": text_feats,
                "image_feats": image_feats,
                "cls_feats": cls_feats,
                "raw_cls_feats": x[:, 0],
                "image_labels": image_labels,
                "image_masks": image_masks,
                "text_labels": text_labels,
                "text_ids": text_ids,
                "text_masks": text_masks,
                "patch_index": patch_index,
                "bbox_patch_label":bbox_patch_label
            }

        return ret

    def forward(self, batch):
        ret = dict()

        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        # Masked Language Modeling
        if "mlm" in self.current_tasks:
            ret.update(objectives.compute_mlm(self, batch))

        # Masked Patch Prediction
        if "mpp" in self.current_tasks:
            ret.update(objectives.compute_mpp(self, batch))

        # Image Text Matching
        if "itm" in self.current_tasks:
            ret.update(objectives.compute_itm_wpa(self, batch))

        # Visual Question Answering
        if "vqa" in self.current_tasks:
            ret.update(objectives.compute_vqa(self, batch))

        # Natural Language for Visual Reasoning 2
        if "nlvr2" in self.current_tasks:
            ret.update(objectives.compute_nlvr2(self, batch))

        # Image Retrieval and Text Retrieval
        if "irtr" in self.current_tasks:
            ret.update(objectives.compute_irtr(self, batch))

        # Referring expression comprehension (REC) task, including refcoco/refcocog/refcoco+/referitgame
        if "rec" in self.current_tasks:
            ret.update(objectives.compute_rec(self, batch))

        # mdetr pretrain for modulated detection task
        if "mdetr_pretrain" in self.current_tasks:
            ret.update(objectives.compute_mdetr_pretrain(self, batch))

        # mdetr pretrain ablade for modulated detection task
        if "mdetr_pretrain_ablade" in self.current_tasks:
            ret.update(objectives.compute_mdetr_pretrain_ablade(self, batch))
            
        # flickr_mdetr 
        if "flickr_mdetr" in self.current_tasks:
            ret.update(objectives.compute_flickr_mdetr(self, batch))

        # gqa_mdetr 
        if "gqa_mdetr" in self.current_tasks:
            ret.update(objectives.compute_gqa_mdetr(self, batch))

        # clever_mdetr 
        if "clever_mdetr" in self.current_tasks:
            ret.update(objectives.compute_clever_mdetr(self, batch))

        # snlive
        if "snlive" in self.current_tasks:
            ret.update(objectives.compute_snlive(self, batch))

        # copsref 
        if "copsref" in self.current_tasks:
            ret.update(objectives.compute_copsref(self, batch))


        return ret

    def training_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])

        return total_loss

    def training_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)
        ret = dict()

        if self.hparams.config["loss_names"]["vqa"] > 0:
            ret.update(objectives.vqa_test_step(self, batch, output))
        return ret

    def test_epoch_end(self, outs):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            objectives.vqa_test_wrapup(outs, model_name)
        vilt_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return vilt_utils.set_schedule(self)


