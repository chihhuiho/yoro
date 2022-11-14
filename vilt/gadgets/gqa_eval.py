# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn.functional as F
from torch import nn


class QACriterionGQA(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, answers):
        loss = {}

        device = output["pred_answer_type"].device
        loss["loss_answer_type"] = F.cross_entropy(output["pred_answer_type"], answers["answer_type"])

        type_acc = output["pred_answer_type"].argmax(-1) == answers["answer_type"]
        loss["accuracy_answer_type"] = type_acc.sum() / answers["answer_type"].numel()

        is_obj = answers["answer_type"] == 0
        is_attr = answers["answer_type"] == 1
        is_rel = answers["answer_type"] == 2
        is_global = answers["answer_type"] == 3
        is_cat = answers["answer_type"] == 4

        ## OBJ type
        obj_norm = is_obj.sum() if is_obj.any() else 1.0
        loss["loss_answer_obj"] = (
            F.cross_entropy(output["pred_answer_obj"], answers["answer_obj"], reduction="none")
            .masked_fill(~is_obj, 0)
            .sum()
            / obj_norm
        )
        obj_acc = (output["pred_answer_obj"].argmax(-1)) == answers["answer_obj"]
        loss["accuracy_answer_obj"] = (
            obj_acc[is_obj].sum() / is_obj.sum() if is_obj.any() else torch.as_tensor(1.0, device=device)
        )

        ## ATTR type
        attr_norm = is_attr.sum() if is_attr.any() else 1.0
        loss["loss_answer_attr"] = (
            F.cross_entropy(output["pred_answer_attr"], answers["answer_attr"], reduction="none")
            .masked_fill(~is_attr, 0)
            .sum()
            / attr_norm
        )
        attr_acc = (output["pred_answer_attr"].argmax(-1)) == answers["answer_attr"]
        loss["accuracy_answer_attr"] = (
            attr_acc[is_attr].sum() / is_attr.sum() if is_attr.any() else torch.as_tensor(1.0, device=device)
        )

        ## REL type
        rel_norm = is_rel.sum() if is_rel.any() else 1.0
        loss["loss_answer_rel"] = (
            F.cross_entropy(output["pred_answer_rel"], answers["answer_rel"], reduction="none")
            .masked_fill(~is_rel, 0)
            .sum()
            / rel_norm
        )
        rel_acc = (output["pred_answer_rel"].argmax(-1)) == answers["answer_rel"]
        loss["accuracy_answer_rel"] = (
            rel_acc[is_rel].sum() / is_rel.sum() if is_rel.any() else torch.as_tensor(1.0, device=device)
        )

        ## GLOBAL type
        global_norm = is_global.sum() if is_global.any() else 1.0
        loss["loss_answer_global"] = (
            F.cross_entropy(output["pred_answer_global"], answers["answer_global"], reduction="none")
            .masked_fill(~is_global, 0)
            .sum()
            / global_norm
        )
        global_acc = (output["pred_answer_global"].argmax(-1)) == answers["answer_global"]
        loss["accuracy_answer_global"] = (
            global_acc[is_global].sum() / is_global.sum() if is_global.any() else torch.as_tensor(1.0, device=device)
        )

        ## CAT type
        cat_norm = is_cat.sum() if is_cat.any() else 1.0
        loss["loss_answer_cat"] = (
            F.cross_entropy(output["pred_answer_cat"], answers["answer_cat"], reduction="none")
            .masked_fill(~is_cat, 0)
            .sum()
            / cat_norm
        )
        cat_acc = (output["pred_answer_cat"].argmax(-1)) == answers["answer_cat"]
        loss["accuracy_answer_cat"] = (
            cat_acc[is_cat].sum() / is_cat.sum() if is_cat.any() else torch.as_tensor(1.0, device=device)
        )

        loss["accuracy_answer_total"] = (
            type_acc
            * (is_obj * obj_acc + is_rel * rel_acc + is_attr * attr_acc + is_global * global_acc + is_cat * cat_acc)
        ).sum() / type_acc.numel()

        return loss


