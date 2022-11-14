# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn.functional as F
from torch import nn


class QACriterionClever(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, answers):
        loss = {}
        loss["loss_answer_type"] = F.cross_entropy(output["pred_answer_type"], answers["answer_type"])

        type_acc = output["pred_answer_type"].argmax(-1) == answers["answer_type"]
        loss["accuracy_answer_type"] = type_acc.sum() / answers["answer_type"].numel()

        is_binary = answers["answer_type"] == 0
        is_attr = answers["answer_type"] == 1
        is_reg = answers["answer_type"] == 2

        binary_norm = is_binary.sum() if is_binary.any() else 1.0

        loss["loss_answer_binary"] = (
            F.binary_cross_entropy_with_logits(output["pred_answer_binary"], answers["answer_binary"], reduction="none")
            .masked_fill(~is_binary, 0)
            .sum()
            / binary_norm
        )
        bin_acc = (output["pred_answer_binary"].sigmoid() > 0.5) == answers["answer_binary"]
        loss["accuracy_answer_binary"] = (
            bin_acc[is_binary].sum() / is_binary.sum() if is_binary.any() else torch.as_tensor(1.0)
        )

        reg_norm = is_reg.sum() if is_reg.any() else 1.0
        loss["loss_answer_reg"] = (
            F.cross_entropy(output["pred_answer_reg"], answers["answer_reg"], reduction="none")
            .masked_fill(~is_reg, 0)
            .sum()
            / reg_norm
        )
        reg_acc = (output["pred_answer_reg"].argmax(-1)) == answers["answer_reg"]
        loss["accuracy_answer_reg"] = reg_acc[is_reg].sum() / is_reg.sum() if is_reg.any() else torch.as_tensor(1.0)

        attr_norm = is_attr.sum() if is_attr.any() else 1.0
        loss["loss_answer_attr"] = (
            F.cross_entropy(output["pred_answer_attr"], answers["answer_attr"], reduction="none")
            .masked_fill(~is_attr, 0)
            .sum()
            / attr_norm
        )
        attr_acc = (output["pred_answer_attr"].argmax(-1)) == answers["answer_attr"]
        loss["accuracy_answer_attr"] = (
            attr_acc[is_attr].sum() / is_attr.sum() if is_attr.any() else torch.as_tensor(1.0)
        )

        loss["accuracy_answer_total"] = (
            type_acc * (is_binary * bin_acc + is_reg * reg_acc + is_attr * attr_acc)
        ).sum() / type_acc.numel()

        return loss


