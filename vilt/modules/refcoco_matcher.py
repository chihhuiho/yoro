# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from vilt.transforms.coco import box_cxcywh_to_xyxy, generalized_box_iou
import torch.nn.functional as F

class HungarianMatcherRefCoco(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_bbox: float = 1, cost_giou: float = 1, use_cls=False, cost_class: float = 1, use_conf=False):
        """Creates the matcher

        Params:
            cost_conf: This is the relative weight of the confidence score that measure the alignment between the text and proposed bbox
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_class = cost_class
        self.use_cls = use_cls
        self.use_conf = use_conf
        self.weight_dict = {'loss_bbox': cost_bbox, 'loss_giou':cost_giou, 'loss_class':cost_class}
        assert cost_class!=0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def compute_match_indices(self, outputs, targets, multioutput=False, multioutput_mix=False):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_boxes"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        if self.use_cls:
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets]).long()
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        if self.use_cls:
            cost_class = -out_prob[:, tgt_ids] # https://github.com/hustvl/YOLOS/blob/main/models/matcher.py
        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        sizes = [len(v["boxes"]) for v in targets]

        self.mutlioutput = multioutput
        if self.mutlioutput:
            tmp = -cost_giou.view(bs, num_queries, -1).cpu()
            mask_lst = []
            for i,c in enumerate(tmp.split(sizes, -1)):
                mask = (c[i] >= 0.5).int().view(num_queries) 
                #min_idx = (c[i] == torch.max(c[i])).nonzero()[0,0]
                #mask[min_idx] = 1
                mask_lst.append(mask.view(1, -1))
            mask_lst = torch.cat(mask_lst, dim = 0)
            return mask_lst

        self.mutlioutput_mix = multioutput_mix
        if multioutput_mix:
            tmp = -cost_giou.view(bs, num_queries, -1).cpu()
            mask_lst = []
            for i,c in enumerate(tmp.split(sizes, -1)):
                mask = (c[i] >= 0.5).int().view(num_queries) 
                #min_idx = (c[i] == torch.max(c[i])).nonzero()[0,0]
                #mask[min_idx] = 1
                mask_lst.append(mask.view(1, -1))
            mask_lst = torch.cat(mask_lst, dim = 0)


        # Final cost matrix
        if self.use_cls:
            C = self.cost_bbox * cost_bbox + self.cost_giou * cost_giou + self.cost_class*cost_class
        else:
            C = self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

        if multioutput_mix:
            return mask_lst, indices
        
        return indices

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        if self.mutlioutput:
            '''
            losses_dict = {}
            indices = indices.view(-1).to(outputs["pred_boxes"].device)
            num_boxes = sum(indices)
            bs, num_queries = outputs["pred_boxes"].shape[:2]
            src_boxes = outputs['pred_boxes']
            target_boxes = torch.cat([t["boxes"].view(1,1,-1) for t in targets], dim=0).expand(-1, num_queries,-1)
            loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none').mean(-1).view(-1)
            loss_bbox *= indices
            losses_dict['loss_bbox'] = loss_bbox.sum() / num_boxes
            loss_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_boxes.view(-1, 4)),box_cxcywh_to_xyxy(target_boxes.contiguous().view(-1,4))))
            loss_giou *= indices
            losses_dict['loss_giou'] = loss_giou.sum() / num_boxes
            if self.use_cls:
                src_logits = outputs['pred_logits'].view(bs*num_queries, -1)
                target_classes = torch.cat([t["labels"] for t in targets], dim=0).view(bs,1).expand(-1, num_queries).contiguous().view(-1).long()
                loss_ce = F.cross_entropy(src_logits, target_classes, reduction='none')
                loss_ce *= indices
                losses_dict['loss_class'] = loss_ce.sum() / num_boxes
            '''
            losses_dict = {}
            indices = indices.view(-1).to(outputs["pred_boxes"].device)
            bs, num_queries = outputs["pred_boxes"].shape[:2]
            num_boxes = bs*num_queries
            src_boxes = outputs['pred_boxes']
            target_boxes = torch.cat([t["boxes"].view(1,1,-1) for t in targets], dim=0).expand(-1, num_queries,-1)
            loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none').mean(-1).view(-1)
            losses_dict['loss_bbox'] = loss_bbox.sum() / num_boxes
            loss_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_boxes.view(-1, 4)),box_cxcywh_to_xyxy(target_boxes.contiguous().view(-1,4))))
            losses_dict['loss_giou'] = loss_giou.sum() / num_boxes
        
            src_logits = outputs['pred_logits'].view(bs*num_queries, -1)

            target_classes = indices.long()
            loss_ce = F.cross_entropy(src_logits, target_classes, reduction='none')
            losses_dict['loss_class'] = loss_ce.sum() / num_boxes
            loss = sum(losses_dict[k] * self.weight_dict[k] for k in losses_dict.keys() if k in self.weight_dict)
        else:
            idx = self._get_src_permutation_idx(indices)
            bs, num_queries = outputs["pred_boxes"].shape[:2]
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
            losses_dict = {}
            losses_dict['loss_bbox'] = loss_bbox.sum() / num_boxes
            loss_giou = 1 - torch.diag(generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(target_boxes)))
            losses_dict['loss_giou'] = loss_giou.sum() / num_boxes
            if self.use_cls:
                src_logits = outputs['pred_logits'][idx]
                target_classes = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)], dim=0).long()
                loss_ce = F.cross_entropy(src_logits, target_classes)
                losses_dict['loss_class'] = loss_ce
            elif self.use_conf:
                src_logits = outputs['pred_logits'].view(bs*num_queries, 2)
                target_classes = torch.zeros((bs, num_queries))
                target_classes[idx] = 1
                target_classes = target_classes.long().to(src_logits.device).view(-1)
                loss_ce = F.cross_entropy(src_logits, target_classes)
                losses_dict['loss_class'] = loss_ce

            #print("=============") 
            #print(self.weight_dict['loss_bbox']*losses_dict['loss_bbox'])
            #print(self.weight_dict['loss_giou']*losses_dict['loss_giou'])
            #print(self.weight_dict['loss_class']*losses_dict['loss_class'])
            #print("=============")            
            loss = sum(losses_dict[k] * self.weight_dict[k] for k in losses_dict.keys() if k in self.weight_dict)
        return loss, losses_dict, src_boxes


    def loss_boxesv2(self, outputs, targets, indices, mask, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs

        losses_dict = {}
        idx = self._get_src_permutation_idx(indices)
        mask = mask.to(outputs["pred_boxes"].device)
        target_classes = mask.long().clone().view(-1)
        mask[idx] = 1
        mask = mask.view(-1)


        bs, num_queries = outputs["pred_boxes"].shape[:2]
        num_boxes = sum(mask)
        src_boxes = outputs['pred_boxes']
        target_boxes = torch.cat([t["boxes"].view(1,1,-1) for t in targets], dim=0).expand(-1, num_queries,-1)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none').mean(-1).view(-1)
        loss_bbox *= mask
        losses_dict['loss_bbox'] = loss_bbox.sum() / num_boxes
        loss_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_boxes.view(-1, 4)),box_cxcywh_to_xyxy(target_boxes.contiguous().view(-1,4))))
        loss_giou *= mask
        losses_dict['loss_giou'] = loss_giou.sum() / num_boxes
        
        src_logits = outputs['pred_logits'].view(bs*num_queries, -1)
        loss_ce = F.cross_entropy(src_logits, target_classes, reduction='mean')
        losses_dict['loss_class'] = loss_ce.sum() 
        loss = sum(losses_dict[k] * self.weight_dict[k] for k in losses_dict.keys() if k in self.weight_dict)

        return loss, losses_dict, src_boxes



    def loss_boxes_multistage(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        losses_dict = {}
        indices = indices.view(-1).to(outputs["pred_boxes"].device)
        num_boxes = sum(indices)
        bs, num_queries = outputs["pred_boxes"].shape[:2]
        src_boxes = outputs['pred_boxes']

        losses_dict['loss_bbox'] = 0
        losses_dict['loss_giou'] = 0 
        
        src_logits = outputs['pred_logits'].view(bs*num_queries, -1)
        target_classes = indices.long()
        loss_ce = F.cross_entropy(src_logits, target_classes, reduction='mean')
        losses_dict['loss_class'] = loss_ce 


        loss = losses_dict['loss_class']

        return loss, losses_dict, src_boxes



if __name__ == "__main__":
    
    matcher = HungarianMatcherRefCoco(cost_bbox=5, cost_giou=2)
    bs = 8
    query=10
    max_bbox=10
    outputs = {"pred_boxes": torch.rand(bs, query, 4)}
    targets = [{"boxes":torch.rand(i , 4)} for i in range(bs)]
    for i in range(len(targets)):
        targets[i]["boxes"][:, 2:] += targets[i]["boxes"][:, :2]
    loss, loss_dict  = matcher(outputs, targets)
    print(loss)
    print(loss_dict)

