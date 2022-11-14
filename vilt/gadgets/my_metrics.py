import torch
from pytorch_lightning.metrics import Metric
import numpy as np
from vilt.transforms.coco import box_cxcywh_to_xyxy
import torch.nn.functional as F


'''
# code from https://github.com/ChenRocks/UNITER/blob/5b4c9faf8ed922176b20d89ac56a3e0b39374a22/data/re.py#L226
# input is bbox [x,y,w,h]
def calculate_iou(pred_boxes, gt_boxes, img_width, img_height):
    pred_boxes = pred_boxes.cpu().numpy().astype(float)
    pred_boxes = np.clip(pred_boxes, 0.0, 1.0).astype(float)
    gt_boxes = gt_boxes.cpu().numpy().astype(float)
    iou = []
    for b in range(pred_boxes.shape[0]):
        box1 = pred_boxes[b].astype(float)
        box2 = gt_boxes[b].astype(float)
        box1[[0,2]], box2[[0,2]] = box1[[0,2]]*img_width[b], box2[[0,2]]*img_width[b] 
        box1[[1,3]], box2[[1,3]] = box1[[1,3]]*img_height[b], box2[[1,3]]*img_height[b] 

        # each box is of [x1, y1, w, h]
        inter_x1 = max(box1[0], box2[0])
        inter_y1 = max(box1[1], box2[1])
        inter_x2 = min(box1[0]+box1[2], box2[0]+box2[2])
        inter_y2 = min(box1[1]+box1[3], box2[1]+box2[3])
        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
        else:
            inter = 0
        union = box1[2]*box1[3] + box2[2]*box2[3] - inter
        iou.append(float(inter)/union)
    return np.array(iou)
'''
# code from https://github.com/jackroos/VL-BERT/blob/master/refcoco/function/test.py#L20
# input is bbox [x,y,w,h]
def calculate_iou(pred_boxes, gt_boxes, img_width, img_height):
 
    img_width = np.repeat(np.expand_dims(np.asarray(img_width), axis=1),2, axis=1).astype(float) 
    img_height = np.repeat(np.expand_dims(np.asarray(img_height), axis=1),2, axis=1).astype(float) 
    pred_boxes = pred_boxes.cpu().numpy().astype(float) 
    gt_boxes = gt_boxes.cpu().numpy().astype(float) 
    # compute the non-scale box
    pred_boxes = np.clip(pred_boxes, 0.0, 1.0).astype(float)
    pred_boxes[:, [2,3]] += pred_boxes[:, [0, 1]]
    #pred_boxes = np.clip(pred_boxes, 0.0, 1.0).astype(float)
    gt_boxes = gt_boxes.astype(float) 
    gt_boxes[:, [2,3]] += gt_boxes[:, [0, 1]]

    gt_boxes[:,[0,2]] = gt_boxes[:,[0,2]]*img_width
    gt_boxes[:,[1,3]] = gt_boxes[:,[1,3]]*img_height

    pred_boxes[:,[0,2]] = pred_boxes[:,[0,2]]*img_width
    pred_boxes[:,[1,3]] = pred_boxes[:,[1,3]]*img_height
  
    x11, y11, x12, y12 = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2] , pred_boxes[:, 3]
    x21, y21, x22, y22 = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2], gt_boxes[:, 3]
    xA = np.maximum(x11, x21)
    yA = np.maximum(y11, y21)
    xB = np.minimum(x12, x22)
    yB = np.minimum(y12, y22)
    interArea = (xB - xA + 1).clip(0.0) * (yB - yA + 1).clip(0.0)
    #interArea = (xB - xA + 1)* (yB - yA + 1)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    iou = interArea / (boxAArea + boxBArea - interArea)
    return iou


class DetAcc_Flickr_Anybox(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("iou", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("detacc", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, _pred_boxes, _gt_boxes, img_width, img_height, _pred_logits=None, pos_thres=0.5):

        pred_logits = _pred_logits.detach()
        pred_boxes = _pred_boxes.detach()
        gt_boxes = torch.cat([g.detach() for g in _gt_boxes])
        prob = F.softmax(pred_logits, -1)
        scores = 1-prob[:,:,-1]
 
        _, idx = torch.max(scores, 1)
        idx = idx.long()
        idx = torch.unsqueeze(torch.unsqueeze(idx, 1),2)
        idx = torch.repeat_interleave(idx, 4, dim=2)
        pred_boxes = torch.gather(pred_boxes, dim=1, index=idx).squeeze()

        
        pred_boxes, gt_boxes = (
            pred_boxes.detach().to(self.iou.device),
            gt_boxes.detach().to(self.iou.device),
        )
        pred_boxes = box_cxcywh_to_xyxy(pred_boxes)
        if list(pred_boxes.shape) != 2:
            pred_boxes = pred_boxes.view(-1,4)

        pred_boxes[:,[2,3]] -= pred_boxes[:,[0,1]]
        gt_boxes = box_cxcywh_to_xyxy(gt_boxes)
        gt_boxes[:,[2,3]] -= gt_boxes[:,[0,1]]
        

        iou = torch.Tensor(calculate_iou(pred_boxes, gt_boxes, img_width, img_height))
        self.iou += torch.sum(iou)
        self.detacc += torch.sum(iou >= pos_thres)
        self.total += pred_boxes.shape[0]

    def compute(self):
        return self.detacc / self.total



class DetAcc(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("iou", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("detacc", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, _pred_boxes, _gt_boxes, img_width, img_height, _pred_logits=None, pos_thres=0.5):

        if _pred_logits is not None:
            pred_logits = _pred_logits.detach()
            pred_boxes = _pred_boxes.detach()
            gt_boxes = torch.cat([g.detach() for g in _gt_boxes])
            prob = F.softmax(pred_logits, -1)
            scores = 1-prob[:,:,-1]

            _, idx = torch.max(scores, 1)
            idx = idx.long()
            idx = torch.unsqueeze(torch.unsqueeze(idx, 1),2)
            idx = torch.repeat_interleave(idx, 4, dim=2)
            pred_boxes = torch.gather(pred_boxes, dim=1, index=idx).squeeze()

        else:
            pred_boxes = _pred_boxes
            gt_boxes = _gt_boxes
            
        pred_boxes, gt_boxes = (
            pred_boxes.detach().to(self.iou.device),
            gt_boxes.detach().to(self.iou.device),
        )
        pred_boxes = box_cxcywh_to_xyxy(pred_boxes)
        if list(pred_boxes.shape) != 2:
            pred_boxes = pred_boxes.view(-1,4)

        pred_boxes[:,[2,3]] -= pred_boxes[:,[0,1]]
        gt_boxes = box_cxcywh_to_xyxy(gt_boxes)
        gt_boxes[:,[2,3]] -= gt_boxes[:,[0,1]]


        iou = torch.Tensor(calculate_iou(pred_boxes, gt_boxes, img_width, img_height))
        self.iou += torch.sum(iou)
        self.detacc += torch.sum(iou >= pos_thres)
        self.total += pred_boxes.shape[0]


    def compute(self):
        return self.detacc / self.total


class Accuracy(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, logits, target):
        logits, target = (
            logits.detach().to(self.correct.device),
            target.detach().to(self.correct.device),
        )
        preds = logits.argmax(dim=-1)
        preds = preds[target != -100]
        target = target[target != -100]
        if target.numel() == 0:
            return 1

        assert preds.shape == target.shape

        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        return self.correct / self.total


class Scalar(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("scalar", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, scalar, total=1):
        if isinstance(scalar, torch.Tensor):
            scalar = scalar.detach().to(self.scalar.device)
        else:
            scalar = torch.tensor(scalar).float().to(self.scalar.device)
        self.scalar += scalar
        self.total += total
 
    def compute(self):
        return self.scalar / self.total


class VQAScore(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, logits, target):
        logits, target = (
            logits.detach().float().to(self.score.device),
            target.detach().float().to(self.score.device),
        )
        logits = torch.max(logits, 1)[1]
        one_hots = torch.zeros(*target.size()).to(target)
        one_hots.scatter_(1, logits.view(-1, 1), 1)
        scores = one_hots * target

        self.score += scores.sum()
        self.total += len(logits)

    def compute(self):
        return self.score / self.total
