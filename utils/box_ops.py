# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
from torchvision.ops.boxes import box_area

from skimage.draw import polygon
from skimage.feature import peak_local_max
import numpy as np
from shapely.geometry import Polygon


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)

def get_coordinated_fron_cxcy_theta(box_coordinate, angle):

    scale_factor = 224

    center_y = box_coordinate[0] * 224
    centre_x = box_coordinate[1] * 224
    width = box_coordinate[2] * 224
    length = box_coordinate[3] * 224

    xo = torch.cos(angle)
    yo = torch.sin(angle)

    y1 = center_y + length / 2 * yo
    x1 = centre_x - length / 2 * xo
    y2 = center_y - length / 2 * yo
    x2 = centre_x + length / 2 * xo

    return [
            y1 - width / 2 * xo, x1 - width / 2 * yo,
            y2 - width / 2 * xo, x2 - width / 2 * yo,
            y2 + width / 2 * xo, x2 + width / 2 * yo,
            y1 + width / 2 * xo, x1 + width / 2 * yo
            ]
    


def iou_inclined_boxes(boxes_src, boxes_trgt, angle_src, angle_trgt ):

    #print("calculating the IOU")
    #print(boxes_src)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    iou = []
    #with torch.no_grad():
    for i in range(boxes_src.shape[0]):
        #print()
        boxes1 = boxes_src[i, :]
        boxes2 = boxes_trgt[i, :]
        #print(boxes1)
        #print(boxes2)
        angles1 = angle_src[i, :]
        angles2 = angle_trgt[i, :]

        boxes1 =  get_coordinated_fron_cxcy_theta(boxes1, angles1)
        boxes2 = get_coordinated_fron_cxcy_theta(boxes2, angles2)
        #print
        #boxes1 = boxes1.cpu().detach().numpy()
        #boxes2 = boxes2.cpu().detach().numpy()
        #print(boxes1)
        #print(boxes2)

        boxes1 = [torch.round(num) for num in boxes1]
        
        a = Polygon([(boxes1[0], boxes1[1]), (boxes1[2], boxes1[3]), (boxes1[4], boxes1[5]), (boxes1[6], boxes1[7])])
        b = Polygon([(boxes2[0], boxes2[1]), (boxes2[2], boxes2[3]), (boxes2[4], boxes2[5]), (boxes2[6], boxes2[7])])

        try:
            if a.union(b).area > 0:
            #print(a.intersection(b).area / a.union(b).area) 
                iou.append(a.intersection(b).area / a.union(b).area)
            else:
                iou.append(0)
        except:
            iou.append(0)

    #print("box tensor")
    #print(iou)
    #print("box_tensor")
    print(torch.tensor( iou).to(device))


    return  torch.tensor( iou).to(device)


def final_iou(box_src, boxes_trgt, angle_src, angle_trgt ):

    #print("calculating the IOU")
    #print(boxes_src)
    """
    print("final iou")
    print(box_src )
    print( boxes_trgt )
    print( angle_src )
    print( angle_trgt )
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    iou = []

    max_iou = 0 
    boxes1 = box_src
    angles1 = angle_src
    boxes1 =  get_coordinated_fron_cxcy_theta(boxes1, angles1)
    boxes1 = [torch.round(num) for num in boxes1]
    final_angle = 0
    #with torch.no_grad():
    for i in range(boxes_trgt.shape[0]):
        #print()
        
        boxes2 = boxes_trgt[i, :]
        #print(boxes1)
        #print(boxes2)
        
        angles2 = angle_trgt[i, :]

        
        boxes2 = get_coordinated_fron_cxcy_theta(boxes2, angles2)
        #print
        #boxes1 = boxes1.cpu().detach().numpy()
        #boxes2 = boxes2.cpu().detach().numpy()
        #print(boxes1)
        #print(boxes2)

        
        
        a = Polygon([(boxes1[0], boxes1[1]), (boxes1[2], boxes1[3]), (boxes1[4], boxes1[5]), (boxes1[6], boxes1[7])])
        b = Polygon([(boxes2[0], boxes2[1]), (boxes2[2], boxes2[3]), (boxes2[4], boxes2[5]), (boxes2[6], boxes2[7])])

        try:
            if a.union(b).area > 0:
            #print(a.intersection(b).area / a.union(b).area) 
                final_iou = a.intersection(b).area / a.union(b).area
                iou.append( final_iou )

                if final_iou > max_iou:
                    max_iou = final_iou
                    final_angle = angles2
            else:
                iou.append(0)
        except:
            iou.append(0)

    #print("box tensor")
    #print(iou)
    #print("box_tensor")
    #print(torch.tensor( iou).to(device))


    return  max_iou, torch.tensor( iou).to(device) , final_angle
