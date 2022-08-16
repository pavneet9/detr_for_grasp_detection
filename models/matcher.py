# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, iou_inclined_boxes


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_centre: float = 1, cost_angle: float = 1, cost_dim: float = 1 , cost_class: float = 3):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        """
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"
        """
        self.cost_centre = cost_centre
        self.cost_angle = cost_angle
        self.cost_dim = cost_dim
        self.cost_class = cost_class
        assert cost_centre != 0 or cost_angle != 0 or cost_dim != 0, "all costs cant be 0"
 

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        #print("matcher")

        #print(outputs["pred_logits"].shape[:2]) ( Batch size * number of queries)
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 8]
        out_angle = outputs["pred_angles"].flatten(0, 1)

        #print(out_bbox.shape) #( Batch size * number of queries, size of output)
        # Also concat the target labels and boxes

        tgt_ids = torch.cat([v["classes"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        tgt_angles = torch.cat([v["angles"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        """
        print( outputs["pred_logits"].flatten(0, 1) )
        print( outputs["pred_logits"].flatten(0, 1).softmax(-1) )
        print(out_prob )
        print(cost_class.shape)
        print(cost_class)
        """

        # Compute the L1 cost between boxes
        # spliting into different inputs and output
        #print(tgt_bbox[:, :2 ]) 
        #print(tgt_bbox[:, 2:4 ] ) 
        #print(tgt_bbox[:, -1: ] ) 

        cost_centre = torch.cdist(out_bbox[:, :2 ].double(), tgt_bbox[:, :2 ].double(), p=1)
        cost_dim = torch.cdist(out_bbox[:, 2:4 ].double(), tgt_bbox[:, 2:4 ].double(), p=1)
        
        
        #print( out_angle.shape )
        #print( tgt_angles.shape )

        cost_angle = torch.cdist(out_angle.double(), tgt_angles.double(), p=1)

        #print(cost_bbox.shape)
        # Compute the giou cost betwen boxes
        #print("Matcher")
        #cost_giou = -iou_inclined_boxes(out_bbox, tgt_bbox)

        # Final cost matrix
        #C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = self.cost_centre * cost_centre +  cost_angle * self.cost_angle + self.cost_dim * cost_dim + self.cost_class * cost_class
        C = C.view(bs, num_queries, -1).cpu()

        """
        for v in targets:
            print("testing sizes")
            print(v)
            print(len(v["boxes"]))
        """

        sizes = [len(v["boxes"]) for v in targets]
        #indices = [print(c[i].size()) for i, c in enumerate(C.split(sizes, -1))]
        #indices = [print( linear_sum_assignment(c[i]) ) for i, c in enumerate(C.split(sizes, -1))]

        
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

        #print(C.size())    # Batch size , number of queries , total number of targets) 
        #print(sizes)
        #print(indices)
        #print([(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices])

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    #return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)
    return HungarianMatcher()
