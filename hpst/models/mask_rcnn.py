from hpst.utils.options import Options
import hpst.layers.heterogenous_point_set_attention as psa
from typing import Tuple
from torch import Tensor, nn
import torch
import torchvision
    
class MaskRCNN(nn.Module):
    def __init__(self, num_classes):
        super(MaskRCNN, self).__init__()
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
            num_classes=num_classes,
            min_size=80,
            max_size=90,
            rpn_batch_size_per_image=128,
            box_detections_per_img=50,
            box_batch_size_per_image=256,
            rpn_pre_nms_top_n_train=200,
            rpn_pre_nms_top_n_test=400,
            rpn_post_nms_top_n_train=200,
            rpn_post_nms_top_n_test=400,
        )

        # replace the first layer of the resnet with a 1 channel conv2d
        #self.model.backbone.body.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        #print(self.model.backbone)
        #asdf
        
    def forward(self, images, targets=None):
        return self.model(images, targets=targets)