from hpst.utils.options import Options
import hpst.layers.heterogenous_point_set_attention as psa
from typing import Tuple
from torch import Tensor, nn
import torch
import torchvision
from pytorch3dunet.unet3d.model import UNet3D
    
class Unet3d(nn.Module):
    def __init__(self, num_classes, num_objects):
        super(Unet3d, self).__init__()
        self.num_classes = num_classes
        self.num_objects = num_objects
        #self.model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(num_classes=num_classes)
        self.model = UNet3D(in_channels=1, out_channels=num_classes + num_objects)
        # replace the first layer of the resnet with a 1 channel conv2d
        #self.model.backbone.body.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        #print(self.model.backbone)
        #asdf
        
    def forward(self, images):
        #print(images.shape)
        output = self.model(images)

        return output[:,:self.num_classes], output[:,self.num_classes:]
        #dasf
        #return self.model(images, targets=targets)