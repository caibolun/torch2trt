#!/usr/bin/env python
# coding=utf-8
'''
@Author: ArlenCai
@Date: 2020-06-17 18:01:23
@LastEditTime: 2020-06-19 11:46:18
'''
from collections import OrderedDict
from torch import nn
from torch.nn import functional as F
import torch
import torchvision
from torchvision.models.segmentation._utils import _SimpleSegmentationModel
from torch2trt.module_test import add_module_test

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model
    def forward(self, x):
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.model.backbone(x)
        x = features["out"]
        x = self.model.classifier(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)
        return x


@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True)
def fcn_resnet50():
    bb = torchvision.models.segmentation.fcn_resnet50(pretrained=False, pretrained_backbone=False)
    model = ModelWrapper(bb)
    return model


@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True)
def fcn_resnet101():
    bb = torchvision.models.segmentation.fcn_resnet101(pretrained=False, pretrained_backbone=False)
    model = ModelWrapper(bb)
    return model