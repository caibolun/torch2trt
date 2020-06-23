#!/usr/bin/env python
# coding=utf-8
'''
@Author: ArlenCai
@Date: 2020-06-17 18:01:23
@LastEditTime: 2020-06-19 13:26:00
'''
import torch
import torchvision
from torch2trt.module_test import add_module_test
from .retinanet import *


class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model
    def forward(self, x):
        return self.model.forward_dummy(x)

@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True)  
def retinanet_resnet18():
    bb = retinanet18(num_classes=80, pretrained=False)
    model = ModelWrapper(bb)
    return model


@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True)
def retinanet_resnet34():
    bb = retinanet34(num_classes=80, pretrained=False)
    model = ModelWrapper(bb)
    return model


@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True)
def retinanet_resnet50():
    bb = retinanet50(num_classes=80, pretrained=False)
    model = ModelWrapper(bb)
    return model


@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True)
def retinanet_resnet101():
    bb = retinanet101(num_classes=80, pretrained=False)
    model = ModelWrapper(bb)
    return model


@add_module_test(torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], fp16_mode=True)
def retinanet_resnet152():
    bb = retinanet152(num_classes=80, pretrained=False)
    model = ModelWrapper(bb)
    return model