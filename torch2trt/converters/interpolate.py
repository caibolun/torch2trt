import torch.nn.functional as F
import torch.nn as nn
from torch2trt.torch2trt import *                                 
from torch2trt.module_test import add_module_test
import collections

                                                  
@tensorrt_converter('torch.nn.functional.interpolate')
@tensorrt_converter('torch.nn.functional.upsample')
def convert_interpolate(ctx):
    #parse args                     
    input = get_arg(ctx, 'input', pos=0, default=None) 
    size = get_arg(ctx, 'size', pos=1, default=None)
    scale_factor = get_arg(ctx, 'scale_factor', pos=2, default=None)
    mode = get_arg(ctx, 'mode', pos=3, default='nearest')
    align_corners = get_arg(ctx, 'align_corners', pos=4, default=None)

    input_dim = input.dim() - 2
    
    input_trt = trt_(ctx.network, input)
    output = ctx.method_return
    layer = ctx.network.add_resize(input=input_trt)
    
    scales = scale_factor
    if scales != None:
        if isinstance(scales, collections.Sequence):
            shape = input.size()[-len(scales):]
            shape = [int(s * r) for s, r in zip(shape, scales)]
            shape  = [input.size(1)] + list(shape)
        else:
            shape = input.size()[2:]
            shape = [int(scales * s) for s in shape]
            shape = [input.size(1)] + shape
        layer.shape = shape

        # if not isinstance(scales, collections.Sequence):
        #     scales = []
        #     for i in range(input_dim)
        #     scales = [scales] * input_dim
        # shape = [input.size(1)] + [shape] * input_dim
        # print(scales)
        # layer.scales = [1] + list(scales)

    shape = size
    if shape != None:
        if isinstance(shape, collections.Sequence):
           shape  = [input.size(1)] + list(shape)
        else:
            shape = [input.size(1)] + [shape] * input_dim
        layer.shape = shape

    print(layer.shape)

    resize_mode = mode
    if resize_mode.lower() in ["linear","bilinear","trilinear"]:
        layer.resize_mode = trt.ResizeMode.LINEAR
    else:
        layer.resize_mode=trt.ResizeMode.NEAREST

    if align_corners != None:
        layer.align_corners = align_corners

    output._trt = layer.get_output(0)


class Interpolate(torch.nn.Module):
    def __init__(self, size, mode, align_corners):
        super(Interpolate, self).__init__()
        self.size = size
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(x, self.size, mode=self.mode, align_corners=self.align_corners)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 112, 112)])
def test_interpolate_nearest():
    return Interpolate((224, 224), 'nearest', None)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 112, 112)])
def test_interpolate_bilinear():
    return Interpolate((224, 224), 'bilinear', True)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 112, 112)])
def test_upsample_scale_factor2():
    return nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

@add_module_test(torch.float32, torch.device('cuda'), [(1,2,12,12)])
def test_nearest_mode():
    return torch.nn.Upsample(scale_factor=2, mode="nearest")

@add_module_test(torch.float32, torch.device('cuda'), [(1,4,12,12)])
def test_bilinear_mode():
    return torch.nn.Upsample(scale_factor=3, mode="bilinear", align_corners=True)

@add_module_test(torch.float32, torch.device('cuda'), [(1,3,12,12)])
def test_align_corner():
    return torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

@add_module_test(torch.float32, torch.device('cuda'), [(1,5,13,13)])
def test_bilinear_mode_odd_input_shape():
    return torch.nn.Upsample(scale_factor=2,mode="bilinear", align_corners=True)

@add_module_test(torch.float32, torch.device('cuda'), [(1,4,12,12)])
def test_size_parameter():
    return torch.nn.Upsample(size=3,mode="nearest")

@add_module_test(torch.float32, torch.device('cuda'), [(1,3,13,13)])
@add_module_test(torch.float32, torch.device('cuda'), [(1,3,1,1)])
def test_size_parameter_odd_input():
    return torch.nn.Upsample(size=[6,3],mode="nearest")


@add_module_test(torch.float32, torch.device('cuda'), [(1,4,6,6,6)])
def test_nearest_mode_3d():
    return torch.nn.Upsample(scale_factor=2, mode="nearest")

@add_module_test(torch.float32, torch.device('cuda'), [(1,3,5,5,5)])
def test_bilinear_mode_3d():
    return torch.nn.Upsample(scale_factor=3, mode="trilinear", align_corners=True)

@add_module_test(torch.float32, torch.device('cuda'), [(1,4,8,8,8)])
def test_align_corner_3d():
    return torch.nn.Upsample(scale_factor=4, mode="trilinear", align_corners=True)

@add_module_test(torch.float32, torch.device('cuda'), [(1,6,7,7,7)])
@add_module_test(torch.float32, torch.device('cuda'), [(1,3,2,4,4)])
@add_module_test(torch.float32, torch.device('cuda'), [(1,3,1,1,1)])
def test_bilinear_mode_odd_input_shape_3d():
    return torch.nn.Upsample(scale_factor=2, mode="trilinear",align_corners=True)

@add_module_test(torch.float32, torch.device('cuda'), [(1,1,12,12,12)])
def test_size_parameter_3d():
    return torch.nn.Upsample(size=3,mode="trilinear", align_corners=True)

@add_module_test(torch.float32, torch.device('cuda'), [(1,3,7,9,5)])
@add_module_test(torch.float32, torch.device('cuda'), [(1,4,3,5,1)])
def test_size_parameter_odd_input_3d():
    return torch.nn.Upsample(size=[11,14,17],mode="trilinear", align_corners=True)
