###
 # @Author: ArlenCai
 # @Date: 2020-06-17 18:01:23
 # @LastEditTime: 2020-06-23 20:43:56
### 
#!/bin/bash

OUTPUT_FILE=$1

touch $OUTPUT_FILE

echo "| Name | Data Type | Input Shapes | torch2trt kwargs | Max Error | Throughput (PyTorch) | Throughput (TensorRT) | Latency (PyTorch) | Latency (TensorRT) |" >> $OUTPUT_FILE
echo "|------|-----------|--------------|------------------|-----------|----------------------|-----------------------|-------------------|--------------------|" >> $OUTPUT_FILE

python3 -m torch2trt.test -o $OUTPUT_FILE --name alexnet --include=torch2trt.tests.torchvision.classification
python3 -m torch2trt.test -o $OUTPUT_FILE --name squeezenet1_0 --include=torch2trt.tests.torchvision.classification
python3 -m torch2trt.test -o $OUTPUT_FILE --name squeezenet1_1 --include=torch2trt.tests.torchvision.classification
python3 -m torch2trt.test -o $OUTPUT_FILE --name resnet18 --include=torch2trt.tests.torchvision.classification
python3 -m torch2trt.test -o $OUTPUT_FILE --name resnet34 --include=torch2trt.tests.torchvision.classification
python3 -m torch2trt.test -o $OUTPUT_FILE --name resnet50 --include=torch2trt.tests.torchvision.classification
python3 -m torch2trt.test -o $OUTPUT_FILE --name resnet101 --include=torch2trt.tests.torchvision.classification
python3 -m torch2trt.test -o $OUTPUT_FILE --name resnet152 --include=torch2trt.tests.torchvision.classification
python3 -m torch2trt.test -o $OUTPUT_FILE --name densenet121 --include=torch2trt.tests.torchvision.classification
python3 -m torch2trt.test -o $OUTPUT_FILE --name densenet169 --include=torch2trt.tests.torchvision.classification
python3 -m torch2trt.test -o $OUTPUT_FILE --name densenet201 --include=torch2trt.tests.torchvision.classification
python3 -m torch2trt.test -o $OUTPUT_FILE --name densenet161 --include=torch2trt.tests.torchvision.classification
python3 -m torch2trt.test -o $OUTPUT_FILE --name vgg11$ --include=torch2trt.tests.torchvision.classification
python3 -m torch2trt.test -o $OUTPUT_FILE --name vgg13$ --include=torch2trt.tests.torchvision.classification
python3 -m torch2trt.test -o $OUTPUT_FILE --name vgg16$ --include=torch2trt.tests.torchvision.classification
python3 -m torch2trt.test -o $OUTPUT_FILE --name vgg19$ --include=torch2trt.tests.torchvision.classification
python3 -m torch2trt.test -o $OUTPUT_FILE --name vgg11_bn --include=torch2trt.tests.torchvision.classification
python3 -m torch2trt.test -o $OUTPUT_FILE --name vgg13_bn --include=torch2trt.tests.torchvision.classification
python3 -m torch2trt.test -o $OUTPUT_FILE --name vgg16_bn --include=torch2trt.tests.torchvision.classification
python3 -m torch2trt.test -o $OUTPUT_FILE --name vgg19_bn --include=torch2trt.tests.torchvision.classification
python3 -m torch2trt.test -o $OUTPUT_FILE --name mobilenet_v2 --include=torch2trt.tests.torchvision.classification

python3 -m torch2trt.test -o $OUTPUT_FILE --name fcn_resnet50 --include=torch2trt.tests.torchvision.segmentation
python3 -m torch2trt.test -o $OUTPUT_FILE --name fcn_resnet101 --include=torch2trt.tests.torchvision.segmentation

python3 -m torch2trt.test -o $OUTPUT_FILE --name retinanet_resnet18 --include=torch2trt.tests.torchvision.detection
python3 -m torch2trt.test -o $OUTPUT_FILE --name retinanet_resnet34 --include=torch2trt.tests.torchvision.detection
python3 -m torch2trt.test -o $OUTPUT_FILE --name retinanet_resnet50 --include=torch2trt.tests.torchvision.detection
python3 -m torch2trt.test -o $OUTPUT_FILE --name retinanet_resnet101 --include=torch2trt.tests.torchvision.detection
python3 -m torch2trt.test -o $OUTPUT_FILE --name retinanet_resnet152 --include=torch2trt.tests.torchvision.detection
