## This is a simple demo for using Onnxruntime GPU C++ for yolo darknet detection

# 1. Requirements
Windows Platform
  - Install CUDA+CUDNN（For GPU inference）
  - Visual Studio
  - OpenCV
  - Onnxruntime

# 2. Train
Simply use [darknet](https://github.com/AlexeyAB/darknet) for training

# 3. Convert
There are many convertion approaches for converting a darknet model to onnx model.
In this case, I use [darknet->pytorch->onnx](https://github.com/Tianxiaomo/pytorch-YOLOv4) for converting the model

# 4. Create C++ project
a) on Visual Studio, use config the OpenCV path

b) use Nuget to install Onnxruntme package, in this case, I use [GPU version](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.gpu)

c) create a project that load and postprocess the model output. hint can be found [here](https://github.com/microsoft/onnxruntime/tree/master/samples/c_cxx). for the model structure, use [netron](https://netron.app/) to observe the output.

d) for CPU inferences, simply uncomment the CUDA c part
