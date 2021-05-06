// Copyright(c) Microsoft Corporation.All rights reserved.
// Licensed under the MIT License.
//

#include <assert.h>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include<opencv.hpp>
#include <ctime>

struct BOX
{
    float weight;
    float area;
    std::vector<float> bbound;

};

inline bool CMP_WEIGHT(BOX b1, BOX b2) 
{
    return b1.weight > b2.weight;
}



#ifdef __cplusplus
extern "C" {
#endif

    /**
     * \param device_id cuda device id, starts from zero.
     */
    ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_CUDA, _In_ OrtSessionOptions* options, int device_id);

#ifdef __cplusplus
}
#endif

int main(int argc, char* argv[]) {
    
    
    clock_t start, end;
    //https://leimao.github.io/blog/ONNX-Runtime-CPP-Inference/
  
    //*************************************************************************
    // initialize  enviroment...one enviroment per process
    // enviroment maintains thread pools and other state info
    
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");

    // initialize session options if needed
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(0);

    // If onnxruntime.dll is built with CUDA enabled, we can uncomment out this line to use CUDA for this
    // session (we also need to include cuda_provider_factory.h above which defines it)
   //#include "cuda_provider_factory.h"
   //OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 1);
   
    // Sets graph optimization level
    // Available levels are
    // ORT_DISABLE_ALL -> To disable all optimizations
    // ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node removals)
    // ORT_ENABLE_EXTENDED -> To enable extended optimizations (Includes level 1 + more complex optimizations like node fusions)
    // ORT_ENABLE_ALL -> To Enable All possible opitmizations
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    //*************************************************************************
    // create session and load model into memory
    // using squeezenet version 1.3
    // URL = https://github.com/onnx/models/tree/master/squeezenet
#ifdef _WIN32
    const wchar_t* model_path = L"../yolov4.onnx";
#else
    const char* model_path = "squeezenet.onnx";
#endif

    printf("Using Onnxruntime C++ API\n");
    Ort::Session session(env, model_path, session_options);

    //*************************************************************************
    // print model input layer (node names, types, shape etc.)
    Ort::AllocatorWithDefaultOptions allocator;

    // print number of model input nodes
    size_t num_input_nodes = session.GetInputCount();
    std::vector<const char*> input_node_names(num_input_nodes);
    std::vector<int64_t> input_node_dims;  // simplify... this model has only 1 input node {1, 3, 224, 224}.
                                           // Otherwise need vector<vector<>>

    printf("Number of inputs = %zu\n", num_input_nodes);
    
    // iterate over all input nodes
    for (int i = 0; i < num_input_nodes; i++) {
        // print input node names
        char* input_name = session.GetInputName(i, allocator);
        printf("Input %d : name=%s\n", i, input_name);
        input_node_names[i] = input_name;

        // print input node types
        Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        
        ONNXTensorElementDataType type = tensor_info.GetElementType();
        printf("Input %d : type=%d\n", i, type);

        // print input shapes/dims
        input_node_dims = tensor_info.GetShape();
        printf("Input %d : num_dims=%zu\n", i, input_node_dims.size());
        for (int j = 0; j < input_node_dims.size(); j++)
            printf("Input %d : dim %d=%jd\n", i, j, input_node_dims[j]);
    }

    // Results should be...
    // Number of inputs = 1
    // Input 0 : name = data_0
    // Input 0 : type = 1
    // Input 0 : num_dims = 4
    // Input 0 : dim 0 = 1
    // Input 0 : dim 1 = 3
    // Input 0 : dim 2 = 224
    // Input 0 : dim 3 = 224

   




    //*************************************************************************
    // Similar operations to get output node information.
    // Use OrtSessionGetOutputCount(), OrtSessionGetOutputName()
    // OrtSessionGetOutputTypeInfo() as shown above.

    //*************************************************************************
    // Score the model using sample data, and inspect values

    size_t input_tensor_size = 416 * 416 * 3;  // simplify ... using known dim values to calculate size
                                               // use OrtGetTensorShapeElementCount() to get official size!

    std::vector<float> input_tensor_values(input_tensor_size);
    std::vector<const char*> output_node_names = { "boxes","confs" };

    // initialize input data with values in [0.0, 1.0]
    for (unsigned int i = 0; i < input_tensor_size; i++)
        input_tensor_values[i] = (float)i / (input_tensor_size + 1);

    const cv::String imageFilepath = "../1.jpg";

    cv::Mat imageBGR = cv::imread(imageFilepath, cv::ImreadModes::IMREAD_COLOR);
    cv::Mat resizedImageBGR, resizedImageRGB, resizedImage, preprocessedImage;
    cv::resize(imageBGR, resizedImageBGR,
        cv::Size(416, 416),
        cv::InterpolationFlags::INTER_CUBIC);

    cv::cvtColor(resizedImageBGR, resizedImageRGB,
        cv::ColorConversionCodes::COLOR_BGR2RGB);
    resizedImageRGB.convertTo(resizedImage, CV_32F, 1.0 / 255.0);
    
    cv::Mat channels[3];
    cv::split(resizedImage, channels);
    // Normalization per channel
    // Normalization parameters obtained from
    // https://github.com/onnx/models/tree/master/vision/classification/squeezenet
    //channels[0] = (channels[0]) / 255.0;
    //channels[1] = (channels[1]) / 255.0;
    //channels[2] = (channels[2]) / 255.0;

    cv::merge(channels, 3, resizedImage);
    // HWC to CHW
    cv::dnn::blobFromImage(resizedImage, preprocessedImage);

    std::vector<float> inputTensorValues(input_tensor_size);
    inputTensorValues.assign(preprocessedImage.begin<float>(),
        preprocessedImage.end<float>());

    // create input tensor object from data values
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, inputTensorValues.data(), input_tensor_size, input_node_dims.data(), 4);
    assert(input_tensor.IsTensor());
    start = clock();
    // score model & input tensor, get back output tensor
    auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 2);
    assert(output_tensors.size() == 2 && output_tensors.front().IsTensor());
    end = clock();
    // Get pointer to output tensor float values
    float* boxarr = output_tensors.front().GetTensorMutableData<float>();
    float* confarr = output_tensors.back().GetTensorMutableData<float>();


   /* int myarray[5] = { 1,3,5,7,9 };
    const std::vector<int> myvector(myarray, myarray + 5);

    
    const std::vector<int64_t> index(myarray, myarray + 5);
    
    const std::vector<int64_t>& location = index;
    std::vector<Ort::Value> x = output_tensors[0].At(&index);*/

    size_t boxes_size = 1 * 10647 * 1 * 4;
    size_t conf_size = 1 * 10647 * 6;

    output_tensors.front().GetTensorTypeAndShapeInfo();
    //assert(abs(floatarr[0] - 0.000045) < 1e-6);
    std::vector<Ort::Value> boxesTensorValues ;
    std::vector<Ort::Value> confTensorValues;

    //std::vector<Ort::Value> boxesTensorValues = output_tensors.front();
    
    std::vector<float> boxesTemp(boxes_size);
    std::vector<float> confsTemp(conf_size);

    boxesTemp.assign(boxarr,boxarr+ boxes_size);
    confsTemp.assign(confarr,confarr + conf_size);

    std::vector<int64_t> boxdims;

    // print input node types
    Ort::TypeInfo type_info = session.GetOutputTypeInfo(0);
    auto box_info = type_info.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType type = box_info.GetElementType();
    printf("Input %d : type=%d\n", 0, type);

    // print input shapes/dims
    boxdims = box_info.GetShape();
   

    std::vector<int64_t> confvdims;

    Ort::TypeInfo type_info1 = session.GetOutputTypeInfo(1);
    auto confs_info = type_info1.GetTensorTypeAndShapeInfo();

    type = confs_info.GetElementType();
    printf("Input %d : type=%d\n", 1, type);

    // print input shapes/dims
    confvdims = confs_info.GetShape();
   
    //load input
    boxesTensorValues.push_back(Ort::Value::CreateTensor<float>(
        memory_info, boxesTemp.data(), boxes_size,
        boxdims.data(), boxdims.size()));
    
    confTensorValues.push_back(Ort::Value::CreateTensor<float>(
        memory_info, confsTemp.data(), conf_size,
        confvdims.data(), confvdims.size()));

   
   int myarray[5] = {0,0};
   const std::vector<int64_t> index(myarray, myarray + 2);

   const std::vector<int64_t, std::allocator<int64_t>>& location1 = index;

    //nums classes
   int num_class = 6;

    //找每组confs最大值及其对应的id
   std::vector<int> grid_class;
   std::vector<float> grid_class_weight;
   std::vector<std::vector<float>> grid_max_bound;

   float thresh = 0.4;

   for(int i=0;i< confsTemp.size()/num_class;i++)
   {
       float max_weight=0;
       int max_class=0;
       for (int j = 0; j < num_class; j++) 
       {    
           if (max_weight < confsTemp[i * num_class + j])
           {
               max_weight = confsTemp[i * num_class + j];
               max_class = j;

           }
       }

       if (max_weight > thresh) 
       {
           grid_class_weight.push_back(max_weight);
           grid_class.push_back(max_class);
           std::vector<float> bound;
           for (int j = 0; j < 4; j++)
           {
               bound.push_back(boxesTemp[i * 4 + j]);
           }
           grid_max_bound.push_back(bound);

       }
   }

   //impolement nms
   float nms_thresh = 0.5;
   std::vector<std::vector<float>> result;
   bool min_mode = false;
   

   for (int n = 0; n < num_class; n++)
   {
       std::vector<BOX> Sboxes;

       for (int i = 0; i < grid_class.size(); i++)
       {
           if (grid_class.at(i) == n)
           {
               BOX box = BOX();
               box.weight = grid_class_weight.at(i);
               box.bbound = grid_max_bound.at(i);
               box.area = (grid_max_bound.at(i).at(2) - grid_max_bound.at(i).at(0)) * (grid_max_bound.at(i).at(3) - grid_max_bound.at(i).at(1));
               Sboxes.push_back(box);
           }
       }

      
       if(Sboxes.size() > 1)
       {
   
           std::sort(Sboxes.begin(), Sboxes.end(), CMP_WEIGHT);
           
           for (int i = 0; i < Sboxes.size(); i++)
           {
               for (int j = i + 1; j < Sboxes.size(); j++)
               {
                   if (Sboxes.at(i).weight > 1e-10 && Sboxes.at(j).weight > 1e-10)
                   {
                       float xx1 = std::max(Sboxes.at(j).bbound.at(0), Sboxes.at(i).bbound.at(0));
                       float yy1 = std::max(Sboxes.at(j).bbound.at(1), Sboxes.at(i).bbound.at(1));
                       float xx2 = std::min(Sboxes.at(j).bbound.at(2), Sboxes.at(i).bbound.at(2));
                       float yy2 = std::min(Sboxes.at(j).bbound.at(3), Sboxes.at(i).bbound.at(3));
                       float inter = std::max(0.0f, xx2 - xx1) * std::max(0.0f, yy2 - yy1);
                       float IoU = inter / Sboxes.at(j).area + Sboxes.at(i).area - inter;
                       if (IoU > nms_thresh)
                       {
                           Sboxes.at(j).weight = 0;
                       }
                   }
               }
           }

           for (int i = 0; i < Sboxes.size(); i++)
           {
               if (Sboxes.at(i).weight > 1e-10)
               {
                   std::cout << n << Sboxes.at(i).bbound.at(0) << Sboxes.at(i).bbound.at(1) << Sboxes.at(i).bbound.at(2) << Sboxes.at(i).bbound.at(3) << std::endl;
               }
           }
       }
       else if (Sboxes.size() == 1)
       {
           std::cout << n << Sboxes.at(0).bbound.at(0) << Sboxes.at(0).bbound.at(1) << Sboxes.at(0).bbound.at(2) << Sboxes.at(0).bbound.at(3) << std::endl;
       }
   }
  
   
   std::cout << (double)(end - start) / CLOCKS_PER_SEC << std::endl;
   printf("Done!\n");
   return 0;
}