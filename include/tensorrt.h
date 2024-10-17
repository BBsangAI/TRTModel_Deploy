#ifndef TENSORRT_H
#define TENSORRT_H

#include "NvInfer.h"

using namespace nvinfer1;

class Logger : public ILogger           
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};

void ONNX2TensorRT(std::string onnx_file_path, std::string engine_file_path);
int TensorRT_Inference(std::vector<cv::Mat> inputs, std::string engine_path);
#endif