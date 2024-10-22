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


class TensorRT{
    
public:
    TensorRT(int inputs_samples, int outputs_classes) : samples(inputs_samples), classes(outputs_classes) {
        totalSize = samples * 3 * 112 * 112 * sizeof(float);
    }
   
    void ONNX2TensorRT(std::string onnx_file_path, std::string engine_file_path);
    int TensorRT_Construct(std::string engine_path);
    int TensorRT_Inference(std::vector<cv::Mat> inputs);

    ~TensorRT(){
        if (buffers[inputIndex]) {
            cudaFree(buffers[inputIndex]);
            }
        if (buffers[outputIndex]) {
            cudaFree(buffers[outputIndex]);
            }
        std::cout << "Freed GPU buffers" << std::endl;
        }

private:
    int samples = 0;      // 输入图片的时间维度数
    int classes = 0;      // 输出总类数
    size_t totalSize = 0; // 存储输入数据的总大小
    const int inputIndex = 0;  // 输入绑定索引
    const int outputIndex = 1; // 输出绑定索引
    void* buffers[2]= {nullptr, nullptr};          // 存储输入和输出的 GPU 内存指针
    nvinfer1::IExecutionContext* context = nullptr;  // TensorRT 执行上下文
};


#endif