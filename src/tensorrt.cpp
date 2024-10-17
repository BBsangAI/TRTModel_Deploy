#include "common.h"
#include <cuda_runtime.h>

using namespace nvinfer1;

Logger logger;
// 模型导入阶段：创建builder--> 创建网络定义-->创建ONNX解析器-->使用解析器加载ONNX网路模型参数到定义好的网络
// 构建引擎阶段：配置引擎（最大工作空间、模型精度等）-->创建引擎-->保存-->销毁
void ONNX2TensorRT(std::string onnx_file_path, std::string engine_file_path){
    // ---------------------------创建builder---------------------------
    IBuilder *builder = createInferBuilder(logger);
    // 创建网络定义
    const auto explicitBatch = 1U << static_cast<uint32_t>
        (NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);  // 显性批处理标志
    INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
    // 创建ONNX解析器
    nvonnxparser::IParser * parser = nvonnxparser::createParser(*network, logger);
    // 解析ONNX文件
    parser->parseFromFile(onnx_file_path.c_str(), 
    static_cast<int32_t>(ILogger::Severity::kWARNING));
    for (int32_t i = 0; i < parser->getNbErrors(); ++i)
    {
    std::cout << parser->getError(i)->desc() << std::endl;
    }
    // 加载成功
    printf("tensorRT load mask onnx model successfully!!!...\n");

    // ---------------------------构建推理引擎---------------------------
    IBuilderConfig* config = builder->createBuilderConfig();
    // 设定最大工作空间的大小
    config->setMaxWorkspaceSize(16 * (1<<20));
    // 设置模型输出的精度（FP16）
    config->setFlag(BuilderFlag::kFP16);
    // 创建推理引擎
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

    // ---------------------------保存---------------------------
    std::cout<<"try to save engine file..."<<std::endl;
    std::ofstream file_ptr(engine_file_path, std::ios::binary);
	if (!file_ptr) {
		std::cerr << "could not open plan output file" << std::endl;
		return;
	}
	// 将模型转化为文件流数据
	IHostMemory* model_stream = engine->serialize();
	// 将文件保存到本地
	file_ptr.write(reinterpret_cast<const char*>(model_stream->data()), model_stream->size());

	// ---------------------------销毁创建的对象---------------------------
	model_stream->destroy();
	engine->destroy();
	network->destroy();
	parser->destroy();
	std::cout << "convert onnx model to TensorRT engine model successfully!" << std::endl;
}

int TensorRT_Inference(std::vector<cv::Mat> inputs, std::string engine_path){
    /*******************读取engine模型文件 *******************/
    /*******************读取engine模型文件 *******************/
    size_t size = 0;
    char* trtModelStream{nullptr};
    std::ifstream file(engine_path, std::ios::binary);
    if(!file.good()){
        std::cerr<<"open engine file"<<engine_path<< "failed!"<<std::endl;
        return -1;
    }
    file.seekg(0, file.end); //将都指针从文件末尾移动0个位置
    size = file.tellg();   //返回当前读指针位置，可作为二进制文件大小
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();
    std::cout<<"read engine《"<<engine_path<<"》successful!"<<endl;
     /******************* ******************* *******************/
     /******************* ******************* *******************/
     /******************* 使用模型进行推理 *******************/
     /******************* 使用模型进行推理 *******************/
    IRuntime* runtime = createInferRuntime(logger);    //创建运行时
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);  //反序列化引擎
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;  //清理模型流内存
    assert(engine->getNbBindings() == 2);
    // int numBindings = engine->getNbBindings();  // 获取绑定的数量
    // for (int i = 0; i < numBindings; ++i) {
    // const char* bindingName = engine->getBindingName(i);  // 获取每个绑定的名称
    // std::cout << "Binding " << i << ": " << bindingName << std::endl;
    const int inputIndex = engine->getBindingIndex("input.1");  // 使用实际的输入绑定名称
    const int outputIndex = engine->getBindingIndex("978");     // 使用实际的输出绑定名称
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    void* buffers[2];
    cudaMalloc(&buffers[inputIndex], 1 * 32 * 3 * 112 * 112 * sizeof(float));
    cudaMalloc(&buffers[outputIndex], 1 * 7 * sizeof(float));
    std::vector<float> input_data(32 * 3 * 112 * 112);
    for(int i=0;i<inputs.size(); ++i){
        cv::Mat resized_image;
        inputs[i].convertTo(resized_image, CV_32FC3); // 转换到 float
        std::memcpy(&input_data[i * 3 * 112 * 112], resized_image.data, 3 * 112 * 112 * sizeof(float));
    }
   nvinfer1::Dims dims;
    dims.nbDims = 5; // 4D 数据
    dims.d[0] = 1;   // batch size
    dims.d[1] = 3;  // 通道数
    dims.d[2] = 16;  // 深度
    dims.d[3] = 112; // 高度
    dims.d[4] = 112; // 宽度
    context->setBindingDimensions(inputIndex, dims);

    // 使用 enqueueV2 进行推理
    context->enqueueV2(buffers, 0, nullptr);

    std::vector<float> output_data(7); // 假设输出有 7 个元素
    cudaMemcpyAsync(output_data.data(), buffers[outputIndex], 1 * 7 * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(buffers[inputIndex]);
    cudaFree(buffers[outputIndex]);

   for (size_t i = 0; i < output_data.size(); ++i) {
        std::cout << "Output[" << i << "] = " << output_data[i] << std::endl;
    }
    return 1;

}

