#include "common.h"

using namespace std;

int main(){

    //ONNX2TensorRT("../model/gesture_7classification_model.onnx","../model/gesture_7classification_model.engine");
    
     //=============测试opencv库===========================//
    vector <cv::Mat> images (32);
    if(images.empty()){
        std:cout<<"Failed"<<endl;
        return -1;
    }
    cout<<"HELLO OPENCV C++"<<endl;
    //===============打开摄像头并且显示======================//
    VideoProcessor processor;
   // processor.toggleDisplay();
    //processor.start(); // 开始捕获和显示
    images = processor.GetFramesFromShm("shared_memory1");
    TensorRT_Inference(images, "../model/gesture_7classification_model.engine");
    std::cout << "Press Enter to stop the video..." << std::endl;
    std::cin.get(); // 按Enter停止程序
    //processor.stop(); // 停止捕获和显示
    return 0;
}
