#include "common.h"

using namespace std;

int main(){
    const char * shm_name = "shared_memory1";
    const char* sem_name = "my_semaphore1";
    int clip_nums = 2;
    size_t frame_size = clip_nums * 3 * 112 * 112 * sizeof(float); 
    
    int shm_fd = shm_open(shm_name, O_RDONLY, 0666);
    if (shm_fd == -1){
        cerr<<"open shared file failed!"<<endl;
        return 0;
    }
    std::cout<<"open shared file successful!!"<<endl;

    sem_t* semaphore = sem_open(sem_name, 0);
    if (semaphore == SEM_FAILED) {
        std::cerr << "Failed to open semaphore!" << std::endl;
        return 0;
    }
    std::cout<<"open semaphore"<<semaphore<<"successful!"<<endl;

    //ONNX2TensorRT("../model/gesture_recognition_model.onnx","../model/gesture_recognition_model.engine");   //onnx转tensorrt
    std::vector<cv::Mat> images(16);
    //===============打开摄像头并且显示======================//
    VideoProcessor processor;  
    TensorRT HandGesture_detector(2, 2);    //手势检测器 （输入2帧，输出为2个类别）
    TensorRT HandGesture_classify(16, 7);    //手势分类器 （输入16帧，输出为7个类别）
   // processor.toggleDisplay();
    //processor.start(); // 开始捕获和显示
    HandGesture_detector.TensorRT_Construct("../model/gesture_recognition_model.engine");
    HandGesture_classify.TensorRT_Construct("../model/gesture_7classification_model.engine");
    while(true){
        images = processor.GetFramesFromShm(semaphore, shm_fd, frame_size);
        //processor.save_images(images);
        HandGesture_detector.TensorRT_Inference(images);
    }
   

    return 0;
}
