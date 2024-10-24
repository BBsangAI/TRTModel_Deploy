#include "common.h"
#include "Params.h"
#include <csignal>
using namespace std;

std::condition_variable convar;
std::mutex mtx;
bool handGesture_detected = false; // 标志是否检测到手势
bool classification_done = true;   // 标志分类是否完成
int pipe_fd[2]; 

void handGestureDetection(InferenceParams& params){      // 手势检测线程函数
    int clip_nums = 2;
    std::vector<cv::Mat> images(2);
    int result = 0;
    const char * signal_shm_name = "signal_shared_memory1";       // 共享内存名（dataset.py中创建）
    int signal_shm_fd = shm_open(signal_shm_name, O_RDWR, 0666);
    if(signal_shm_fd != -1)
        cout<<"signal_shared_memory1 open successful!"<<endl;
    void* signal_ptr = mmap(nullptr, sizeof(bool), PROT_READ | PROT_WRITE, MAP_SHARED, signal_shm_fd, 0);   //映射共享内存
    memcpy(signal_ptr, &handGesture_detected, sizeof(handGesture_detected));   
    while(true){
        images = params.Video_Processor.GetFramesFromShm(params.frame_semaphore, params.shm_fd, clip_nums);
        //processor.save_images(images);
        result = params.Engine.TensorRT_Inference(images); 
        //std::cout <<classes_map["IsHandGesture_class"][result] << std::endl;
        if(result == 1){ 
            cout<<"detected"<<endl;
            std::unique_lock<std::mutex> lock(mtx);
            handGesture_detected = true;
            classification_done = false;       
            memcpy(signal_ptr, &handGesture_detected, sizeof(handGesture_detected));   
            convar.notify_all();
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            convar.wait(lock, [] { return classification_done; });
        }   
    }
}

void handGestureClassficition(InferenceParams& params){      // 手势分类线程函数
    int clip_nums = 16;
    std::vector<cv::Mat> images(16);
    int result = 0;
    while(true){
        std::unique_lock<std::mutex> lock(mtx);
        convar.wait(lock, [] { return handGesture_detected; }); // 等待手势检测线程通知
        images = params.Video_Processor.GetFramesFromShm(params.frame_semaphore, params.shm_fd, clip_nums);
        params.Video_Processor.save_images(images);
       
        result = params.Engine.TensorRT_Inference(images);
        std::cout << "target = " << classes_map["WhHandGesture_class"][result] << std::endl;

        handGesture_detected = false;  // 重置检测标志
        classification_done = true;    // 分类完成标志
        lock.unlock();
        convar.notify_all();  // 通知手势检测线程继续
    }
}



int main(){
     //-----------初始化定义-------------------------//
    const char * shm_name1 = "shared_memory1";       // 共享内存名（dataset.py中创建）
    const char * shm_name2 = "shared_memory2";       // 共享内存名（dataset.py中创建）
    const char* frame_semname = "frame_semaphore";  //帧准备信号量（dataset.py中创建）
    const char* init_semname = "init_semaphore";    //初始化准备信号量（dataset.py中创建）
   
    std::string HandGesture_detector_model_path = "../model/gesture_recognition_model.engine";
    std::string HandGesture_classify_model_path = "../model/gesture_7classification_model.engine";
    int shm_fd1 = 0, shm_fd2 = 0;
    Params_init();
    //-----------尝试打开共享内存-------------------------//
    while (true) {
        shm_fd1 = shm_open(shm_name1, O_RDONLY, 0666);
        shm_fd2 = shm_open(shm_name2, O_RDONLY, 0666);
        if (shm_fd1 != -1 && shm_fd2 != -1) {
            std::cout << "open shared file successful!!" << std::endl;
            break; // 成功打开，退出循环
        } else {
            std::cerr << "open shared file failed, retrying..." << std::endl;
            sleep(1); // 等待1秒后再次尝试
        }
    }

    //-----------尝试打开信号量-------------------------//
    sem_t* frame_semaphore = sem_open(frame_semname, 0);
    sem_t* init_semaphore = sem_open(init_semname, 0);
    if (frame_semaphore == SEM_FAILED || init_semaphore == SEM_FAILED) {
        std::cerr << "Failed to open semaphore!" << std::endl;
        return 0;
    }

    //===============加载模型、打开各个检测线程======================//
    
    VideoProcessor processor;  
    TensorRT HandGesture_detector(2, 2);     //手势检测器 （输入2帧，输出为2个类别）
    TensorRT HandGesture_classify(16, 7);    //手势分类器 （输入16帧，输出为7个类别）

    if (HandGesture_detector_model_path.substr(HandGesture_detector_model_path.size() - 5) == ".onnx")
        HandGesture_detector.ONNX2TensorRT(HandGesture_detector_model_path,
                                           HandGesture_detector_model_path.replace(HandGesture_detector_model_path.size() - 5, 5, ".engine"));   //onnx转tensorrt
    if(HandGesture_classify_model_path.substr(HandGesture_classify_model_path.size() - 5) == ".onnx")
        HandGesture_classify.ONNX2TensorRT(HandGesture_classify_model_path,
                                           HandGesture_classify_model_path.replace(HandGesture_classify_model_path.size() - 5, 5, ".engine"));   //onnx转tensorrt

    HandGesture_detector.TensorRT_Construct(HandGesture_detector_model_path);
    HandGesture_classify.TensorRT_Construct(HandGesture_classify_model_path);
    sem_post(init_semaphore);      // 通知python进程 初始化完成

    InferenceParams HandGesture_detector_Params{HandGesture_detector, processor, frame_semaphore, shm_fd1};    // 手势检测器参数
    InferenceParams HandGesture_classify_Params{HandGesture_classify, processor, frame_semaphore, shm_fd2};    // 手势分类器参数

    std::thread HandGestureDetect_thread(handGestureDetection, std::ref(HandGesture_detector_Params));
    std::thread HandGestureClass_thread(handGestureClassficition, std::ref(HandGesture_classify_Params)); 

    HandGestureDetect_thread.join();
    HandGestureClass_thread.join();

    return 0;
}
