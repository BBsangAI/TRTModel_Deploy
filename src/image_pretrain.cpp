#include "common.h"
#include <unistd.h> 
#include <semaphore.h>

using namespace std;
//捕获视频流
void VideoProcessor::captureVideo() {
    cv::VideoCapture cap("v4l2src device=/dev/video0 ! image/jpeg, width=800, height=600, framerate=60/1 ! jpegdec ! videoconvert ! appsink", cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video source." << std::endl;
        return;
    }
    while (running) {
        cv::Mat tempframe;
        cap >> tempframe;
        if(!tempframe.empty()){ 
            std::lock_guard<std::mutex> lock(frameMutex);
            frame = tempframe.clone();  // 保存捕获的帧
        }
          
        else{
            std::cerr << "Warning: Empty frame captured." << std::endl;
        }
    }
}

//显示视频
void VideoProcessor::displayVideo() {
    while (showVideo) {
        cv::Mat currentFrame = GetVideoFrames();
        if (!currentFrame.empty()) {
            cv::namedWindow("Display Window", cv::WINDOW_NORMAL);
            cv::imshow("Display Window", currentFrame); // 显示图像
        }
        cv::waitKey(1); // 等待1毫秒
    }
    cv::destroyAllWindows(); // 确保关闭所有窗口
}

std::vector<cv::Mat> VideoProcessor::GetFramesFromShm(const char * shm_name){
    const char * shm = shm_name;
    const size_t frame_size = 32 * 112 * 112 * 3 * sizeof(float); 
    const char* sem_name = "my_semaphore1";
    int shm_fd = shm_open(shm_name, O_RDONLY, 0666);
    
    if (shm_fd == -1){
        cerr<<"open shared file failed!"<<endl;
        return std::vector<cv::Mat>();
    }
    std::cout<<"open shared file successful!!"<<endl;

    sem_t* semaphore = sem_open(sem_name, 0);
    if (semaphore == SEM_FAILED) {
        std::cerr << "Failed to open semaphore!" << std::endl;
        return cv::Mat();
    }
    std::cout<<"open semaphore"<<semaphore<<"successful!"<<endl;
    sem_wait(semaphore);
    // void* mmap(void* addr, size_t length, int prot, int flags, int fd, off_t offset)
    void* ptr = mmap(nullptr, frame_size, PROT_READ, MAP_SHARED, shm_fd, 0);   //映射共享内存
    if(ptr == MAP_FAILED){
        cerr<< "Failed to map shared memory!" <<endl;
        close(shm_fd);
        return std::vector<cv::Mat>();
    }
    std::cout<<"map shared memory successful!"<<endl;
    
    //读取数据到cv::Mat
    std::vector<cv::Mat> images(32);
    for (int i = 0; i < 32; ++i) {
    // 每张图像的指针偏移
    void* image_ptr = static_cast<char*>(ptr) + i * (112 * 112 * 3 * sizeof(float));
    images[i] = cv::Mat(112, 112, CV_16FC3, image_ptr).clone(); // 每张图像
    }
    // 清理
    munmap(ptr, frame_size); // 解除映射
    close(shm_fd);           // 关闭共享内存文件描述符`1
    
    return images;
}