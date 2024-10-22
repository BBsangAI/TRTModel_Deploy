#include "common.h"
#include <unistd.h> 


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

// 获取数据
std::vector<cv::Mat> VideoProcessor::GetFramesFromShm(sem_t* semaphore, int shm_fd, const size_t frame_size){
    sem_wait(semaphore);
    // void* mmap(void* addr, size_t length, int prot, int flags, int fd, off_t offset)
    void* ptr = mmap(nullptr, frame_size, PROT_READ, MAP_SHARED, shm_fd, 0);   //映射共享内存
    if(ptr == MAP_FAILED){
        cerr<< "Failed to map shared memory!" <<endl;
        close(shm_fd);
        return std::vector<cv::Mat>();
    }
    
    //读取数据到cv::Mat
    std::vector<cv::Mat> images(16);
    for (int i = 0; i < 16; ++i) {
    // 每张图像的指针偏移
    void* image_ptr = static_cast<char*>(ptr) + i * (3 * 112 * 112 * sizeof(float));  // void*可接受任何类型的指针，static_cast强制类型转换，char*为按字节进行指针运算
    cv::Mat image = cv::Mat(112, 112, CV_32FC3, image_ptr).clone(); // 原始图像 (112, 112, 3)  
    images[i] = image;  // 存储结果
    }
    // 清理
    munmap(ptr, frame_size); // 解除映射
    //close(shm_fd);           // 关闭共享内存文件描述符`1
    
    return images;
}

//保存图片数据到本地
void VideoProcessor::save_images(std::vector<cv::Mat> images){
    std::string filename = "";
    cv::Mat image_uchar;
    for(int i=0;i< images.size();i++){ 
    images[i].convertTo(image_uchar, CV_8UC3, 255.0);
    filename = "../images_test/output_" + std::to_string(i) + ".jpg";
    bool success = cv::imwrite(filename,image_uchar);
    if(success){
        cout<<"图片"<<filename<<"已保存！"<<endl;
     }
     else
      cout<<"图片"<<filename<<"保存失败！"<<endl;
    }
}

