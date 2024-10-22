#ifndef IMAGE_PRETRAIN_H
#define IMAGE_PRETRAIN_H

#include "common.h"

using namespace std;

class VideoProcessor {
public:
    VideoProcessor() : running(true), showVideo(false) {}

    void start() {
        // 启动捕获和(显示线程)
        captureThread = std::thread(&VideoProcessor::captureVideo, this);
        if(showVideo)
            displayThread = std::thread(&VideoProcessor::displayVideo, this);
    }
    void stop() {
        // 停止视频操作
        running = false;
        if (captureThread.joinable()) {
            captureThread.join();
        }
        if (displayThread.joinable()) {
            displayThread.join();
        }
    }
    void toggleDisplay() {
        // 切换视频显示状态
        showVideo = !showVideo;
    }
    cv::Mat GetVideoFrames(){
        std::lock_guard<std::mutex> lock(frameMutex); // 确保线程安全
        return frame.clone(); // 返回当前帧的副本
    }
    std::vector<cv::Mat> GetFramesFromShm(sem_t* semaphore, int shm_fd, const size_t frame_size);
    void captureVideo();
    void displayVideo();
    void save_images(std::vector<cv::Mat> images);   /* '''保存图片与本地，用于测试是否真正实时读取到了共享内存中的数据''' */


private:
    std::atomic<bool> running;  
    std::atomic<bool> showVideo; 
    std::thread captureThread;    // 视频捕获线程
    std::thread displayThread;    // 视频显示线程

    cv::Mat frame;          // 存储捕获的帧
    std::mutex frameMutex;  // 用于保护frame变量的互斥锁
};

class V2Img_Dataset{
public:

private:
    cv::Mat Inputs;
};



#endif
