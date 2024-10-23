#ifndef COMMON_H
#define COMMON_H

#include <iostream>
#include <semaphore.h>
#include <thread>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <atomic>
#include "image_pretrain.h"
#include "tensorrt.h"
#include "NvOnnxParser.h"
#include "NvInfer.h"
#include <sys/mman.h>   // for mmap, shm_open
#include <fcntl.h>  
#include <fstream>    //c++中文件被抽象为数据流
#include <unistd.h> 
#include <condition_variable> // 条件变量

#endif