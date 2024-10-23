#ifndef PARAMS_H
#define PARAMS_H
#include "common.h"


void Params_init();
extern std::map<std::string, std::vector<std::string>> classes_map;

struct InferenceParams {
    TensorRT& Engine;
    VideoProcessor& Video_Processor;
    sem_t* frame_semaphore;
    int shm_fd;
    std::map<std::string, std::vector<std::string>> classes_map;
};

#endif