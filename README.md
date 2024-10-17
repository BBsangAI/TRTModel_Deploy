# USB摄像头0实时采集图像，python通过opencv的gstreamer接口拉取视频流，再写入共享内存（加入信号量和互斥锁） 
# 数据：32*3*112*112
# tensorrt.cpp读取engine模型文件,使用模型进行推理（到创建buffer）
# image_pretrain.cpp 访问共享内存，将图片数据映射到本地，使用std::vector<cv::Mat>存放。