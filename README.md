## 2024/10/16
### USB摄像头0实时采集图像，python通过opencv的gstreamer接口拉取视频流，再写入共享内存（加入信号量和互斥锁） 
#### (数据：32 x 3 x 112 x 112)

### tensorrt.cpp读取engine模型文件,使用模型进行推理（到创建buffer）

### image_pretrain.cpp 访问共享内存，将图片数据映射到本地，使用std::vector<cv::Mat>存放。

## 2024/10/17-v1

###  可以实现拉取视频流到推理全过程，但是推理结果不正确，考虑是视频流拉取的问题，没有获取到真正实时的共享内存中的数据。
     --调试1： 将获取到的图片保存到本地查看，结果全是黑色（正则化之后未将数据乘回去）
     --调试2： 图片显示黑白色的九宫格样子
     --调试3： 图片维度错了，改成112 x 112 x 3图片数据问题解决了。
## 2024/10/17-v2
### 可以从共享内存中读取并保存图片，正常显示。但是推理结果不好，基本都是一个结果就是数值大小不同。