// 设置暂停管道 ...
// 管道正在使用且不需要 PREROLL ...
// 设置播放管道 ...
// 错误：来自组件 /GstPipeline:pipeline0/GstV4l2Src:v4l2src0：Internal data stream error.
// 额外的调试信息：
// gstbasesrc.c(3072): gst_base_src_loop (): /GstPipeline:pipeline0/GstV4l2Src:v4l2src0:
// streaming stopped, reason not-negotiated (-4)
// Execution ended after 0:00:00.000082338
// 设置 NULL 管道 ...
// 释放管道资源 ...


// gst-launch-1.0 v4l2src device=/dev/video0 ! 'video/x-raw, width=640, height=480, framerate=120/1' ! videoconvert ! autovideosink

// cap = cv2.VideoCapture("v4l2src ! video/x-raw, width=640, height=480 ! appsink", cv2.CAP_GSTREAMER)
// gst-launch-1.0 v4l2src device=/dev/video0 ! image/jpeg,width=800,height=600,framerate=60/1 ! jpegdec ! videoconvert ! queue max-size-buffers=1 ! ximagesink

// cmake -D CMAKE_BUILD_TYPE=RELEASE \
//       -D CMAKE_INSTALL_PREFIX=/usr/local \
//       -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
//       -D WITH_CUDA=ON \
//       -D CUDA_ARCH_BIN="8.6" \
//       -D WITH_GSTREAMER=ON \
//       -D BUILD_opencv_python3=ON \
//       -D OPENCV_PYTHON3_INSTALL_PATH=/usr/local/python/cv2 \
//       -D BUILD_EXAMPLES=ON ..

// jetson@unbutu:~/Desktop/MyAlgorithm/build$ gst-launch-1.0 v4l2src device=/dev/video0 ! image/jpeg ,width=320,height=240,framerate=120/1 ! nvjpegdec ! videoconvert ! ximagesink
// 设置暂停管道 ...
// 管道正在使用且不需要 PREROLL ...
// 设置播放管道 ...
// New clock: GstSystemClock
// 错误：来自组件 /GstPipeline:pipeline0/GstV4l2Src:v4l2src0：Internal data stream error.
// 额外的调试信息：
// gstbasesrc.c(3072): gst_base_src_loop (): /GstPipeline:pipeline0/GstV4l2Src:v4l2src0:
// streaming stopped, reason not-negotiated (-4)
// Execution ended after 0:00:00.000137540
// 设置 NULL 管道 ...
// 释放管道资源 ...

// jetson@unbutu:~/Desktop/MyAlgorithm/build$ GST_DEBUG=3 gst-launch-1.0 v4l2src device=/dev/video0 ! "image/jpeg,width=320,height=240,framerate=120/1" ! jpegdec ! videoconvert ! ximagesink
// 设置暂停管道 ...
// 管道正在使用且不需要 PREROLL ...
// 0:00:00.095263515 10029 0xaaaaf4844120 FIXME           videodecoder gstvideodecoder.c:946:gst_video_decoder_drain_out:<jpegdec0> Sub-class should implement drain()
// 设置播放管道 ...
// 0:00:00.095653573 10029 0xaaaaf4844120 WARN                 basesrc gstbasesrc.c:3072:gst_base_src_loop:<v4l2src0> error: Internal data stream error.
// 0:00:00.095673158 10029 0xaaaaf4844120 WARN                 basesrc gstbasesrc.c:3072:gst_base_src_loop:<v4l2src0> error: streaming stopped, reason not-negotiated (-4)
// New clock: GstSystemClock
// 错误：来自组件 /GstPipeline:pipeline0/GstV4l2Src:v4l2src0：Internal data stream error.
// 额外的调试信息：
// gstbasesrc.c(3072): gst_base_src_loop (): /GstPipeline:pipeline0/GstV4l2Src:v4l2src0:
// streaming stopped, reason not-negotiated (-4)
// Execution ended after 0:00:00.000108387
// 设置 NULL 管道 ...
// 释放管道资源 ...
// jetson@unbutu:~/Desktop/MyAlgorithm/build$ 
// gst-launch-1.0 v4l2src device=/dev/video0 ! "image/jpeg,width=800,height=600,framerate=60/1" ! jpegdec ! videoconvert ! 
//                        videoscale! video/x-raw,width=180,height=150 ! videoflip method=horizontal-flip ! ximagesink



