import torch.utils.data as data
from PIL import Image
import torch
import cv2
import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
from torchvision import transforms
import atexit
import time
import posix_ipc
import select


class GetData():
    def __init__(self, 
                 width=112, 
                 height=112,
                 spatial_transform=None,
                 sample_duration=2):
        self.need_sample_duration = sample_duration
        self.spatial_transform = spatial_transform
        self.width = width
        self.height = height
        # 初始化共享内存块，大小为（sample_duration,height,width,3）
        self.frame_shape1 = (2,height,width,3)      
        self.frame_shape2 = (16,height,width,3)          
        self.shared_mem1 = shared_memory.SharedMemory(create=True, name='shared_memory1', size=np.prod(self.frame_shape1)* np.dtype(np.float32).itemsize)    
        self.shared_mem2 = shared_memory.SharedMemory(create=True, name='shared_memory2', size=np.prod(self.frame_shape2)* np.dtype(np.float32).itemsize)    
        self.signal_shared_mem = shared_memory.SharedMemory(create=True, name='signal_shared_memory1', size=np.dtype(np.bool).itemsize)
        # 使用锁和信号量保证线程安全
        self.lock = mp.Lock()
        self.frame_semname = "frame_semaphore"
        self.init_semname = "init_semaphore"
        try:   
            self.frames_semaphore = posix_ipc.Semaphore(self.frame_semname, flags=posix_ipc.O_CREX, initial_value=0)  # 信号量初始化为0 当信号量>0 则可访问，每当访问时，nums--
            self.init_semaphore = posix_ipc.Semaphore(self.init_semname, flags=posix_ipc.O_CREX, initial_value=0)
            print(" create semaphore in Python! ")
        except posix_ipc.ExistentialError:
            self.frames_semaphore = posix_ipc.Semaphore(self.frame_semname)
            self.init_semaphore = posix_ipc.Semaphore(self.init_semname)
            print(f"Semaphore already exists and is opened.")
        atexit.register(self.cleanup)  # 注册清理函数

    def cleanup(self):
        print("Cleaning up shared memory...")
        self.shared_mem1.close()  # 关闭共享内存
        self.shared_mem1.unlink()  # 释放共享内存
        self.shared_mem2.close()  # 关闭共享内存
        self.shared_mem2.unlink()  # 释放共享内存
        self.signal_shared_mem.close()  # 关闭共享内存
        self.signal_shared_mem.unlink()  # 释放共享内存
        self.frames_semaphore.unlink()
        self.init_semaphore.unlink()

    def check_detect_signal(self):
        if bool(self.signal_shared_mem.buf[0]):  
            print("Gesture event detected.")
            return True  # 返回 True 表示检测到手势信号
        else:
            # 如果信号量没有被释放，继续执行代码
            print("No gesture event detected.")
            return False  # 返回 False 表示没有检测到手势信号
    
    def capture(self):
        cap = cv2.VideoCapture("v4l2src device=/dev/video0 ! image/jpeg,width=1024,height=768,framerate=30/1 ! jpegdec ! videoconvert ! videoscale ! video/x-raw,width=180,height=150 ! appsink", cv2.CAP_GSTREAMER)
        frame_count = 0
        frame = []
        frames = []      
        if not cap.isOpened():
            print("Error: Could not open video source.")
            return;
        self.init_semaphore.acquire()
        while True:
            frames = []
            frame_count = 0
            if self.check_detect_signal():
                self.need_sample_duration = 16  # 如果检测到手势，采集 16 帧
            else:
                self.need_sample_duration = 2   # 否则保持默认帧数
            while frame_count < self.need_sample_duration:     # 采集图像
                ok, frame = cap.read()
                if frame_count == 0:
                    self.start_time = time.time()
                if not ok:
                    print("Failed to capture image from camera.")
                    return
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # OpenCV 默认是 BGR，需要转换为 RGB
                frame_pil = Image.fromarray(frame_rgb)

                # 图像transform后添加到图像队列直到满足帧数
                if self.spatial_transform is not None:
                    frame_pil = self.spatial_transform(frame_pil)
                '''================== TEST:保存图像到本地 ========================'''
                # file_name = "../images_test/output_"+str(frame_count)+".jpg"
                # if cv2.imwrite(file_name, frame):
                #     print(f"save{file_name} successful!")
                '''================== TEST:保存图像到本地 ========================'''
                frames.append(frame_pil)
                frame_count += 1
                if frame_count == self.need_sample_duration:
                # 写入共享内存
                    if self.need_sample_duration == 16:
                        self.write_to_shared_memory(frames, self.shared_mem2)
                        self.signal_shared_mem.buf[0] = False
                    else:
                        self.write_to_shared_memory(frames, self.shared_mem1)
    '''==================写入共享内存========================'''
    '''==================写入共享内存========================'''
    def write_to_shared_memory(self, frames, inshared_mem):
        print("writing....")
        frames_tensor = torch.stack(frames)  # 堆叠所有frame为一个Tensor
        frames_tensor = frames_tensor.permute(0, 2, 3, 1)  # 转换为 (sample_duration, height, width, 3)
        frames_np = frames_tensor.numpy()  # 将Tensor转换为numpy数组
        with self.lock:    # 退出with时自动释放锁
            np_shm = np.ndarray((self.need_sample_duration,112,112,3), dtype=np.float32, buffer=inshared_mem.buf)
            np_shm[:] = frames_np[:]
        process_time = time.time() - self.start_time
        print(process_time)
        # 释放信号量，通知其他进程数据已写入
        self.frames_semaphore.release()
    '''============== ======================= ================'''
    '''============== ======================= ================'''
norm_method = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
crop_method = transforms.CenterCrop(112)
spatial_transform = transforms.Compose([
        crop_method,
        transforms.ToTensor(),
        norm_method
    ])

if __name__ == '__main__': 
    get_datas = GetData(spatial_transform=spatial_transform)
    get_datas.capture()

    
    

       
        
