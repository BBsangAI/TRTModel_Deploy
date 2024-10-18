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

class GetData():
    def __init__(self, 
                 width=112, 
                 height=112,
                 spatial_transform=None,
                 sample_duration=16):
        self.need_sample_duration = sample_duration
        self.spatial_transform = spatial_transform
        self.width = width
        self.height = height
        # 初始化共享内存块，大小为（sample_duration,height,width,3）
        self.frame_shape = (self.need_sample_duration,height,width,3)             
        self.shared_mem = shared_memory.SharedMemory(create=True, name='shared_memory1', size=np.prod(self.frame_shape)* np.dtype(np.float32).itemsize)
        # 使用锁和信号量保证线程安全
        self.lock = mp.Lock()
        self.sem_name = "my_semaphore1"
        try:   
            self.semaphore = posix_ipc.Semaphore(self.sem_name, flags=posix_ipc.O_CREX, initial_value=0)  # 信号量初始化为0 当信号量>0 则可访问，每当访问时，nums--
            print(" create semaphore in Python! ")
        except posix_ipc.ExistentialError:
            self.semaphore = posix_ipc.Semaphore(self.sem_name)
            print(f"Semaphore '{self.sem_name}' already exists and is opened.")
        atexit.register(self.cleanup)  # 注册清理函数

    def cleanup(self):
        print("Cleaning up shared memory...")
        self.shared_mem.close()  # 关闭共享内存
        self.shared_mem.unlink()  # 释放共享内存

    def capture(self):
        cap = cv2.VideoCapture("v4l2src device=/dev/video0 ! image/jpeg,width=1024,height=768,framerate=30/1 ! jpegdec ! videoconvert ! videoscale ! video/x-raw,width=180,height=150 ! videoflip method=horizontal-flip ! appsink", cv2.CAP_GSTREAMER)
        frame_count = 0
        frame = []
        frames = []
        if not cap.isOpened():
            print("Error: Could not open video source.")
            return;
    
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
                self.write_to_shared_memory(frames)
                frames = []
                frame_count = 0
        print("write successful!!")
        cap.release()
        return frames
    '''==================写入共享内存========================'''
    '''==================写入共享内存========================'''
    def write_to_shared_memory(self, frames):
        print("writing....")
        frames_tensor = torch.stack(frames)  # 堆叠所有frame为一个Tensor
        frames_tensor = frames_tensor.permute(0, 2, 3, 1)  # 转换为 (sample_duration, height, width, 3)
        frames_np = frames_tensor.numpy()  # 将Tensor转换为numpy数组
        with self.lock:    # 退出with时自动释放锁
            np_shm = np.ndarray(self.frame_shape, dtype=np.float32, buffer=self.shared_mem.buf)
            np_shm[:] = frames_np[:]
        process_time = time.time() - self.start_time
        print(process_time)
        # 释放信号量，通知其他进程数据已写入
        self.semaphore.release()
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

    
    

       
        
