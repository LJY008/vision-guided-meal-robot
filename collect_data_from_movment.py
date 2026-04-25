import sys
import os
import cv2
import numpy as np
import torch
import pyrealsense2 as rs
from sort import Sort  # 假设你已经下载了 sort.py 并放在项目的根目录下
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import threading
import time
import queue
import csv
import scipy.io

# 将 yolov5 目录添加到系统路径
sys.path.append(os.path.join(os.getcwd(), 'yolov5'))

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, set_logging, scale_boxes
from utils.torch_utils import select_device, time_sync

# 初始化 RealSense 管道
pipeline = rs.pipeline()
config = rs.config()

# 获取可用设备列表
device_list = []
devices = rs.context().query_devices()
for dev in devices:
    device_list.append(dev.get_info(rs.camera_info.serial_number))

if not device_list:
    print("未找到 RealSense 设备。")
    exit(1)

# 使用第一个设备
serial_number = device_list[0]
config.enable_device(serial_number)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 启动管道
profile = pipeline.start(config)

# 设置 YOLOv5 参数
weights = 'yolov5/runs/train/my_custom_train8/weights/best.pt'  # 训练好的权重文件路径
imgsz = 640  # 图像大小
conf_thres = 0.4  # 置信度阈值
iou_thres = 0.7  # IoU 阈值
agnostic_nms = False  # 是否使用无类别 NMS
classes = None  # 类别索引列表（None 表示所有类别）
augment = False  # 是否增强推理

# 加载模型
set_logging()
device = select_device('')  # 选择设备（'' 表示自动选择 CPU 或 GPU）

model = attempt_load(weights)  # 移除 map_location 参数
stride = int(model.stride.max())  # 模型步幅
imgsz = check_img_size(imgsz, s=stride)  # 检查图像大小

# 统一使用 float32 类型
model.float()

# 将模型移到设备上
model.to(device)

names = model.module.names if hasattr(model, 'module') else model.names  # 获取类别名称


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """调整图像大小以适应模型输入要求"""
    shape = img.shape[:2]  # 当前形状 [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # 计算缩放比例（新 / 旧）
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # 仅缩小，不放大（更好的测试 mAP）
        r = min(r, 1.0)

    # 计算填充
    ratio = r, r  # 宽高比
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh 填充

    if auto:  # 最小矩形
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh 填充
    elif scaleFill:  # 拉伸
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # 宽高比

    dw /= 2  # 将填充分为两部分
    dh /= 2

    if shape[::-1] != new_unpad:  # 缩放
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # 添加边框
    return img, ratio, (dw, dh)


def plot_one_box(x, im, color=None, label=None, line_thickness=3):
    """在图像 'im' 上绘制一个边界框"""
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # 线条/字体粗细
    color = color or [np.random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # 字体粗细
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # 填充
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def detect(frame):
    """进行目标检测"""
    img0 = frame.copy()
    img, ratio, pad = letterbox(img0, new_shape=imgsz, auto=False)

    # 将 BGR 转换为 RGB，并交换通道顺序
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.float()  # 统一使用 float32 类型
    img /= 255.0  # 归一化到 [0, 1]
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img, augment=augment)[0]

    # 应用 NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)

    detections = []
    for i, det in enumerate(pred):  # 每个图像的预测
        s, im0 = '', img0.copy()
        if len(det):
            # 将边界框重新缩放到原始图像尺寸
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0.shape, ratio_pad=(ratio, pad)).round()

            # 打印结果
            for *xyxy, conf, cls in reversed(det.cpu().numpy()):
                cls_int = int(cls)
                if cls_int >= len(names):
                    print(f"Warning: Class index {cls_int} out of range. Valid indices are 0-{len(names) - 1}.")
                    continue
                label = f'{names[cls_int]} {conf:.2f}'
                plot_one_box(xyxy, im0, label=label, color=(0, 255, 0), line_thickness=3)
                detections.append([*xyxy, conf, cls_int])

    return detections, im0

def save_to_matlab(timestamps, displacements):
    """将时间位移数据保存为MATLAB .mat文件"""
    for track_id, disp in displacements.items():
        timestamps_arr = np.array(timestamps[:len(disp)])
        disp_arr = np.array(disp)

        # 创建字典存储数据
        data_dict = {
            'timestamps': timestamps_arr,
            'displacements': disp_arr
        }

        # 保存为.mat文件
        filename = f'track_{track_id}.mat'
        scipy.io.savemat(filename, data_dict)


class TrackerData:
    def __init__(self, window_size=300, num_frames=8):
        self.lock = threading.Lock()
        self.tracker_positions = {}
        self.timestamps = []
        self.window_size = window_size  # 设置时间窗口大小（秒）
        self.num_frames = num_frames  # 设置用于比较的帧数
    def add_position(self, track_id, position, timestamp):
        with self.lock:
            if track_id not in self.tracker_positions:
                self.tracker_positions[track_id] = {'positions': []
                                                    }

            self.tracker_positions[track_id]['positions'].append(position)
            self.timestamps.append(timestamp)

            # 保持时间窗口内的数据
            while self.timestamps and self.timestamps[-1] - self.timestamps[0] > self.window_size:
                self.timestamps.pop(0)
                for pos_data in self.tracker_positions.values():
                    pos_data['positions'].pop(0)


    def get_displacements(self):
        displacements = {}
        with self.lock:
            for track_id, data in self.tracker_positions.items():
                positions = data['positions']
                displacements[track_id] = []
                if len(positions) < self.num_frames:
                    continue  # 如果没有足够的数据点，则跳过

                #print(len(positions))
                #if(len(positions)>self.num_frames):
                for i in range(1, len(positions)):
                    prev_pos = np.array(positions[i - self.num_frames])
                    curr_pos = np.array(positions[i])
                    displacement = (curr_pos[1] - prev_pos[1])
                    if abs(displacement)<=1.5:
                        displacement=0
                    elif abs(displacement)>=8:
                        displacement=0
                    #print(displacement)

                    displacements[track_id].append(displacement)  # 不再乘以 1000
            return displacements

    def get_timestamps(self):
        with self.lock:
            return self.timestamps.copy()


def update_plot(canvas, ax_time,ax_time1,timestamps, displacements):
    """更新绘图"""
    ax_time.clear()
    ax_time1.clear()
    labels_added = False

    for track_id, disp in displacements.items():
        timestamps_arr = np.array(timestamps[:len(disp)])
        disp_arr = np.array(disp)

        if len(disp_arr) < 2:
            continue  # 至少需要两个点才能进行绘图

        # 将 track_id 转换为对应的名字
        if track_id == 1:
            name = 'black'
        elif track_id == 0:
            name = 'red'
        else:
            name = f'Track ID: {track_id}'  # 默认情况使用 Track ID

        # 时间域绘图
        if track_id == 1:
            ax_time1.plot(timestamps_arr, disp_arr, label=name)
        elif track_id == 0:
            ax_time.plot(timestamps_arr, disp_arr, label=name)
        else:
            ax_time.plot(timestamps_arr, disp_arr, label=name)  # 默认情况下绘制在 ax_time 上

        labels_added = True

    ax_time.set_xlabel('Time (s)')
    ax_time.set_ylabel('Displacement (pixels)')
    ax_time.set_title('Time-Displacement Curve')
    ax_time1.set_xlabel('Time (s)')
    ax_time1.set_ylabel('Displacement (pixels)')
    ax_time1.set_title('Time-Displacement Curve')

    if labels_added:
        ax_time.legend()
        ax_time1.legend()

    canvas.draw()
    canvas.flush_events()


def main():
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)  # 创建SORT追踪器实例
    tracker_data = TrackerData(window_size=300)  # 设置时间窗口大小为 300 秒

    start_time = time.time()

    # 创建队列用于数据传输
    data_queue = queue.Queue()

    # 创建 Tkinter 窗口
    root = tk.Tk()
    root.title("Time-Displacement Analysis")

    fig, (ax_time,ax_time1)= plt.subplots(2,1,figsize=(8, 6))
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def process_queue():
        """处理队列中的数据并更新绘图"""
        try:
            while True:
                timestamps, displacements = data_queue.get_nowait()
                update_plot(canvas, ax_time,ax_time1,timestamps, displacements)
        except queue.Empty:
            pass
        root.after(50, process_queue)  # 每 50 ms 检查一次队列

    root.after(50, process_queue)  # 开始处理队列

    def tracking_loop():
        """主追踪循环"""
        try:
            while True:
                frames = pipeline.wait_for_frames()  # 获取帧
                color_frame = frames.get_color_frame()

                if not color_frame:
                    continue  # 如果没有获取到有效的帧，则跳过本次循环

                color_image = np.asanyarray(color_frame.get_data())  # 获取彩色图像数据

                detections, detected_image = detect(color_image)  # 进行目标检测
                #print(detections)
                # 确保 detections 是 NumPy 数组
                dets = np.array(detections)

                # 如果没有检测到任何物体，则跳过追踪步骤
                if len(dets) == 0:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break  # 如果按下 'q' 键，则退出循环
                    elif key == ord('s'):
                        timestamps, displacements = tracker_data.get_timestamps(), tracker_data.get_displacements()
                        save_to_matlab(timestamps, displacements)  # 按 S 键保存数据
                    cv2.imshow('RealSense Detection', detected_image)  # 显示处理后的图像
                    continue

                tracked_objects = tracker.update(dets[:, :4])  # 更新追踪对象，只传递边界框坐标

                current_time = time.time() - start_time

                for trk in tracked_objects:
                    bbox = trk[:4]  # 获取边界框坐标
                    track_id = int(trk[-1])  # 获取跟踪ID

                    # 查找对应的检测结果中的分类ID
                    closest_det_idx = np.argmin(np.sum((dets[:, :4] - bbox) ** 2, axis=1))
                    class_id = int(dets[closest_det_idx][-1])

                    x_center = int((bbox[0] + bbox[2]) // 2)  # 计算边界框中心点的X坐标并转换为整数
                    y_center = int((bbox[1] + bbox[3]) // 2)  # 计算边界框中心点的Y坐标并转换为整数
                    position = [x_center, y_center]

                    tracker_data.add_position(class_id, position, current_time)

                    cv2.circle(detected_image, (int(x_center), int(y_center)), 5, (0, 255, 0), -1)  # 绘制中心点
                    cv2.putText(detected_image, f'Class ID: {class_id}', (int(bbox[0]), int(bbox[1] - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)  # 显示分类ID


                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break  # 如果按下 'q' 键，则退出循环
                elif key == ord('s'):
                    timestamps, displacements = tracker_data.get_timestamps(), tracker_data.get_displacements()
                    save_to_matlab(timestamps, displacements)  # 按 S 键保存数据

                cv2.imshow('RealSense Detection', detected_image)  # 显示处理后的图像

                # 发送数据到绘图程序
                data_queue.put(
                    (tracker_data.get_timestamps(), tracker_data.get_displacements()))

        finally:
            pipeline.stop()  # 停止RealSense管道
            cv2.destroyAllWindows()  # 关闭所有OpenCV窗口
            root.quit()  # 关闭 Tkinter 窗口

    # 创建并启动追踪线程
    tracking_thread = threading.Thread(target=tracking_loop)
    tracking_thread.daemon = True
    tracking_thread.start()

    # 运行 Tkinter 主事件循环
    root.mainloop()


if __name__ == "__main__":
    main()



