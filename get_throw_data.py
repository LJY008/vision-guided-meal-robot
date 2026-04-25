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
import time
import csv
from PIL import Image, ImageTk

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
serial_number = device_list[1]  # 修改这里以选择不同的设备
config.enable_device(serial_number)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

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


class TrackerData:
    def __init__(self, window_size=300, num_frames=2, detection_interval=0.1, swallow_timeout=0.2, displacement_threshold=5):
        self.tracker_positions = {}
        self.timestamps = []
        self.window_size = window_size  # 设置时间窗口大小（秒）
        self.num_frames = num_frames  # 设置用于比较的帧数
        self.detection_interval = detection_interval  # 设置检测间隔时间
        self.swallow_timeout = swallow_timeout  # 设置吞咽超时时间（秒）
        self.displacement_threshold = displacement_threshold  # 设置位移差距阈值（像素）
        self.last_detection_time = 0  # 初始化上次检测时间为0
        self.last_change_time = {0: None, 1: None}  # 初始化最后变化时间为None
        self.last_displacement = {0: 0, 1: 0}  # 初始化最后位移为0
        self.swallow_sequence = False  # 标记是否进入吞咽序列

    def add_position(self, track_id, position, timestamp):
        if track_id not in self.tracker_positions:
            self.tracker_positions[track_id] = {'positions': []}

        self.tracker_positions[track_id]['positions'].append(position)
        self.timestamps.append(timestamp)

        # 保持时间窗口内的数据
        while self.timestamps and self.timestamps[-1] - self.timestamps[0] > self.window_size:
            self.timestamps.pop(0)
            for pos_data in self.tracker_positions.values():
                pos_data['positions'].pop(0)

    def get_displacements(self):
        displacements = {}
        for track_id, data in self.tracker_positions.items():
            positions = data['positions']
            displacements[track_id] = []
            if len(positions) < self.num_frames:
                continue  # 如果没有足够的数据点，则跳过

            for i in range(1, len(positions)):
                prev_pos = np.array(positions[i - self.num_frames])
                curr_pos = np.array(positions[i])
                displacement = (curr_pos[1] - prev_pos[1])
                if abs(displacement) <= 1:
                    displacement = 0
                elif abs(displacement) >= 8:
                    displacement = 0
                displacements[track_id].append(displacement)  # 不再乘以 1000
        return displacements

    def get_timestamps(self):
        return self.timestamps.copy()

    def monitor_chewing_swallowing(self, displacements):
        current_time = time.time()
        if current_time - self.last_detection_time < self.detection_interval:
            return

        latest_0 = None
        latest_1 = None

        for track_id, disp in displacements.items():
            if not disp:
                continue

            latest_displacement = disp[-1]

            if track_id == 0:
                if latest_displacement != self.last_displacement[0]:
                    self.last_change_time[0] = current_time
                    self.last_displacement[0] = latest_displacement
                    #print(f"Track 0 changed at {current_time}, displacement={latest_displacement}")
                    latest_0 = latest_displacement

            elif track_id == 1:
                if latest_displacement != self.last_displacement[1]:
                    self.last_change_time[1] = current_time
                    self.last_displacement[1] = latest_displacement
                    #print(f"Track 1 changed at {current_time}, displacement={latest_displacement}")
                    latest_1 = latest_displacement

        # 确保 latest_0 和 latest_1 不是 None
        if latest_0 is not None and latest_1 is not None:
            time_diff = abs(self.last_change_time[0] - self.last_change_time[1])

            # 检测咀嚼
            if time_diff <= self.detection_interval:
                if ((latest_0 > 0 and latest_1 > 0) or (latest_0 < 0 and latest_1 < 0)) and abs(latest_0 - latest_1) <= self.displacement_threshold:
                    print("Chewing detected")

            # 检测吞咽
            if self.swallow_sequence:
                if latest_0 == 0 and latest_1 == 0:
                    #print("Swallowing detected")
                    self.swallow_sequence = False
            else:
                if latest_1 == 0:
                    if latest_0 == 0:
                        #print("Swallowing detected")
                        self.swallow_sequence = False
                    elif self.last_displacement[0] == 0:
                        #print("Swallowing detected")
                        self.swallow_sequence = False
                    else:
                        if current_time - self.last_change_time[1] <= self.swallow_timeout:
                            self.swallow_sequence = True
                elif latest_0 == 0:
                    if latest_1 == 0:
                        #print("Swallowing detected")
                        self.swallow_sequence = False
                    elif self.last_displacement[1] == 0:
                        #print("Swallowing detected")
                        self.swallow_sequence = False
                    else:
                        if current_time - self.last_change_time[0] <= self.swallow_timeout:
                            self.swallow_sequence = True
                else:
                    if abs(latest_0 - latest_1) > self.displacement_threshold:
                        self.swallow_sequence = True
                    else:
                        self.swallow_sequence = False

        self.last_detection_time = current_time


def update_plot(canvas, ax_time, ax_time1, timestamps, displacements):
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
        ax_time.legend(loc='upper left')
        ax_time1.legend(loc='upper left')

    canvas.draw()
    canvas.flush_events()


def save_to_csv(filename, timestamps, displacements):
    """将时间戳和位移数据保存到CSV文件"""
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ['Timestamp'] + [f'Track_{track_id}' for track_id in sorted(displacements.keys())]
        writer.writerow(header)

        max_length = max(len(disp) for disp in displacements.values())
        for i in range(max_length):
            row = [timestamps[i]] + [displacements[track_id][i] if i < len(displacements[track_id]) else '' for track_id in sorted(displacements.keys())]
            writer.writerow(row)


def main():
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)  # 创建SORT追踪器实例
    tracker_data = TrackerData(window_size=300, detection_interval=0.1, swallow_timeout=0.2, displacement_threshold=5)  # 设置时间窗口大小为 300 秒

    start_time = time.time()

    # 启动 RealSense 管道
    profile = pipeline.start(config)

    # 创建 Tkinter 窗口
    root = tk.Tk()
    root.title("Time-Displacement Analysis")

    # 设置固定窗口大小
    root.geometry("1280x480")  # 总宽度为1280，高度为480

    # 创建两个 Frame 用于放置 OpenCV 图像和 Matplotlib 绘图
    image_frame = tk.Frame(root)
    plot_frame = tk.Frame(root)

    image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    # 设置每个 Frame 的宽度为总宽度的一半
    image_frame.config(width=640, height=480)
    plot_frame.config(width=640, height=480)

    fig, (ax_time, ax_time1) = plt.subplots(2, 1, figsize=(6.4, 4.8))  # 设置与 Tkinter Frame 对应的大小
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # 紧凑布局以避免标签被裁剪

    # 调整子图之间的间距
    plt.subplots_adjust(hspace=0.4)  # 增加 hspace 值以避免标签重叠

    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    # 创建标签用于显示 OpenCV 图像
    image_label = tk.Label(image_frame)
    image_label.pack(side=tk.TOP)

    running = True

    def on_key(event):
        nonlocal running
        if event.char == 'q':
            running = False
        elif event.char == 's':
            filename = "time_displacement_data.csv"
            timestamps = tracker_data.get_timestamps()
            displacements = tracker_data.get_displacements()
            save_to_csv(filename, timestamps, displacements)
            print(f"Data saved to {filename}")

    root.bind('<Key>', on_key)

    def update_image_and_plot():
        nonlocal running
        if not running:
            root.quit()
            return

        frames = pipeline.wait_for_frames()  # 获取帧
        color_frame = frames.get_color_frame()

        if not color_frame:
            root.after(10, update_image_and_plot)  # 如果没有获取到有效的帧，则稍后重试
            return

        color_image = np.asanyarray(color_frame.get_data())  # 获取彩色图像数据

        detections, detected_image = detect(color_image)  # 进行目标检测
        # 确保 detections 是 NumPy 数组
        dets = np.array(detections)
        #print(dets)
        # 如果没有检测到任何物体，则跳过追踪步骤
        if len(dets) == 0:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                running = False
                return

            # 显示处理后的图像
            rgb_image = cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            photo_image = ImageTk.PhotoImage(image=pil_image)
            image_label.image = photo_image  # 保留对 PhotoImage 的引用
            image_label.configure(image=photo_image)
            root.after(10, update_image_and_plot)
            return

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

            cv2.circle(detected_image, (int(x_center), int(y_center)), 5, (0, 255, 0),
                       -1)  # 绘制中心点
            #

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            running = False
            return

        # 显示处理后的图像
        rgb_image = cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        photo_image = ImageTk.PhotoImage(image=pil_image)
        image_label.image = photo_image  # 保留对 PhotoImage 的引用
        image_label.configure(image=photo_image)

        # 发送数据到绘图程序
        timestamps = tracker_data.get_timestamps()
        displacements = tracker_data.get_displacements()

        # 更新绘图
        update_plot(canvas, ax_time, ax_time1, timestamps, displacements)

        # 监控咀嚼和吞咽动作
        tracker_data.monitor_chewing_swallowing(displacements)

        # 处理 Tkinter 事件
        root.after(10, update_image_and_plot)

    # 启动 Tkinter 主循环
    try:
        update_image_and_plot()
        root.mainloop()
    finally:
        pipeline.stop()  # 停止RealSense管道
        cv2.destroyAllWindows()  # 关闭所有OpenCV窗口


if __name__ == "__main__":
    main()