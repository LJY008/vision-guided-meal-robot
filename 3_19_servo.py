import queue
import random
import sys
import torch
import cv2
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R
import os
import matplotlib.pyplot as plt
import pids_curve
import time
from threading import Thread, Event, Lock,Condition
from sort import Sort

dll_directory = os.path.join(os.path.dirname(__file__), 'bin')
os.environ['PATH'] = dll_directory
import math
from bin import DianaApi

# 将 yolov5 目录添加到系统路径
sys.path.append(os.path.join(os.getcwd(), 'yolov5'))

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, set_logging, scale_boxes
from utils.torch_utils import select_device, time_sync

# 初始化MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 初始化变量
threshold = 20  # 初始阈值

# 初始化RealSense相机
pipeline = rs.pipeline()
config = rs.config()

# 获取可用的设备信息
device_list = rs.context().devices
if not device_list:
    print("No RealSense devices found.")
    exit(1)

# 打印所有连接的设备及其序列号
for i, dev in enumerate(device_list):
    print(f"Device {i}: {dev.get_info(rs.camera_info.serial_number)}")

# 选择第一个设备（如果有多个设备）
serial_number = device_list[0].get_info(rs.camera_info.serial_number)
print(f"Using device with serial number: {serial_number}")

config.enable_device(serial_number)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# 启动管道
profile = pipeline.start(config)

# 获取深度传感器和校准数据
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
align_to = rs.align(rs.stream.color)

# 取食流程相机内参矩阵
fcamera_matrix = np.array([
    [582.13431296, 0, 316.00723293],
    [0, 582.99623535, 268.45817221],
    [0, 0, 1]
])

# 手眼标定结果
frotation_matrix = np.array([
    [0.01396891, 0.99845847, 0.04233127],
    [0.99925939, 0.03645072, -0.01232946],
    [-0.01385346, 0.0418573, -0.99902755]
])
ftranslation_vector = np.array([
    [-51.52974737],
    [-28.13069681],
    [193.99693148]
])

fhand_eye_matrix = np.eye(4)
fhand_eye_matrix[:3, :3] = frotation_matrix
fhand_eye_matrix[:3, 3] = ftranslation_vector.flatten()

# 用于喂食的相机内参矩阵
# 相机内参矩阵
camera_matrix = np.array([
    [569.60940903, 0, 319.00747461],
    [0, 569.7973498, 236.88064889],
    [0, 0, 1]
])

# 手眼标定结果
rotation_matrix = np.array([
    [-0.00310148, 0.99806818, -0.06205076],
    [0.99985684, 0.00412721, 0.01640924],
    [0.01663363, -0.06199098, -0.99793809]
])
translation_vector = np.array([
    [-9.96500977],
    [-45.99854704],
    [170.172746]
])

hand_eye_matrix = np.eye(4)
hand_eye_matrix[:3, :3] = rotation_matrix
hand_eye_matrix[:3, 3] = translation_vector.flatten()

# 初始化Diana API
dll_directory = os.path.join(os.path.dirname(__file__), 'bin')
os.environ['PATH'] = dll_directory
netInfo = ('192.168.10.75', 0, 0, 0, 0, 0)
DianaApi.initSrv(netInfo)
ipAddress = '192.168.10.75'
joints = [-0.6789181783265179, -0.45611943983119424, 0.7721917808314761, 2.6023867647909382, -1.5208100281084818, -0.5678124226065379, -0.9311741433949958]
# 解除刹车
DianaApi.releaseBrake(ipAddress)
# 工具坐标系设置
# matrix = (-28.8008,4.77148,-226.509,0.0,0.0,0.0)
# DianaApi.setDefaultActiveTcpPose(matrix,' 192.168.10.75')
#共享位置
target_queue = queue.Queue()  # 用于在线程之间传递目标坐标
# 设置 YOLOv5 参数
weights = 'yolov5/runs/train/food_custom_train7/weights/best.pt'  # 训练好的权重文件路径
imgsz = 640  # 图像大小
conf_thres = 0.4  # 置信度阈值
iou_thres = 0.7  # IoU 阈值
agnostic_nms = False  # 是否使用无类别 NMS
classes = None  # 类别索引列表（None 表示所有类别）
augment = False  # 是否增强推理

# 加载模型
set_logging()
device = select_device(0)  # 选择设备（'' 表示自动选择 CPU 或 GPU）

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
                label = f'{names[cls_int]} '
                plot_one_box(xyxy, frame, label=label, color=(0, 255, 0), line_thickness=3)
                time.sleep(0.5)
                detections.append([*xyxy, conf, cls_int])
    #print(detections)
    return detections, im0

# 初始化速度和加速度
velocity = 0.1
acceleration = 0.2
food = None
new_food=None

def move_to_init():
    vel = velocity + 0.4
    acc = acceleration
    DianaApi.moveJ(joints, vel, acc, ipAddress)


def prepare_get():
    vel = 0.3 + acceleration
    acc = acceleration
    poses = [0.651522613066998, -0.3167548801000275, -0.5354430292078041, 2.532722461263783, -0.5248130210831543,
             -0.7934394383481043, -1.9412885682002963]
    DianaApi.moveJ(poses, vel, acc, ipAddress)
    DianaApi.wait_move()
    global prepare_get_completed
    prepare_get_completed = True
    #ready = True

def fork_get():
    vel = 0.4 + velocity
    acc = acceleration
    poses = [-0.18573134140015227, 0.4093214132934251, -0.05678107373248098, 2.0317338934590126, 0.2997616226783828,
             -1.206535695458201, -2.8001500814597162]
    DianaApi.moveJ(poses, vel, acc, ipAddress)
    DianaApi.wait_move()


def fork_go_get():
    vel = 0.4 + velocity
    acc = acceleration
    poses = [0.48320416423870377, 0.2336326657090324, -0.8739614502866808, 1.7745164589839906, 0.4278130734245793,
             -1.7461733908864154, -2.7907664327995128]
    DianaApi.moveJ(poses, vel, acc, ipAddress)
    DianaApi.wait_move()


def fork_ready_feed():
    vel = 0.3 + velocity
    acc = acceleration
    POSES = [0.0] * 6

    DianaApi.getTcpPos(POSES, ipAddress)
    POSES[2] += 0.05
    DianaApi.moveLToPose(POSES, vel, acc, ipAddress)
    DianaApi.wait_move()

    poses = [0.010162810579007253, -0.3453013055260863, 0.14569241811187883, 2.5668055989261314, -1.4039877967679182,
             -0.7451669775520648, -1.1526306486209157]
    DianaApi.moveJ(poses, vel, acc, ipAddress)
    DianaApi.wait_move()


def go_to_get():
    vel = 0.4 + velocity
    acc = acceleration
    poses = [0.242884487887419, 0.37318897305011367, -0.21159330028740528, 2.461116713186243, -2.2410260351739146,
             -0.6758142638910227, 1.2844215522675]
    DianaApi.moveJ(poses, vel, acc, ipAddress)
    DianaApi.wait_move()
    get_it()


def get_it():
    vel = 0.3 + velocity
    acc = acceleration
    poses = [0.24005621064117655, 0.39273524503579926, -0.21443356175926723, 2.469577576473732, -2.0823548879694567,
             -0.6570709350218555, -0.2634849972205848]
    DianaApi.moveJ(poses, vel, acc, ipAddress)
    DianaApi.wait_move()


def ready_feed():
    vel = 0.1 + velocity
    acc = acceleration
    poses = [0.4046657083196351, -0.07034316241071939, 0.3924901975958283, 0.0, 0.0, 0.0]
    poses[3:] = axis2rpy(0.0, 68.0, -135.0)
    DianaApi.moveLToPose(poses, vel, acc, ipAddress)
    DianaApi.wait_move()


def estimate_mouth_position(landmarks, width, height):
    """估算嘴巴中心位置"""
    mouth_x1, mouth_y1 = landmarks[61].x * width, landmarks[61].y * height
    mouth_x2, mouth_y2 = landmarks[291].x * width, landmarks[291].y * height
    mouth_center_x = int((mouth_x1 + mouth_x2) / 2)
    mouth_center_y = int((mouth_y1 + mouth_y2) / 2)
    return mouth_center_x, mouth_center_y


def calculate_mouth_openness(landmarks, height):
    """计算嘴巴张开程度"""
    upper_lip = landmarks[13].y * height
    lower_lip = landmarks[14].y * height
    openness = abs(lower_lip - upper_lip)
    return openness


def xyz_rpy_to_homogeneous_matrix(x, y, z, roll, pitch, yaw):
    """将 Roll-Pitch-Yaw 转换为齐次变换矩阵"""
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
    rotation_matrix = r.as_matrix()

    homogeneous_matrix = np.eye(4)
    homogeneous_matrix[:3, :3] = rotation_matrix
    homogeneous_matrix[:3, 3] = [x, y, z]

    return homogeneous_matrix

def auto_pick_food(target_coords, tool):
    # prepare_get()
    # fork_go_get()
    #print(f"Picking food at position: {food['position']} using {tool}")
    global detect_food_flag
    detect_food_flag = False  # 暂停食物检测
    POSES = [0.0] * 6
    DianaApi.getTcpPos(POSES, ipAddress)
    angle_euler = POSES[3:]
    DianaApi.axis2RPY(angle_euler)
    for i in range(3):
        POSES[i] *= 1000.0
    for i in range(3):
        angle_euler[i] = math.degrees(angle_euler[i])
    POSES[3:] = angle_euler
    arm_position = POSES

    tool_base_matrix = xyz_rpy_to_homogeneous_matrix(
        arm_position[0], arm_position[1], arm_position[2],
        arm_position[3], arm_position[4], arm_position[5]
    )

    base_coords_homogeneous = tool_base_matrix @ target_coords
    base_coords = base_coords_homogeneous[:3] / 1000.0

    spoon = (0.0, 68.0, -135.0)
    fork = (1.6, -26.0, -51.0)
    if tool == 'spoon':
        poses = tuple(base_coords.tolist()) + spoon
    elif tool == 'fork':
        poses = tuple(base_coords.tolist()) + fork
    poses = list(poses)

    vel = velocity
    acc = acceleration
    arr = poses[3:]
    for i in range(3):
        arr[i] = math.radians(arr[i])
    DianaApi.rpy2Axis(arr)
    poses[3:] = arr

    DianaApi.moveLToPose(poses, vel, acc, ipAddress)
    DianaApi.wait_move()
    fork_ready_feed()
    DianaApi.wait_move()

    global auto_pick_food_completed
    auto_pick_food_completed = True


def pick_food(tool):
    """模拟机器人拿起食物的动作"""
    if tool == 'spoon':
        print(f"Picking up food with {tool}")
        # 这里可以添加具体的机械臂动作代码
        #T1_start = time.time()
        prepare_get()
        # detections, detected_image = detect(frame)
        # foodcenter(detections)
        go_to_get()
        ready_feed()

        #T1_end = time.time()
        #T1 = T1_end - T1_start
        # return T1
    elif tool == 'fork':
        print(f"Picking up food with {tool}")
        # 这里可以添加具体的机械臂动作代码
        T1_start = time.time()
        prepare_get()
        fork_go_get()
        fork_ready_feed()
        T1_end = time.time()
        T1 = T1_end - T1_start
        return T1


def place_food(target_coords, tool):
    """移动机械臂到目标位置并记录时间参数"""
    print('place food')
    POSES = [0.0] * 6
    DianaApi.getTcpPos(POSES, ipAddress)
    angle_euler = POSES[3:]
    DianaApi.axis2RPY(angle_euler)
    for i in range(3):
        POSES[i] *= 1000.0
    for i in range(3):
        angle_euler[i] = math.degrees(angle_euler[i])
    POSES[3:] = angle_euler
    arm_position = POSES

    tool_base_matrix = xyz_rpy_to_homogeneous_matrix(
        arm_position[0], arm_position[1], arm_position[2],
        arm_position[3], arm_position[4], arm_position[5]
    )

    base_coords_homogeneous = tool_base_matrix @ target_coords#基坐标系工具末端变化矩阵左乘工具系下目标的坐标

    base_coords = base_coords_homogeneous[:3] / 1000.0

    spoon = (0.0, 68.0, -135.0)
    fork = (0.0, 60.0, -135.0)
    if tool == 'spoon':
        poses = tuple(base_coords.tolist()) + spoon
    elif tool == 'fork':
        poses = tuple(base_coords.tolist()) + fork
    poses = list(poses)

    vel = velocity
    acc = acceleration
    arr = poses[3:]
    for i in range(3):
        arr[i] = math.radians(arr[i])
    DianaApi.rpy2Axis(arr)
    poses[3:] = arr

    # global place_food_completed
    # place_food_completed = True
    return poses
# def clean_tool(tool):
#     """清理工具"""
#     # 这里可以添加具体的机械臂动作代码
#     vel = velocity
#     acc = acceleration
#     if tool == 'spoon':
#         poses = [-0.5371208207945579, -0.5598309283438362, 0.5904629835347688, 2.522164358492852, -1.2195625487067843,
#                  -0.564708508171043, -1.3319985534705479]
#         DianaApi.moveJ(poses, vel, acc, ipAddress)
#         DianaApi.wait_move()
#         poses = [-1.25225551619725, 0.5385353339247296, 0.8656327879880594, 2.134594501952657, -2.9079242224574267,
#                  0.4313244515311432, -1.584937619399347]
#         DianaApi.moveJ(poses, vel, acc, ipAddress)
#         DianaApi.wait_move()
#         poses = [-1.3560868469660892, 0.8015291651484335, 1.0414773305055078, 2.178145177854546, -2.9702421956797216,
#                  0.16311748216288177, -1.5527959262873865]
#         DianaApi.moveJ(poses, vel, acc, ipAddress)
#         DianaApi.wait_move()
#         poses = [-1.3671363029874264, 0.7873877789172202, 1.0582792148242879, 2.189182649650264, -2.9478916148990333,
#                  0.16312946638850168, -1.5531674372815962]
#         DianaApi.moveJ(poses, vel, acc, ipAddress)
#         DianaApi.wait_move()
#         poses = [-1.4191118894999444, 0.6459859008307096, 1.1141496746631994, 2.2752054211482697, -3.1203326373404945,
#                  0.10071561936124951, -1.2423685300610203]
#         DianaApi.moveJ(poses, vel, acc, ipAddress)
#         DianaApi.wait_move()
#         poses = [-1.25225551619725, 0.5385353339247296, 0.8656327879880594, 2.134594501952657, -2.9079242224574267,
#                  0.4313244515311432, -1.584937619399347]
#         DianaApi.moveJ(poses, vel, acc, ipAddress)
#         DianaApi.wait_move()
#         poses = [-1.25225551619725, 0.5385353339247296, 0.8656447722136785, 2.134582517727038, -2.4072831971956257,
#                  0.4313244515311432, -1.584937619399347]
#         DianaApi.moveJ(poses, vel, acc, ipAddress)
#         DianaApi.wait_move()
#     elif tool == 'fork':
#         poses = [0.5438316105846452, 0.06364078447073893, 0.20560625835759933, 1.1881117570115416, 0.4921543797344951,
#                  -2.0999639147872893]
#         DianaApi.moveLToPose(poses, vel, acc, ipAddress)
#         DianaApi.wait_move()
#         poses = [-1.0396913063811262, 0.10694741668347917, 0.9691165762139288, 2.5824330291341835, -1.7957641165006184,
#                  -0.5358145402020136, -2.318540006481954]
#         DianaApi.moveJ(poses, vel + 0.3, acc, ipAddress)
#         DianaApi.wait_move()
#         poses = [-1.2373711079776173, 0.22041406485053416, 1.0322135241015014, 2.532003407726603, -1.7585411117259169, -0.28427762867071626, -2.317761031816675]
#         DianaApi.moveJ(poses, vel, acc, ipAddress)
#         DianaApi.wait_move()
#         poses = [-1.3538457967752104, -0.014237072782644855, 1.0419327310790552, 2.563414063075764, -1.8300869386753589,
#                  -0.011181095249628736, -2.317029994053875]
#         DianaApi.moveJ(poses, vel, acc, ipAddress)
#         DianaApi.wait_move()
#         poses = [-1.3502505290893085, -0.3366726630799226, 1.2992340551334156, 2.599270866129823, -1.2613635276682007,
#                  -0.29182769081110926, -1.3404234640811774]
#         DianaApi.moveJ(poses, vel, acc, ipAddress)
#         DianaApi.wait_move()
#         prepare_get()


# def record_time_parameters(T1, T2, T3, condition, file_writer):
#     """记录时间参数到CSV文件"""
#     file_writer.writerow({'Condition': condition, 'T1': T1, 'T2': T2, 'T3': T3})
#
#
# def plot_time_parameters(time_data):
#     """绘制时间参数的多维度折线图"""
#     conditions = sorted(time_data.keys())
#     T1_values = [time_data[cond]['T1'] for cond in conditions]
#     T2_values = [time_data[cond]['T2'] for cond in conditions]
#     T3_values = [time_data[cond]['T3'] for cond in conditions]
#
#     plt.figure(figsize=(10, 6))
#     plt.plot(conditions, T1_values, label='T1 (送餐时间)', marker='o')
#     plt.plot(conditions, T2_values, label='T2 (吞咽时间)', marker='s')
#     plt.plot(conditions, T3_values, label='T3 (餐具撤回时间)', marker='^')
#
#     plt.xlabel('实验条件')
#     plt.ylabel('时间参数 (秒)')
#     plt.title('不同条件下各阶段耗时分布')
#     plt.legend()
#     plt.grid(True)
#     plt.show()


def move_to_initial_position():
    """移动机械臂到初始位置"""
    move_to_init()
    DianaApi.wait_move()


def axis2rpy(rx, ry, rz):
    axis = [rx, ry, rz]
    for i in range(3):
        axis[i] = math.radians(axis[i])
    # 将欧拉角转换为轴角
    DianaApi.rpy2Axis(axis)
    return axis


def foodcenter(detections, depth_image, frame, tracker, camera_matrix, hand_eye_matrix):
    dets = np.array(detections)
    #print(dets)
    # 如果没有检测到任何物体，则跳过追踪步骤
    if len(dets) == 0:
        return None

    tracked_objects = tracker.update(dets[:, :4])

    for trk in tracked_objects:
        bbox = trk[:4]  # 获取边界框坐标
        track_id = int(trk[-1])  # 获取跟踪ID

        # 查找对应的检测结果中的分类ID
        closest_det_idx = np.argmin(np.sum((dets[:, :4] - bbox) ** 2, axis=1))
        class_id = int(dets[closest_det_idx][-1])
        x_center = int((bbox[0] + bbox[2]) // 2)  # 计算边界框中心点的X坐标并转换为整数
        y_center = int((bbox[1] + bbox[3]) // 2)  # 计算边界框中心点的Y坐标并转换为整数
        depth_value = depth_image[y_center, x_center]
        pixel_coords = np.array([x_center, y_center, 1.0])
        inv_camera_matrix = np.linalg.inv(camera_matrix)
        normalized_coords = inv_camera_matrix @ pixel_coords
        camera_coords = depth_value * normalized_coords

        lip_z_offset = 5
        camera_coords[2] += lip_z_offset

        camera_coords_homogeneous = np.append(camera_coords, 1)
        arm_coords_homogeneous = hand_eye_matrix @ camera_coords_homogeneous

        target_coords = arm_coords_homogeneous
        print('target_coords',target_coords)
        # cv2.circle(frame, (int(x_center), int(y_center)), 5, (0, 255, 0),
        #            -1)  # 绘制中心点
        # cv2.putText(frame, f'food', (int(bbox[0]), int(bbox[1] - 10)),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)  # 显示分类ID

        return target_coords

    return None


target_coords = None
time_data = {}
condition_counter = 0  # 用于跟踪实验条件或试验次数

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)  # 创建SORT追踪器实例

# 标志位用于控制YOLO检测是否开启
detect_food_flag = False

# 共享变量和锁

frame_lock = Lock()
shared_frame = np.zeros((480, 640, 3), dtype=np.uint8)  # 假设分辨率为480x640
robot_action_lock = Lock()

detect_food_flag = False
detect_face_flag = False  # 初始状态为人脸检测开启

food = None
velocity = 0.2
acceleration = 0.3
threshold = 20  # 嘴巴张开的阈值
prepare_get_completed = False  # 新增标志位
auto_process_steps = []
feeding_start_time = None
feeding_in_progress = False
place_food_completed=False

def curve_thread_worker(stop_event):
    """
    曲线控制线程，从队列中获取目标并调用 custom_servoL。
    """
    while not stop_event.is_set():
        try:
            # 从队列中获取目标坐标，超时时间为 0.1 秒
            target = target_queue.get(timeout=0.1)
            if target is not None:
                print(f"Curve thread received target: {target}")
                pids_curve.custom_servoL(target)  # 调用自定义伺服函数
                #time.sleep(0.05)
                target_queue.task_done()  # 标记任务完成
        except queue.Empty:
            # 如果队列为空，继续等待
            continue
        except Exception as e:
            print(f"Error in curve_thread_worker: {e}")
            break
def update_target_queue(new_target):
    """
    更新目标队列，确保队列中只保留最新的目标。
    :param new_target: 新的目标坐标
    """
    global target_queue
    with robot_action_lock:
        # 如果队列未满，则直接放入新目标；如果队列已满，则丢弃旧目标并放入新目标
        if target_queue.full():
            try:
                target_queue.get_nowait()  # 清空队列中的旧目标
            except queue.Empty:
                pass
        target_queue.put(new_target)  # 放入新目标
def process_frames(stop_event):
    global velocity, acceleration, detect_food_flag, detect_face_flag, target_coords, food, prepare_get_completed,place_food_completed, new_food

    while not stop_event.is_set():

        frames = pipeline.wait_for_frames()
        aligned_frames = align_to.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        height, width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if detect_face_flag:
            results = face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1),
                    )

                    mouth_x, mouth_y = estimate_mouth_position(face_landmarks.landmark, width, height)

                    if 0 <= mouth_x < width and 0 <= mouth_y < height:
                        openness = calculate_mouth_openness(face_landmarks.landmark, height)
                        cv2.circle(frame, (mouth_x, mouth_y), 5, (0, 0, 255), -1)
                        cv2.putText(frame, f'Mouth Openness: {openness:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                    (0, 255, 0), 2)

                        if openness > threshold:
                            cv2.putText(frame, 'Mouth Open!', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                            depth_value = depth_image[mouth_y, mouth_x]
                            cv2.putText(frame, f'Depth: {depth_value:.2f} mm', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                        (0, 0, 255), 2)

                            #print(f"Mouth Depth: {depth_value:.2f} mm")

                            pixel_coords = np.array([mouth_x, mouth_y, 1.0])
                            inv_camera_matrix = np.linalg.inv(camera_matrix)
                            normalized_coords = inv_camera_matrix @ pixel_coords
                            camera_coords = depth_value * normalized_coords

                            lip_z_offset = -55
                            camera_coords[2] += lip_z_offset

                            camera_coords_homogeneous = np.append(camera_coords, 1)
                            arm_coords_homogeneous = hand_eye_matrix @ camera_coords_homogeneous
                            if depth_value>100:
                                with robot_action_lock:
                                    target_coords = arm_coords_homogeneous


                        # else:
                        #   # 嘴巴闭合，停止喂食
                        #             with robot_action_lock:
                        #                 auto_process_steps.append('end_feeding')
                        #                 place_food_completed=True
        elif detect_food_flag:
            detections, detected_image = detect(frame)
            if detections:
                i = random.randint(0, len(detections) - 1)
                selected_detection = [detections[i]]
                #print("selected_detection",selected_detection)
                new_food =[]
                new_food = foodcenter(selected_detection, depth_image, frame, tracker, fcamera_matrix, fhand_eye_matrix)
                time.sleep(1)
                #print('new_food',new_food)
                if new_food is not None:
                    with robot_action_lock:
                        food = new_food
                        if 'auto_pick_food' not in auto_process_steps:
                            auto_process_steps.append('auto_pick_food')  # 添加自动拾取食物步骤

        # 将帧放入队列以便主线程显示
        with frame_lock:
            shared_frame[:] = frame.copy()

        # 控制帧率
        time.sleep(0.03)  # 约30帧每秒

def display_frames(stop_event):
    while not stop_event.is_set():
        with frame_lock:
            frame = shared_frame.copy()
        cv2.imshow('Mouth Detection', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            stop_event.set()  # 设置停止事件
            # 停止运动
            DianaApi.stop(ipAddress)

            # 启用刹车
            DianaApi.holdBrake(ipAddress)
            DianaApi.destroySrv(ipAddress)
            break

        elif key == ord('f'):
            with robot_action_lock:
                auto_process_steps.append('start_auto_process')  # 添加新的自动流程启动命令
                print("Added 'start_auto_process' to queue")  # 调试信息
                print('auto_process_steps',auto_process_steps)

def execute_robot_actions(stop_event):
    global target_coords, feeding_in_progress, velocity, acceleration, detect_food_flag, detect_face_flag, prepare_get_completed, auto_process_steps, feeding_start_time, auto_pick_food_completed
    feeding_in_progress = False
    move_to_initial_after_end_feeding = False
    none_target_timer = None  # 初始化计时器
    while not stop_event.is_set():
        with robot_action_lock:
            if auto_process_steps:
                action_name = auto_process_steps.pop(0)
                print('action',action_name)
            else:
                action_name = None

        if action_name == 'start_auto_process':
            auto_process_steps.extend(['move_to_initial_position', 'prepare_get'])
            print("Started auto process steps:", auto_process_steps)  # 调试信息

        elif action_name == 'move_to_initial_position':

            print("Executing move_to_initial_position")  # 调试信息
            move_to_initial_position()

        elif action_name == 'prepare_get':

            print("Executing prepare_get")  # 调试信息
            prepare_get()
            if auto_process_steps:
                auto_process_steps.pop(0)
            # 等待 prepare_get 完成
            while not prepare_get_completed and not stop_event.is_set():
                time.sleep(0.05)
            prepare_get_completed = False  # 重置标志位
            with robot_action_lock:
                detect_food_flag = True  # 启动食物检测
                detect_face_flag = False  # 关闭人脸检测

        elif action_name == 'auto_pick_food':

            if food is not None:
                print(f"Food object: {food}")  # 调试信息
                auto_pick_food(food, 'fork')
                if auto_process_steps:
                    auto_process_steps.pop(0)
                # 等待 auto_pick_food 完成
                while not auto_pick_food_completed and not stop_event.is_set():
                    time.sleep(0.05)
                auto_pick_food_completed = False  # 重置标志位
                time.sleep(0.5)
                with robot_action_lock:
                    detect_food_flag = False  # 关闭食物检测
                    detect_face_flag = True  # 人脸检测
                    auto_process_steps.append('start_feeding')  # 添加开始喂食步骤

        elif action_name == 'start_feeding':
            other_progress = False
            feeding_in_progress = True

            detect_face_flag = True
            print("Starting feeding")  # 调试信息

        elif action_name == 'end_feeding':
            feeding_in_progress = False
            print("Feeding ended.")
            move_to_initial_after_end_feeding = True  # 设置标志
            with robot_action_lock:
                detect_food_flag = False  # 关闭食物检测
                detect_face_flag = False  # 关闭人脸检测
                target_coords = None  # 清除目标坐标
            continue


        if feeding_in_progress:
            with robot_action_lock:
                current_target = target_coords
                #print("current_target",current_target)
                target_coords=None
            if current_target is not None:
                target=place_food(current_target, 'fork')
                pids_curve.custom_servoL(target)
                DianaApi.servoL_ex(target, t=1, ah_t=0.03, gain=200, scale=1, ipAddress='192.168.10.75')

                #update_target_queue(target)
                print("target",target)
                # time.sleep(2)
                none_target_timer = None  # 重置计时器
            else:
                if none_target_timer is None:
                    none_target_timer = time.time()  # 开始计时
                else:
                    elapsed_time = time.time() - none_target_timer
                    if elapsed_time > 2:
                        print("target_coords has been None for more than 3 seconds. Ending...")
                        # 嘴巴闭合，停止喂食
                        with robot_action_lock:
                            auto_process_steps.append('end_feeding')
                            none_target_timer = None  # 重置计时器
        # 在 feeding_in_progress 结束后检查标志并移动到初始位置
        if not feeding_in_progress and move_to_initial_after_end_feeding:
            time.sleep(1)
            move_to_initial_position()
            DianaApi.wait_move()
            # 重新启动取食和喂食流程
            with robot_action_lock:
                auto_process_steps.append('start_auto_process')
            move_to_initial_after_end_feeding = False  # 重置标志

        # 自动调整速度
        # if feeding_in_progress and len(auto_process_steps) == 0:
        #     if target_coords is None:
        #         velocity = max(velocity - 0.05, 0.05)
        #         acceleration = max(acceleration - 0.05, 0.1)
        #         print(f"Reduced Velocity: {velocity}, Reduced Acceleration: {acceleration}")
        #     else:
        #         velocity = min(velocity + 0.05, 1.0)
        #         acceleration = min(acceleration + 0.05, 2.0)
        #         print(f"Increased Velocity: {velocity}, Increased Acceleration: {acceleration}")

        time.sleep(0.005)  # 调整机器人动作检查频率以提高响应速度


stop_event = Event()

frame_thread = Thread(target=process_frames, args=(stop_event,))
display_thread = Thread(target=display_frames, args=(stop_event,))
robot_thread = Thread(target=execute_robot_actions, args=(stop_event,))
curve_thread=Thread(target=curve_thread_worker,args=(stop_event,))
try:
    frame_thread.start()
    display_thread.start()
    robot_thread.start()
    curve_thread.start()  # 启动曲线线程
    while not stop_event.is_set():
        time.sleep(0.1)  # 主线程等待停止事件

finally:
    stop_event.set()
    frame_thread.join()
    display_thread.join()
    robot_thread.join()
    curve_thread.join()
    pipeline.stop()
    cv2.destroyAllWindows()