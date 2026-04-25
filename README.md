# vision-guided-meal-robot

基于 RGB-D 视觉引导的机器人辅助喂食系统，融合食物目标检测、人脸网格口部定位与机械臂伺服控制，为运动功能障碍用户提供自适应闭环进食辅助。

---

## 系统概述

系统采用四线程并发架构运行：

- **帧采集线程** — 从 RGB-D 相机持续采集彩色帧与深度帧，执行视觉检测与坐标解算
- **显示线程** — 实时渲染视觉反馈画面
- **机器人动作调度线程** — 基于有限状态机管理完整喂食流程
- **曲线伺服线程** — 执行平滑轨迹规划，驱动机械臂末端运动

主流程循环如下：

```
系统初始化 → 食物目标检测与三维定位 → 机械臂取食动作
→ 口部状态检测与三维定位 → 实时伺服送餐跟踪
→ 安全撤回与复位 → 下一轮循环
```

---

## 硬件要求

- **机械臂**：Diana 7（通过局域网 Diana API 控制）
- **RGB-D 相机**：Intel RealSense D400 系列（彩色流 + 深度流，640×480，30 fps）
- **末端执行器**：安装于机械臂法兰的餐叉或餐勺

---

## 依赖模块

### 核心库

| 库 | 用途 | 官方链接 |
|----|------|---------|
| **YOLOv5** | 食物目标检测 | [github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5) |
| **MediaPipe** | 人脸网格关键点检测与口部定位 | [mediapipe.dev](https://mediapipe.dev) · [PyPI](https://pypi.org/project/mediapipe/) |
| **pyrealsense2** | Intel RealSense SDK Python 封装 | [github.com/IntelRealSense/librealsense](https://github.com/IntelRealSense/librealsense) · [PyPI](https://pypi.org/project/pyrealsense2/) |
| **PyTorch** | YOLOv5 深度学习推理后端 | [pytorch.org](https://pytorch.org) |
| **OpenCV** | 图像处理与可视化 | [opencv.org](https://opencv.org) · [PyPI](https://pypi.org/project/opencv-python/) |
| **NumPy** | 数值计算与坐标变换 | [numpy.org](https://numpy.org) |
| **SciPy** | 旋转矩阵与 RPY/轴角转换（scipy.spatial.transform） | [scipy.org](https://scipy.org) |
| **SORT** | 食物目标帧间稳定跟踪 | [github.com/abewley/sort](https://github.com/abewley/sort) |

### 机械臂 SDK

Diana API（`bin/` 目录）包含厂商私有 DLL 及 Python 封装文件 `DianaApi.py`，**该目录不包含在本仓库中**，原因是相关文件属于 [Agile Robots（思灵机器人）](https://www.agile-robots.com/) 的私有 SDK，未经授权不可公开分发。

请通过以下方式获取：
1. 联系 Agile Robots 官方获取 Diana SDK
2. 将获取到的 DLL 及 `DianaApi.py` 放入项目根目录下的 `bin/` 文件夹后再运行

---

## 项目结构

```
vision-guided-meal-robot/
├── 4_24servo.py                  # 主程序入口，完整喂食流程
├── pids_curve.py                 # 基于 PID 的笛卡尔伺服轨迹控制器
├── sort.py                       # SORT 多目标跟踪器
├── food_custom_data.yaml         # 食物类别配置（YOLOv5 格式）
├── .gitignore
├── bin/                          # ⚠️ 不含于仓库（厂商私有 SDK，需自行获取）
│   ├── DianaApi.py               # Diana 机械臂 Python API
│   └── *.dll                     # Diana SDK 原生库（Windows）
├── food_dataset1/                # 食物训练数据集
│   ├── images/
│   └── labels/
└── yolov5/
    ├── models/                   # YOLOv5 网络结构定义
    ├── utils/                    # 推理工具函数（NMS、坐标缩放等）
    ├── requirements.txt
    └── runs/train/food_custom_train7/weights/
        └── best.pt               # 食物检测训练权重
```

---

## 安装

```bash
# 1. 克隆仓库
git clone https://github.com/LJY008/vision-guided-meal-robot.git
cd vision-guided-meal-robot

# 2. 安装 YOLOv5 依赖
pip install -r yolov5/requirements.txt

# 3. 安装其余依赖
pip install mediapipe pyrealsense2 scipy
```

> **注意**：PyTorch 需根据本机 GPU 型号单独安装对应 CUDA 版本。
> 请参考 [pytorch.org/get-started](https://pytorch.org/get-started/locally/) 选择安装命令。

---

## 运行前配置

运行前请在 `4_24servo.py` 中确认以下参数：

| 参数 | 位置 | 说明 |
|------|------|------|
| `ipAddress` | 约第 99 行 | Diana 机械臂 IP 地址 |
| `weights` | 约第 37 行 | YOLOv5 训练权重路径 |
| `fcamera_matrix` / `fhand_eye_matrix` | 约第 70 行 | 取食相机内参与手眼标定矩阵 |
| `camera_matrix` / `hand_eye_matrix` | 约第 83 行 | 喂食相机内参与手眼标定矩阵 |
| `threshold` | 约第 104 行 | 张口检测阈值（关键点像素距离） |

---

## 运行

连接机械臂与 RealSense 相机后执行：

```bash
python 4_24servo.py
```

**运行期间按键说明：**

| 按键 | 功能 |
|------|------|
| `f` | 启动自动取食与喂食循环 |
| `q` | 急停 — 停止所有关节运动，激活制动器，断开连接 |

---

## 手眼标定说明

系统使用两套独立的相机-机械臂坐标变换：

- **取食相机**（`fcamera_matrix` / `fhand_eye_matrix`）：朝向餐盘区域，用于食物三维定位
- **喂食相机**（`camera_matrix` / `hand_eye_matrix`）：朝向用户面部，用于口部实时跟踪

标定矩阵直接存储于 `4_24servo.py` 中，部署前请替换为实际标定结果。

---

## 许可证

本项目仅供学术研究使用。内嵌 YOLOv5 遵循 [AGPL-3.0 许可证](https://github.com/ultralytics/yolov5/blob/master/LICENSE)，SORT 遵循 [GPL-3.0 许可证](https://github.com/abewley/sort/blob/master/LICENSE)。
