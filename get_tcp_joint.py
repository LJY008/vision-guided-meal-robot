#import numpy as np
# from scipy.spatial.transform import Rotation as R
#
#
# def euler_angles_to_rotation_matrix(rx, ry, rz):
#     # 计算旋转矩阵
#     rx = rx * np.pi / 180
#     ry = ry * np.pi / 180
#     rz = rz * np.pi / 180
#
#     Rx = np.array([[1, 0, 0],
#                    [0, np.cos(rx), -np.sin(rx)],
#                    [0, np.sin(rx), np.cos(rx)]])
#
#     Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
#                    [0, 1, 0],
#                    [-np.sin(ry), 0, np.cos(ry)]])
#
#     Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
#                    [np.sin(rz), np.cos(rz), 0],
#                    [0, 0, 1]])
#
#     R = Rz @ Ry @ Rx
#     return R
#
#
# def xyz_rpy_to_homogeneous_matrix(x, y, z, roll, pitch, yaw):
#     # 将Roll、Pitch、Yaw转换为旋转矩阵
#     r = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
#     rotation_matrix = r.as_matrix()
#
#     # 创建4x4的齐次变换矩阵
#     homogeneous_matrix = np.eye(4)
#
#     # 将旋转矩阵放置在左上角的3x3部分
#     homogeneous_matrix[:3, :3] = rotation_matrix
#
#     # 将平移向量放置在第四列的前三个元素中
#     homogeneous_matrix[:3, 3] = [x, y, z]
#
#     return homogeneous_matrix
#
#
# # 示例使用
# x, y, z = 320, 0.1, 170.3
# roll, pitch, yaw = -179.9, 1.7, 3.8
# homogeneous_matrix = xyz_rpy_to_homogeneous_matrix(x, y, z, roll, pitch, yaw)
#
# print("齐次变换矩阵：")
# print(homogeneous_matrix)

import os
import cv2
import time
import math

dll_directory = os.path.join(os.path.dirname(__file__), 'bin')
os.environ['PATH'] = dll_directory

from bin import DianaApi

netInfo = ('192.168.10.75', 0, 0, 0, 0, 0)
DianaApi.initSrv(netInfo)
init_poses=[0.4,-0.004,0.584, 90.0, 90.0, -45.0]
joints = [0.0] * 7
POSES = [0.0] * 6

ipAddress = '192.168.10.75'

# try:
#     while True:
#
#         DianaApi.getTcpPos(POSES, ipAddress)
#         print(POSES)
#         time.sleep(0.2)
#         # 按下'q'键退出
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
# finally:
#     # 释放资源
#     DianaApi.destroySrv('192.168.10.75')
pose = [0.0] * 6
DianaApi.getTcpPos(pose, ipAddress)
print(f"机械臂原始pose:{pose}")
#
# # 提取轴角部分
# arr = pose[3:]
# print(f"机械臂轴角: {arr}")
arr=init_poses[3:]
for i in range(3):
     arr[i] = math.radians(arr[i])
# 将轴角转换为欧拉角
DianaApi.rpy2Axis(arr)
print(f"转换后的欧拉角 (弧度): {arr}")

# 将欧拉角从弧度转换为角度
# for i in range(3):
#     arr[i] = math.degrees(arr[i])

# 更新pose中的欧拉角部分
# pose[3:] = arr
# print(f"机械臂更新euler: {pose}")
init_poses[3:]=arr
print(init_poses)
vel = 0.05
acc = 0.4
#DianaApi.moveLToPose(init_poses, vel, acc, ipAddress)
DianaApi.getJointPos(joints, ipAddress)
print('joints are:',joints)
joints = [0, 0, 0, 0, 0, 0,0]
joints[0] = -1.0358923068596901
joints[1] = -0.4481619140197322
joints[2] = 1.2921034408897096
joints[3] = 2.6370930821855083
joints[4] = -1.843868798137982
joints[5] = -0.9288132509479201
joints[6] = -0.41986715733168634

#DianaApi.moveJ(joints, vel, acc, ipAddress)
#ianaApi.wait_move()

DianaApi.destroySrv('192.168.10.75')
