import time
import numpy as np
from time import sleep
import os

# 设置 DLL 路径
dll_directory = os.path.join(os.path.dirname(__file__), 'bin')
os.environ['PATH'] = dll_directory + os.pathsep + os.environ['PATH']
from bin import DianaApi


# PID 控制器类
class PIDController:
    def __init__(self, kp, ki, kd):
        """
        初始化 PID 控制器参数
        :param kp: 比例增益
        :param ki: 积分增益
        :param kd: 微分增益
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0

    def compute(self, setpoint, current_value, dt):
        """
        计算 PID 控制输出
        :param setpoint: 目标值
        :param current_value: 当前值
        :param dt: 时间步长 (秒)
        :return: 控制输出
        """
        error = setpoint - current_value
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output


def custom_servoL(tcp_pose_target, t=2.0, dt=0.05, ipAddress='192.168.10.75'):
    """
    替代原生 servoL 函数，使用直线插补与 PID 控制
    :param tcp_pose_target: 目标位姿列表，长度为 6。前三个元素单位：m；后三个元素单位：rad
    :param t: 运动时间。单位：s。
    :param dt: 时间步长 (秒)
    :param ipAddress: 可选参数，需要控制机械臂的 IP 地址字符串，不填仅当只连接一台机械臂时生效。
    :return: True：成功。False：失败。
    """
    # 获取当前 TCP 位姿
    tcp_pose_current = [0, 0, 0, 0, 0, 0]
    DianaApi.getTcpPos(tcp_pose_current, ipAddress)

    # 初始化 PID 控制器
    pid_controllers = [PIDController(kp=2.0, ki=0.0, kd=0.4) for _ in range(6)]

    # 计算插补点数量
    num_steps = int(t / dt)

    # 使用 NumPy 向量化计算插补点
    alphas = np.linspace(0, 1, num_steps)
    tcp_poses_interpolated = [
        tcp_pose_current + alpha * (np.array(tcp_pose_target) - np.array(tcp_pose_current))
        for alpha in alphas
    ]

    # 对每个插值点进行逆运动学和关节角度更新
    for step in range(num_steps):
        tcp_pose_interpolated = tcp_poses_interpolated[step]

        # 使用 PID 控制器更新实际位置
        for axis in range(6):
            target_position = tcp_pose_interpolated[axis]
            current_position = tcp_pose_current[axis]
            control_output = pid_controllers[axis].compute(target_position, current_position, dt)

            tcp_pose_current[axis] += control_output * dt
        print('interpolated_poses', tcp_pose_current)
        # 使用逆运动学计算关节角度
        joints_target = [0.0] * 7
        ret = DianaApi.inverse(tcp_pose_current, joints_target, ipAddress)
        if ret == 0:
            print(f"Error in DianaApi.inverse at step {step}, Error Code:", ret)
            return False

        # 将关节角度发送给机械臂
        # DianaApi.servoJ(joints_target, 2, 0.03,300, ipAddress)
        #DianaApi.moveJ(joints_target,0.3,0.2,ipAddress)
        # 等待一个时间步长
        #DianaApi.wait_move()
        #time.sleep(dt)

    return True

