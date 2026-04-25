import os
import time
import math

# 设置DLL目录
dll_directory = os.path.join(os.path.dirname(__file__), 'bin')
os.environ['PATH'] = dll_directory

# 导入DianaApi模块
from bin import DianaApi

# 初始化参数
velocity = 0.2
acceleration = 0.3
ipAddress = '192.168.10.75'
netInfo = ('192.168.10.75', 0, 0, 0, 0, 0)

# 初始化服务器
DianaApi.initSrv(netInfo)

# 解除刹车
DianaApi.releaseBrake(ipAddress)

# 定义初始位置和目标位置
poses = [-0.02945703931962873, 0.014956500826876162, -0.011516653566979507, 2.3611802557437978, -0.1835262438861327, -0.7736414976244061, 0.503289726377274]
target = [0.6363566240453085, 0.26979626444488086, 0.18326186498732178, 0.08947633925753617, -0.4811303914398447, 0.6324288045963838]

# 移动到初始位置
DianaApi.moveJ(poses, velocity, acceleration, ipAddress)
DianaApi.wait_move()

# 开始计时
start_t = time.time()

try:
    while True:
        t = time.time() - start_t

        # 调用servoL_ex函数
        ret = DianaApi.servoL_ex(target, t=1, ah_t=0.03, gain=200, scale=1,ipAddress='192.168.10.75')

        # 检查返回值
        if ret < 0:
            print(f"Error in servoL_ex function at time {t}: ERROR_CODE_SENDTO_FAIL")
            break

        # 检查时间是否超过10秒
        if t>2:
            target=None
        if t > 10:
            break

        # 增加发送间隔
        time.sleep(0.005)  # 适当增加延迟

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # 停止运动
    DianaApi.stop(ipAddress)

    # 启用刹车
    DianaApi.holdBrake(ipAddress)

    # 销毁服务器
    DianaApi.destroySrv(ipAddress)



