import os
import sys
dll_directory = os.path.join(os.path.dirname(__file__), 'bin')
os.environ['PATH'] = dll_directory
import math
from bin import DianaApi
ipAddress = '192.168.10.75'
netInfo=(ipAddress, 0, 0, 0)
DianaApi.initSrv(netInfo)
#DianaApi.releaseBrake(ipAddress)
tool1 = (-28.801, 4.771, -226.509, 0, 0, 0)
#工具坐标系设置
matrix = (-0.0288008,0.00477148,-0.226509,0.0,0.0,0.0)
DianaApi.setDefaultActiveTcpPose(tool1,' 192.168.10.75')
pose = [0.0] * 6
DianaApi.getTcpPos(pose, ipAddress)
print(f"机械臂原始pose:{pose}")
DianaApi.destroySrv('192.168.10.75')