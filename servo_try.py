import os
dll_directory = os.path.join(os.path.dirname(__file__), 'bin')
os.environ['PATH'] = dll_directory
from bin import DianaApi
import time
import math
import pids_curve
velocity = 0.2
acceleration = 0.3
ipAddress='192.168.10.75'
netInfo = ('192.168.10.75', 0, 0, 0, 0, 0)
DianaApi.initSrv(netInfo)
DianaApi.releaseBrake(ipAddress)
pi=3.141592653
poses=[-0.02945703931962873, 0.014956500826876162, -0.011516653566979507, 2.3611802557437978, -0.1835262438861327, -0.7736414976244061, 0.503289726377274]
target = [0.508609707017957, 0.05934514303553257, 0.3124935331233286, 0.03601568717053123, -0.15302789485019144, 0.5417924801539491]
DianaApi.moveJ(poses, velocity, acceleration, ipAddress)
DianaApi.wait_move()
start_t = time.time()
#for i in range(50):
while True:

    t = time.time() - start_t
  # kRadius = 0.01 # m
  #
  # angle = math.pi / 4 * (1 - math.cos(math.pi / 5.0 * t))
  # delta_x = kRadius * math.sin(angle)   #X方向位移
  # delta_z = kRadius * (math.cos(angle) - 1)  #Z方向位移

  # target[0] += delta_x
  # DianaApi.moveLToPose(target, velocity, acceleration, ipAddress)
  # DianaApi.wait_move()
  # target[2] += delta_z
    #ret = DianaApi.servoL_ex(target, t=0.1, ah_t=0.03, gain=200, scale = 1, realiable = False, ipAddress='192.168.10.75')
    ret=pids_curve.custom_servoL(target)
    time.sleep(0.2)
    if ret < 0:
       break
    # if t>10:
    #    break
    #time.sleep(0.1)

DianaApi.stop(ipAddress)
DianaApi.holdBrake(ipAddress)
DianaApi.destroySrv(ipAddress)
