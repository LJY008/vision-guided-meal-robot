import os
dll_directory = os.path.join(os.path.dirname(__file__), 'bin')
os.environ['PATH'] = dll_directory

from bin import DianaApi
ipAddress = '192.168.10.75'
netInfo = ('192.168.10.75', 0, 0, 0, 0, 0)
DianaApi.initSrv(netInfo)

pose = [0.0] * 6
#DianaApi.setDefaultActiveTcpPose(pose)
DianaApi.setDefaultActiveTcpPose(pose,ipAddress)
DianaApi.getTcpPos(pose,ipAddress)
print(f"机械臂pose:{pose}")

#工具坐标系设置
matrix = [-0.0288008,0.00477148,-0.226509,0,0,0]
#DianaApi.setDefaultActiveTcpPose(matrix)
DianaApi.setDefaultActiveTcpPose(matrix,ipAddress)
DianaApi.getTcpPos(matrix,ipAddress)
print(f"机械臂pose:{matrix}")
DianaApi.destroySrv('192.168.10.75')