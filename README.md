## 指令整合
Isaac启动
```bash
cd /home/dfrobot/.local/share/ov/pkg/isaac-sim-4.2.0
./isaac-sim.sh --/renderer/multiGpu/enabled=false
```

手柄操作
```bash
cd /home/dfrobot/isaac/IsaacSim-ros_workspaces/joystick_ws
source install/setup.sh
# 手柄数据发送节点
ros2 run data_selector joystick_pub 
```


数据采集
```bash
cd /home/dfrobot/isaac/IsaacSim-ros_workspaces/joystick_ws
source install/setup.sh
# 数据接收
ros2 run data_selector data_selector
```

自动驾驶
```bash
cd /home/dfrobot/isaac/IsaacSim-ros_workspaces/joystick_ws
source install/setup.sh
# 自动驾驶
ros2 run data_selector autorun
```