import pygame
import time

# 初始化pygame
pygame.init()

# 初始化手柄
pygame.joystick.init()

# 检查是否有手柄连接
if pygame.joystick.get_count() == 0:
    print("没有检测到手柄。")
    exit()

# 选择第一个手柄
joystick = pygame.joystick.Joystick(0)
joystick.init()

print("手柄已连接：", joystick.get_name())

# 定义读取摇杆数据的函数
def read_joystick_data():
    # 获取手柄事件
    pygame.event.pump()
    
    # 获取左右摇杆的数据
    left_joystick_x = joystick.get_axis(0)  # 左摇杆X轴
    left_joystick_y = joystick.get_axis(1)  # 左摇杆Y轴
    right_joystick_x = joystick.get_axis(3)  # 右摇杆X轴
    right_joystick_y = joystick.get_axis(4)  # 右摇杆Y轴
    
    # 打印摇杆数据
    print(f"左摇杆 (X, Y): ({left_joystick_x:.2f}, {left_joystick_y:.2f})")
    print(f"右摇杆 (X, Y): ({right_joystick_x:.2f}, {right_joystick_y:.2f})")

# 主循环
try:
    while True:
        read_joystick_data()
        time.sleep(0.1)  # 每100毫秒更新一次
except KeyboardInterrupt:
    print("程序结束")
finally:
    pygame.quit()
