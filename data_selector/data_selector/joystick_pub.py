import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import pygame
import time

class TeleopJoystick(Node):
    def __init__(self):
        super().__init__('teleop_joystick')
        
        # 创建一个发布者，发布到 /cmd_vel 主题
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)

        # 初始化线速度和角速度
        self.a = 0.0  # 线速度
        self.b = 0.0  # 角速度
        
        # 设置定时器周期，定时发布消息
        self.timer = self.create_timer(0.1, self.timer_callback)  # 每0.1秒发布一次

        # 初始化pygame
        pygame.init()
        pygame.joystick.init()

        # 检查是否有手柄连接
        if pygame.joystick.get_count() == 0:
            self.get_logger().error("没有检测到手柄。")
            exit()

        # 选择第一个手柄
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()

        self.get_logger().info(f"手柄已连接：{self.joystick.get_name()}")

    def timer_callback(self):
        # 读取手柄数据
        self.read_joystick_data()

        # 创建 Twist 消息并设置线速度和角速度
        msg = Twist()
        msg.linear.x = self.a  # 设置线速度（沿 x 轴）
        msg.angular.z = self.b  # 设置角速度（绕 z 轴）

        # 发布消息
        self.cmd_vel_publisher.publish(msg)
        self.get_logger().info(f"Publishing: Linear.x = {self.a}, Angular.z = {self.b}")

    def read_joystick_data(self):
        # 获取手柄事件
        pygame.event.pump()
        
        # 获取左右摇杆的数据
        left_joystick_y = self.joystick.get_axis(1)  # 左摇杆Y轴
        right_joystick_x = self.joystick.get_axis(3)  # 右摇杆X轴
        
        # 更新线速度和角速度
        self.a = -left_joystick_y /3  # 线速度（取反是因为手柄Y轴向上为负）
        self.b = -right_joystick_x *2  # 角速度

def main(args=None):
    rclpy.init(args=args)

    # 创建 TeleopJoystick 节点实例
    node = TeleopJoystick()

    # 保持节点运行
    rclpy.spin(node)

    # 关闭 ROS 2
    rclpy.shutdown()
    pygame.quit()

if __name__ == '__main__':
    main()