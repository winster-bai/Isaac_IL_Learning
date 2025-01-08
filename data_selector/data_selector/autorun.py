import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import torch
import torch.nn as nn
import cv2
import numpy as np

# 定义模型结构
class AutonomousCarModel(nn.Module):
    def __init__(self):
        super(AutonomousCarModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * (256 // 8) * (256 // 8), 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # 动态计算形状
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class TeleopJoystick(Node):
    def __init__(self):
        super().__init__('teleop_joystick')
        
        # 创建一个发布者，发布到 /cmd_vel 主题
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)

        # 创建一个订阅者，订阅 /rgb 主题
        self.image_subscription = self.create_subscription(
            Image,
            '/rgb',
            self.image_callback,
            10)

        # 初始化模型
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AutonomousCarModel().to(self.device)
        self.model.load_state_dict(torch.load('/home/dfrobot/isaac/IsaacSim-ros_workspaces/joystick_ws/model/autonomous_car_model.pth'))
        self.model.eval()

        # 初始化cv_bridge
        self.bridge = CvBridge()

        self.get_logger().info("TeleopJoystick节点已启动")

    def image_callback(self, msg):
        # 将ROS图像消息转换为OpenCV图像
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        # 预处理图像
        image = cv2.resize(cv_image, (256, 256))  # 调整图像尺寸为 (256, 256)
        image = image / 255.0  # 归一化
        image = image.transpose((2, 0, 1))  # 转换为 (C, H, W)
        image = np.expand_dims(image, axis=0)  # 增加批次维度
        image = torch.tensor(image, dtype=torch.float32).to(self.device)

        # 进行预测
        with torch.no_grad():
            output = self.model(image)
            linear_velocity, angular_velocity = output[0].cpu().numpy()

        # 确保线速度和角速度是float类型
        linear_velocity = float(linear_velocity)
        angular_velocity = float(angular_velocity)

        # 创建 Twist 消息并设置线速度和角速度
        msg = Twist()
        msg.linear.x = linear_velocity  # 设置线速度（沿 x 轴）
        msg.angular.z = angular_velocity  # 设置角速度（绕 z 轴）

        # 发布消息
        self.cmd_vel_publisher.publish(msg)
        self.get_logger().info(f"Publishing: Linear.x = {linear_velocity}, Angular.z = {angular_velocity}")

def main(args=None):
    rclpy.init(args=args)

    # 创建 TeleopJoystick 节点实例
    node = TeleopJoystick()

    # 保持节点运行
    rclpy.spin(node)

    # 关闭 ROS 2
    rclpy.shutdown()

if __name__ == '__main__':
    main()