import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import os
import json
import shutil

class DataSaver(Node):
    def __init__(self):
        super().__init__('data_saver')
        
        # 删除文件夹中的所有内容
        self.image_save_dir = '/home/dfrobot/isaac/IsaacSim-ros_workspaces/joystick_ws/saved_img'
        self.json_save_dir = '/home/dfrobot/isaac/IsaacSim-ros_workspaces/joystick_ws/saved_json'
        self.clear_directory(self.image_save_dir)
        self.clear_directory(self.json_save_dir)
        
        # 创建文件夹
        os.makedirs(self.image_save_dir, exist_ok=True)
        os.makedirs(self.json_save_dir, exist_ok=True)

        self.image_subscription = self.create_subscription(
            Image,
            '/rgb',
            self.image_callback,
            10)
        self.odom_subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10)
        self.cmd_vel_subscription = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10)

        self.bridge = CvBridge()
        self.image_count = 1
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.current_image = None
        self.data = {
            'odom_twist_linear_x': None,
            'odom_twist_angular_z': None,
            'odom_pose_position': None,
            'odom_pose_orientation': None,
            'cmd_vel_linear_x': None,
            'cmd_vel_angular_z': None  # 修改为记录angular.z
        }

    def clear_directory(self, directory):
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory, exist_ok=True)

    def image_callback(self, msg):
        self.current_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

    def odom_callback(self, msg):
        self.data['odom_twist_linear_x'] = msg.twist.twist.linear.x
        self.data['odom_twist_angular_z'] = msg.twist.twist.angular.z
        self.data['odom_pose_position'] = {
            'x': msg.pose.pose.position.x,
            'y': msg.pose.pose.position.y,
            'z': msg.pose.pose.position.z
        }
        self.data['odom_pose_orientation'] = {
            'x': msg.pose.pose.orientation.x,
            'y': msg.pose.pose.orientation.y,
            'z': msg.pose.pose.orientation.z,
            'w': msg.pose.pose.orientation.w
        }

    def cmd_vel_callback(self, msg):
        self.data['cmd_vel_linear_x'] = msg.linear.x
        self.data['cmd_vel_angular_z'] = msg.angular.z  # 修改为记录angular.z

    def timer_callback(self):
        if self.current_image is not None:
            linear_velocity = self.data['cmd_vel_linear_x']
            angular_velocity = self.data['cmd_vel_angular_z']  # 修改为angular.z
            if linear_velocity > 0.1 or angular_velocity > 0.2:
                image_filename = os.path.join(self.image_save_dir, f'{self.image_count}.jpg')
                json_filename = os.path.join(self.json_save_dir, f'{self.image_count}.json')
                cv2.imwrite(image_filename, self.current_image)
                with open(json_filename, 'w') as json_file:
                    json.dump(self.data, json_file)
                self.get_logger().info(f'Saved image {image_filename} and data {json_filename}')
                self.image_count += 1

def main(args=None):
    rclpy.init(args=args)
    data_saver = DataSaver()
    rclpy.spin(data_saver)
    data_saver.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()