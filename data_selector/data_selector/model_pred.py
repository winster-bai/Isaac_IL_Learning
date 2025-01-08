import torch
import torch.nn as nn
import cv2
import numpy as np
import time

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

# 初始化模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutonomousCarModel().to(device)

# 加载模型参数
model.load_state_dict(torch.load('/home/dfrobot/isaac/IsaacSim-ros_workspaces/joystick_ws/model/autonomous_car_model.pth'))

# 将模型设置为评估模式
model.eval()

# 加载并预处理输入图像
image_path = '/home/dfrobot/isaac/IsaacSim-ros_workspaces/joystick_ws/saved_img/75.jpg'
image = cv2.imread(image_path)

# 开始计时
start_time = time.time()

# 预处理图像
image = cv2.resize(image, (256, 256))  # 调整图像尺寸为 (256, 256)
image = image / 255.0  # 归一化
image = image.transpose((2, 0, 1))  # 转换为 (C, H, W)
image = np.expand_dims(image, axis=0)  # 增加批次维度
image = torch.tensor(image, dtype=torch.float32).to(device)

# 进行预测
with torch.no_grad():
    output = model(image)
    linear_velocity, angular_velocity = output[0].cpu().numpy()

# 结束计时
end_time = time.time()

# 打印处理时间
processing_time = end_time - start_time
print(f'Processing time: {processing_time:.4f} seconds')

print(f'Predicted linear velocity: {linear_velocity}')
print(f'Predicted angular velocity: {angular_velocity}')