import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split

# 数据路径
image_dir = '/home/dfrobot/isaac/IsaacSim-ros_workspaces/joystick_ws/saved_img'
json_dir = '/home/dfrobot/isaac/IsaacSim-ros_workspaces/joystick_ws/saved_json'

# 加载数据
def load_data(image_dir, json_dir):
    images = []
    linear_velocities = []
    angular_velocities = []

    for filename in sorted(os.listdir(image_dir), key=lambda x: int(os.path.splitext(x)[0])):
        image_path = os.path.join(image_dir, filename)
        json_path = os.path.join(json_dir, os.path.splitext(filename)[0] + '.json')

        if os.path.exists(image_path) and os.path.exists(json_path):
            image = cv2.imread(image_path)
            image = cv2.resize(image, (256, 256))  # 调整图像尺寸为 (256, 256)
            with open(json_path, 'r') as json_file:
                data = json.load(json_file)
                linear_velocity = data['cmd_vel_linear_x']
                angular_velocity = data['cmd_vel_angular_z']

            images.append(image)
            linear_velocities.append(linear_velocity)
            angular_velocities.append(angular_velocity)

    return np.array(images), np.array(linear_velocities), np.array(angular_velocities)

# 加载数据
images, linear_velocities, angular_velocities = load_data(image_dir, json_dir)

# 数据预处理
images = images / 255.0  # 归一化
linear_velocities = np.array(linear_velocities)
angular_velocities = np.array(angular_velocities)

# 划分训练集和测试集
X_train, X_test, y_train_linear, y_test_linear, y_train_angular, y_test_angular = train_test_split(
    images, linear_velocities, angular_velocities, test_size=0.2, random_state=42)

# 自定义数据集
class AutonomousCarDataset(Dataset):
    def __init__(self, images, linear_velocities, angular_velocities):
        self.images = images
        self.linear_velocities = linear_velocities
        self.angular_velocities = angular_velocities

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].transpose((2, 0, 1))  # 转换为 (C, H, W)
        linear_velocity = self.linear_velocities[idx]
        angular_velocity = self.angular_velocities[idx]
        return torch.tensor(image, dtype=torch.float32), torch.tensor([linear_velocity, angular_velocity], dtype=torch.float32)

# 创建数据集和数据加载器
train_dataset = AutonomousCarDataset(X_train, y_train_linear, y_train_angular)
test_dataset = AutonomousCarDataset(X_test, y_test_linear, y_test_angular)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 定义模型
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

# 初始化模型、损失函数和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutonomousCarModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

# 评估模型
model.eval()
test_loss = 0.0
with torch.no_grad():
    for images, targets in test_loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)
        test_loss += loss.item() * images.size(0)
test_loss /= len(test_loader.dataset)
print(f'Test Loss: {test_loss:.4f}')

# 保存模型
torch.save(model.state_dict(), '/home/dfrobot/isaac/IsaacSim-ros_workspaces/joystick_ws/model/autonomous_car_model.pth')