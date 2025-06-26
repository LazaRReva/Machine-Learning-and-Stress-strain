import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 读取Excel数据
def read_data_from_excel(file_path):
    df = pd.read_excel(file_path)
    sigma = df['sigma'].values
    epsilon = df['epsilon'].values
    return sigma, epsilon

class TensorDataset(Dataset):
    def __init__(self, sigma, epsilon):
        self.sigma = torch.tensor(sigma, dtype=torch.float32)
        self.epsilon = torch.tensor(epsilon, dtype=torch.float32)

    def __len__(self):
        return len(self.sigma)

    def __getitem__(self, idx):
        return self.sigma[idx], self.epsilon[idx]


class FCModel(nn.Module):
    def __init__(self):
        super(FCModel, self).__init__()
        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 数据加载和预处理
file_path = 'test001.xlsx'  # 使用上传的Excel文件路径
sigma, epsilon = read_data_from_excel(file_path)
dataset = TensorDataset(sigma, epsilon)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 模型初始化
model = FCModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 20
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (sigma_batch, epsilon_batch) in enumerate(dataloader):
        # 前向传播
        outputs = model(sigma_batch)
        loss = criterion(outputs, epsilon_batch)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:  # 每100个batch输出一次损失
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}], Loss: {running_loss / 100:.4f}')
            running_loss = 0.0

print('训练完成')

# 预测和可视化
model.eval()
sigma_test, epsilon_test = read_data_from_excel(file_path)  # 这里假设你使用相同的文件进行测试
sigma_test_tensor = torch.tensor(sigma_test, dtype=torch.float32)
epsilon_test_tensor = torch.tensor(epsilon_test, dtype=torch.float32)

with torch.no_grad():
    predictions = model(sigma_test_tensor).numpy()

# 绘制结果图像
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(epsilon_test[0], label='Actual epsilon')
plt.legend()
plt.title('Actual epsilon')

plt.subplot(1, 2, 2)
plt.plot(predictions[0], label='Predicted epsilon')
plt.legend()
plt.title('Predicted epsilon')

plt.show()
