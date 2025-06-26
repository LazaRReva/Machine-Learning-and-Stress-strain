import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import pandas as pd

def read_data_from_excel(file_path):
    df = pd.read_excel(file_path)
    sigma = df[
        ['sigma11', 'sigma12', 'sigma13', 'sigma21', 'sigma22', 'sigma23', 'sigma31', 'sigma32', 'sigma33']].values
    epsilon = df[
        ['epsilon11', 'epsilon12', 'epsilon13', 'epsilon21', 'epsilon22', 'epsilon23', 'epsilon31', 'epsilon32','epsilon33']].values
    return sigma, epsilon


class StressStrainDataset(Dataset):
    def __init__(self, sigma, epsilon):
        self.sigma = sigma
        self.epsilon = epsilon

    def __len__(self):
        return len(self.sigma)

    def __getitem__(self, idx):
        return torch.tensor(self.epsilon[idx], dtype=torch.float32), torch.tensor(self.sigma[idx], dtype=torch.float32)

class StressStrainCNN(nn.Module):
    def __init__(self):
        super(StressStrainCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # flatten the tensor
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 假设sigma和epsilon已经以numpy数组形式存在
file_path = 'test001.xlsx'  # 使用上传的Excel文件路径
sigma, epsilon = read_data_from_excel(file_path)
dataset = StressStrainDataset(sigma, epsilon)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = StressStrainCNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs = inputs.unsqueeze(1)  # 添加通道维度
        targets = targets.unsqueeze(1)  # 添加通道维度

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    val_loss = 0.0
    model.eval()
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.unsqueeze(1)
            targets = targets.unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {running_loss/len(train_loader)}, Validation Loss: {val_loss/len(val_loader)}")

model.eval()
with torch.no_grad():
    test_sigma = np.random.rand(10, 7, 7)  # 示例测试数据
    test_epsilon = torch.tensor(test_sigma, dtype=torch.float32).unsqueeze(1)
    predictions = model(test_epsilon)
    print(predictions)
