import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import scipy.io

# 读取 .mat 文件
mat_file = 'Use_data.mat'
mat_data = scipy.io.loadmat(mat_file)

# 从 .mat 文件中提取数据
epsilon_cell = mat_data['epsilon_t_use']
sigma_cell = mat_data['sigma_use']

# 初始化空列表来存储所有的 3x3 矩阵
epsilon_list = []
sigma_list = []

# 遍历单元格数组并提取每个 3x3 矩阵
for i in range(epsilon_cell.shape[0]):
    for j in range(epsilon_cell.shape[1]):
        epsilon_list.append(epsilon_cell[i, j])
        sigma_list.append(sigma_cell[i, j])

# 将列表转换为 numpy 数组，并调整形状
epsilon = np.array(epsilon_list).reshape(-1, 3, 3)
sigma = np.array(sigma_list).reshape(-1, 3, 3)

# 转换数据类型为 PyTorch 张量
epsilon = torch.tensor(epsilon, dtype=torch.float32)
sigma = torch.tensor(sigma, dtype=torch.float32)

# 检查数据形状
print(f'epsilon shape: {epsilon.shape}')
print(f'sigma shape: {sigma.shape}')

# Step 2: 定义数据集类
class StressStrainDataset(Dataset):
    def __init__(self, sigma, epsilon):
        self.sigma = sigma  # 将应力数据存储在实例变量中
        self.epsilon = epsilon  # 将应变数据存储在实例变量中

    def __len__(self):
        return len(self.sigma)  # 返回数据集的大小

    def __getitem__(self, idx):
        # return torch.tensor(self.epsilon[idx], dtype=torch.float32), torch.tensor(self.sigma[idx], dtype=torch.float32)  # 返回指定索引的应力和应变数据，并转换为PyTorch张量
        return self.epsilon[idx], self.sigma[idx]
# Step 3: 定义FCN模型
class StressStrainFCN(nn.Module):
    def __init__(self, input_size):
        super(StressStrainFCN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # 定义第一个全连接层，输入维度为input_size，输出维度为128
        self.fc2 = nn.Linear(128, 64)  # 定义第二个全连接层，输入维度为128，输出维度为64
        self.fc3 = nn.Linear(64, input_size)  # 定义第三个全连接层，输入维度为64，输出维度为input_size
        self.relu = nn.ReLU()  # 定义ReLU激活函数

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 将输入展平成一维向量
        x = self.relu(self.fc1(x))  # 第一个全连接层和ReLU激活
        x = self.relu(self.fc2(x))  # 第二个全连接层和ReLU激活
        x = self.fc3(x)  # 第三个全连接层（没有激活函数）
        return x  # 返回网络输出

# Step 4: 训练模型
# 假设sigma和epsilon已经以numpy数组形式存在

dataset = StressStrainDataset(sigma, epsilon)  # 创建应力应变数据集对象
train_size = int(0.8 * len(dataset))  # 设置训练集的大小
val_size = len(dataset) - train_size  # 设置验证集的大小
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])  # 随机划分数据集

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # 创建训练数据加载器
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)  # 创建验证数据加载器

input_size = sigma.shape[1] * sigma.shape[2]  # 计算输入张量展平后的大小
model = StressStrainFCN(input_size)  # 创建FCN模型实例
criterion = nn.MSELoss()  # 定义均方误差损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 创建Adam优化器

num_epochs = 15  # 定义训练的epoch数量
for epoch in range(num_epochs):
    model.train()  # 将模型设置为训练模式
    running_loss = 0.0  # 初始化累计损失为0
    for inputs, targets in train_loader:  # 循环遍历训练数据集的每个批次
        optimizer.zero_grad()  # 清零优化器的梯度
        outputs = model(inputs)  # 通过模型前向传播计算输出
        loss = criterion(outputs, targets.view(targets.size(0), -1))  # 计算损失函数，注意将目标张量展平为一维
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新模型参数
        running_loss += loss.item()  # 累加当前批次的损失值

    val_loss = 0.0  # 初始化验证集累计损失为0
    model.eval()  # 将模型设置为评估模式
    with torch.no_grad():  # 在此上下文管理器内，停止自动求导机制
        for inputs, targets in val_loader:  # 循环遍历验证数据集的每个批次
            outputs = model(inputs)  # 通过模型前向传播计算输出
            loss = criterion(outputs, targets.view(targets.size(0), -1))  # 计算损失函数，注意将目标张量展平为一维
            val_loss += loss.item()  # 累加当前批次的损失值

    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {running_loss/len(train_loader)}, Validation Loss: {val_loss/len(val_loader)}")  # 输出当前epoch的训练损失和验证损失

# Step 5: 使用模型进行预测
model.eval()  # 将模型设置为评估模式

with torch.no_grad():  # 在此上下文管理器内，停止自动求导机制
    # test_sigma_ones = np.ones((1, sigma.shape[1], sigma.shape[2]))  # 创建一个形状为 (10, 3, 3) 的 numpy 数组，包含10个样本的测试应力数据，每个样本是一个 3x3 矩阵
    # test_sigma_tensor = torch.tensor(test_sigma_ones, dtype=torch.float32)  # 将 numpy 数组 test_sigma 转换为 PyTorch 张量，并将数据类型设为 float32
    test_sigma_specific = np.array([[-0.3563, -0.0537, 0], [-0.0537, 0.4101, 0], [0, 0, 0]])
    test_sigma_specific = test_sigma_specific.reshape(1, 3, 3)
    test_sigma_specific_tensor = torch.tensor(test_sigma_specific, dtype=torch.float32)

    predictions = model(test_sigma_specific_tensor)  # 通过模型前向传播计算预测结果
    predictions_reshaped = predictions.view(predictions.size(0), 3, 3)  # 将模型输出的预测结果 predictions 重新形状为与原始输入形状相同的 3x3 矩阵
    print(predictions_reshaped)  # 打印重新形状后的预测结果

