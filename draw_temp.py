import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, random_split

# 设置随机种子
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 设置种子
set_seed(42)

# 加载 .mat 文件
mat_file = 'Use_data_new.mat'
mat_data = scipy.io.loadmat(mat_file)

# 提取 epsilon_t_use 和 sigma_use 数据
epsilon = mat_data['epsilon_t_use']
sigma = mat_data['sigma_use']

# 打印 epsilon 形状以进行检查
print(f'epsilon shape before processing: {epsilon.shape}')

# 转换为 PyTorch 张量
epsilon = torch.tensor(epsilon, dtype=torch.float32)
sigma = torch.tensor(sigma, dtype=torch.float32)

# 合并 batch_size 和 91 这两个维度
epsilon = epsilon.reshape(-1, 3, 3)  # 使用 reshape 代替 view
sigma = sigma.reshape(-1, 3, 3)

# 定义数据集类
class StressStrainDataset(Dataset):
    def __init__(self, sigma, epsilon):
        self.sigma = sigma
        self.epsilon = epsilon

    def __len__(self):
        return len(self.sigma)

    def __getitem__(self, idx):
        return self.epsilon[idx], self.sigma[idx]

# 创建数据集对象
dataset = StressStrainDataset(sigma, epsilon)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 定义全连接神经网络（FCN）模型
class StressStrainFCN(nn.Module):
    def __init__(self, input_size=9):
        super(StressStrainFCN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, input_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.reshape(x.size(0), -1)  # 确保展平后的维度为[batch_size, 9]
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# 训练单个模型的函数
def train_model(train_loader, val_loader, input_size, num_epochs=10, learning_rate=0.0001, weight_decay=0.001):
    model = StressStrainFCN(input_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.view(targets.size(0), -1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets.view(targets.size(0), -1))
                val_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {running_loss/len(train_loader)}, Validation Loss: {val_loss/len(val_loader)}")

    return model

# 训练多个模型
num_models = 3
models = []
for i in range(num_models):
    print(f"Training model {i+1}/{num_models}")
    set_seed(i)  # 设置不同的种子以确保模型多样性
    model = train_model(train_loader, val_loader, input_size=9, num_epochs=15, learning_rate=0.001, weight_decay=0.001)
    models.append(model)

# 模型评估与预测
def ensemble_predict(models, test_input):
    with torch.no_grad():
        predictions = []
        for model in models:
            model.eval()
            prediction = model(test_input)
            predictions.append(prediction.numpy())
        predictions = np.array(predictions)
        mean_prediction = np.mean(predictions, axis=0)
        return mean_prediction

# 计算 Von Mises Stress
def calculate_von_mises_stress(sigma):
    s11, s22, s33 = sigma[:, 0, 0], sigma[:, 1, 1], sigma[:, 2, 2]
    s12, s13, s23 = sigma[:, 0, 1], sigma[:, 0, 2], sigma[:, 1, 2]
    von_mises_stress = np.sqrt(0.5 * ((s11 - s22) ** 2 + (s22 - s33) ** 2 + (s33 - s11) ** 2) + 3 * (s12 ** 2 + s13 ** 2 + s23 ** 2))
    return von_mises_stress

# 计算 Effective Strain
def calculate_effective_strain(epsilon):
    e11, e22, e33 = epsilon[:, 0, 0], epsilon[:, 1, 1], epsilon[:, 2, 2]
    e12, e13, e23 = epsilon[:, 0, 1], epsilon[:, 0, 2], epsilon[:, 1, 2]
    effective_strain = np.sqrt(2/3 * (e11**2 + e22**2 + e33**2 + 2 * (e12**2 + e13**2 + e23**2)))
    return effective_strain


# 绘制点阵图和实际值对比图
def plot_results(models, epsilon, sigma):
    epsilon = epsilon.reshape(-1, 3, 3)
    sigma = sigma.reshape(-1, 3, 3)

    with torch.no_grad():
        sigma_predict = ensemble_predict(models, epsilon.reshape(-1, 9))

    sigma_predict = sigma_predict.reshape(-1, 3, 3)

    # 计算 Von Mises Stress 和 Effective Strain
    von_mises_stress_predicted = calculate_von_mises_stress(sigma_predict)
    effective_strain = calculate_effective_strain(epsilon.numpy())

    # 计算实际的 Von Mises Stress 和 Effective Strain
    von_mises_stress_actual = calculate_von_mises_stress(sigma.numpy())

    plt.figure(figsize=(12, 6))

    # 点阵图: 预测值
    plt.subplot(1, 2, 1)
    plt.scatter(effective_strain, von_mises_stress_predicted, alpha=0.5, label='Predicted')
    plt.title('Scatter Plot: Effective Strain vs Predicted Von Mises Stress')
    plt.xlabel('Effective Strain')
    plt.ylabel('Von Mises Stress')
    plt.legend()

    # 点阵图: 实际值
    plt.subplot(1, 2, 2)
    plt.scatter(effective_strain, von_mises_stress_actual, alpha=0.5, label='Actual', color='blue')
    plt.title('Scatter Plot: Effective Strain vs Actual Von Mises Stress')
    plt.xlabel('Effective Strain')
    plt.ylabel('Von Mises Stress')
    plt.legend()

    plt.tight_layout()
    plt.show()


# 绘制图像
plot_results(models, epsilon, sigma)