import torch
import torch.nn as nn
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

# 定义全连接神经网络（FCN）模型
class StressStrainFCN(nn.Module):
    def __init__(self, input_size=9):  # 确保输入大小为9
        super(StressStrainFCN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)  # 增加神经元数量，输入大小应为9
        self.fc2 = nn.Linear(256, 128)  # 增加神经元数量
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, input_size)  # 输出大小应与输入大小一致
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        # x = x.contiguous().view(x.size(0), -1)  # 确保展平后的维度为[batch_size, 9]
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.fc4(x)  # 通过增加的全连接层
        return x

# 加载训练好的模型
def load_model(model_path, input_size=9):
    model = StressStrainFCN(input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Von Mises Stress 计算
def calculate_von_mises_stress(sigma):
    s11, s22, s33 = sigma[:, 0, 0], sigma[:, 1, 1], sigma[:, 2, 2]
    s12, s13, s23 = sigma[:, 0, 1], sigma[:, 0, 2], sigma[:, 1, 2]
    von_mises_stress = np.sqrt(0.5 * ((s11 - s22) ** 2 + (s22 - s33) ** 2 + (s33 - s11) ** 2) + 3 * (s12 ** 2 + s13 ** 2 + s23 ** 2))
    return von_mises_stress

# Effective Strain 计算
def calculate_effective_strain(epsilon):
    e11, e22, e33 = epsilon[:, 0, 0], epsilon[:, 1, 1], epsilon[:, 2, 2]
    e12, e13, e23 = epsilon[:, 0, 1], epsilon[:, 0, 2], epsilon[:, 1, 2]
    effective_strain = np.sqrt(2/3 * (e11**2 + e22**2 + e33**2 + 2 * (e12**2 + e13**2 + e23**2)))
    return effective_strain

# 加载 .mat 文件
mat_file = 'Use_data_new.mat'
mat_data = scipy.io.loadmat(mat_file)

# 提取 epsilon_t_use 数据
epsilon = mat_data['epsilon_t_use']

# 打印 epsilon 形状以进行检查
print(f'epsilon shape before processing: {epsilon.shape}')

# 转换为 PyTorch 张量
epsilon = torch.tensor(epsilon, dtype=torch.float32)

# 合并 batch_size 和 91 这两个维度
epsilon = epsilon.reshape(-1, 3, 3)  # 形状变为 [292 * 91, 3, 3]

# 确保其在内存中是连续的并展平
epsilon = epsilon.contiguous().view(epsilon.size(0), -1)  # [292 * 91, 9]

# 加载训练好的模型
model = load_model('trained_model.pth', input_size=9)

# 使用模型进行预测
with torch.no_grad():
    sigma_predict = model(epsilon)

# 转换预测结果为 numpy 数组并重新调整形状为 3x3
sigma_predict = sigma_predict.numpy().reshape(-1, 3, 3)

# 计算 Von Mises Stress 和 Effective Strain
von_mises_stress = calculate_von_mises_stress(sigma_predict)
effective_strain = calculate_effective_strain(epsilon.view(-1, 3, 3).numpy())

# 绘制点阵图和折线图
plt.figure(figsize=(12, 6))

# 点阵图
plt.subplot(1, 2, 1)
plt.scatter(effective_strain, von_mises_stress, alpha=0.5)
plt.title('Scatter Plot: Effective Strain vs Von Mises Stress')
plt.xlabel('Effective Strain')
plt.ylabel('Von Mises Stress')

# 折线图
plt.subplot(1, 2, 2)
plt.plot(effective_strain, von_mises_stress, alpha=0.75)
plt.title('Line Plot: Effective Strain vs Von Mises Stress')
plt.xlabel('Effective Strain')
plt.ylabel('Von Mises Stress')

plt.tight_layout()
plt.show()
