import scipy.io
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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

# 读取 .mat 文件
mat_file = 'Use_data_new.mat'
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
        self.fc1 = nn.Linear(input_size, 256)  # 增加神经元数量
        self.fc2 = nn.Linear(256, 128)  # 增加神经元数量
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, input_size)  # 增加一个全连接层
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)  # 使用 contiguous() 确保张量在内存中是连续的，然后再展平成一维向量
        # x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)  # 通过增加的全连接层
        return x


# 训练单个模型的函数
def train_model(train_loader, val_loader, input_size, num_epochs=25, learning_rate=0.000001):
    model = StressStrainFCN(input_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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
num_models = 10
models = []
for i in range(num_models):
    print(f"Training model {i+1}/{num_models}")
    set_seed(i)  # 设置不同的种子以确保模型多样性
    model = train_model(train_loader, val_loader, input_size=9, num_epochs=20, learning_rate=0.001)
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

np.set_printoptions(precision=4, suppress=True)

# def filter_predictions(predictions):
#     predictions = predictions.numpy()  # 转换为 numpy 数组
#     predictions[np.abs(predictions) < 1e-3] = 0  # 将绝对值小于 1e-4 的元素置为 0
#     return predictions


test_epsilon_t = np.array([[-0.0138, 0.0011, 0], [0.0011, 0.0149, 0], [0, 0, 0]])
test_epsilon_t = test_epsilon_t.reshape(1, 3, 3)
test_epsilon_t_tensor = torch.tensor(test_epsilon_t, dtype=torch.float32)

# 进行集成预测
ensemble_prediction = ensemble_predict(models, test_epsilon_t_tensor)
ensemble_prediction = ensemble_prediction.reshape(3, 3)
# ensemble_prediction = filter_predictions(ensemble_prediction)
print("Ensemble Predictions for test_epsilon_t reshaped and denormalized:\n", ensemble_prediction)

# 保存训练好的模型
model_path = 'trained_model.pth'
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")
