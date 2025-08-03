import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ----------------- 辅助函数：欧拉角转换与模型定义 -----------------

def euler_to_rotation_matrix(phi1, Phi, phi2):
    """将Bunge欧拉角 (in radians) 转换为3x3的取向矩阵。"""
    c1, s1 = np.cos(phi1), np.sin(phi1);
    c_phi, s_phi = np.cos(Phi), np.sin(Phi);
    c2, s2 = np.cos(phi2), np.sin(phi2)
    g1 = np.array([[c1, s1, 0], [-s1, c1, 0], [0, 0, 1]]);
    g_phi = np.array([[1, 0, 0], [0, c_phi, s_phi], [0, -s_phi, c_phi]]);
    g2 = np.array([[c2, s2, 0], [-s2, c2, 0], [0, 0, 1]])
    return np.dot(g2, np.dot(g_phi, g1))


class AnisoFCNN(nn.Module):
    """ FCNN模型，已加入Dropout正则化以提升抗噪能力 """

    def __init__(self, input_size=12, output_size=3, dropout_rate=0.2):
        super(AnisoFCNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x): return self.network(x)


class VectorDataset(Dataset):
    def __init__(self, X, Y): self.X, self.Y = X, Y

    def __len__(self): return len(self.X)

    def __getitem__(self, idx): return self.X[idx], self.Y[idx]


def predict_stress_strain_curve(model, euler_angles_deg, scaler_x, scaler_y, strain_range, steps=100):
    """根据给定的、动态的应变范围，预测应力-应变曲线"""
    model.eval()
    phi1, Phi, phi2 = np.deg2rad(euler_angles_deg)
    rot_matrix_flat = euler_to_rotation_matrix(phi1, Phi, phi2).flatten()
    strain_xx_range = np.linspace(strain_range[0], strain_range[1], steps)
    input_vectors_unscaled = []
    for exx in strain_xx_range:
        strain_vector = np.array([exx, -0.3 * exx, 0])
        input_vectors_unscaled.append(np.concatenate((strain_vector, rot_matrix_flat)))

    input_vectors_scaled = scaler_x.transform(np.array(input_vectors_unscaled))
    with torch.no_grad():
        predicted_stress_scaled = model(torch.tensor(input_vectors_scaled, dtype=torch.float32))
    predicted_stress = scaler_y.inverse_transform(predicted_stress_scaled.numpy())
    return strain_xx_range, predicted_stress[:, 0]


# ----------------- 数据加载、清理与特征工程 -----------------

def clean_data_by_trendline(exx, sxx, std_dev_threshold=2.0):
    """ 使用多项式回归趋势线来识别并剔除噪声点 """
    if len(exx) < 3: return np.ones_like(exx, dtype=bool)  # 数据太少，不清理
    coeffs = np.polyfit(exx, sxx, 2);
    poly = np.poly1d(coeffs)
    sxx_trend = poly(exx);
    distances = np.abs(sxx - sxx_trend)
    distance_std = np.std(distances)
    good_indices = distances < (std_dev_threshold * distance_std)
    num_original = len(exx);
    num_cleaned = np.sum(good_indices)
    print(
        f"    趋势线去噪: 原始点数={num_original}, 清理后点数={num_cleaned}, 移除了{num_original - num_cleaned}个噪点。")
    return good_indices


def load_and_prepare_fcnn_data(mat_file, perform_cleaning=True):
    try:
        data = scipy.io.loadmat(mat_file)
        grain_data_list = data['Grains_Data_Cell']
        original_grains_with_euler = []
    except FileNotFoundError:
        print(f"错误: 找不到文件 {mat_file}。");
        return None, None, None

    all_X, all_Y = [], []
    print("开始加载数据并进行预处理...")
    for i in range(len(grain_data_list)):
        print(f"  处理晶粒 {i + 1}...")
        grain_struct = grain_data_list[i, 0]
        sxx_raw = grain_struct['Stress_xx'][0, 0].flatten();
        syy_raw = grain_struct['Stress_yy'][0, 0].flatten();
        sxy_raw = grain_struct['Stress_xy'][0, 0].flatten()
        exx_raw = grain_struct['Strain_xx'][0, 0].flatten();
        eyy_raw = grain_struct['Strain_yy'][0, 0].flatten();
        exy_raw = grain_struct['Strain_xy'][0, 0].flatten()
        phi1_raw = grain_struct['Euler_phi1'][0, 0].flatten();
        phi_cap_raw = grain_struct['Euler_Phi'][0, 0].flatten();
        phi2_raw = grain_struct['Euler_phi2'][0, 0].flatten()

        if perform_cleaning:
            clean_indices = clean_data_by_trendline(exx_raw, sxx_raw)
            sxx, syy, sxy = sxx_raw[clean_indices], syy_raw[clean_indices], sxy_raw[clean_indices]
            exx, eyy, exy = exx_raw[clean_indices], eyy_raw[clean_indices], exy_raw[clean_indices]
            phi1, phi_cap, phi2 = phi1_raw[clean_indices], phi_cap_raw[clean_indices], phi2_raw[clean_indices]
        else:
            print("    跳过数据去噪步骤。")
            sxx, syy, sxy = sxx_raw, syy_raw, sxy_raw
            exx, eyy, exy = exx_raw, eyy_raw, exy_raw
            phi1, phi_cap, phi2 = phi1_raw, phi_cap_raw, phi2_raw

        original_grains_with_euler.append({
            'exx': exx, 'sxx': sxx,
            'euler_angles': np.stack((phi1, phi_cap, phi2), axis=1)
        })

        strain_vectors = np.stack((exx, eyy, exy), axis=1);
        stress_vectors = np.stack((sxx, syy, sxy), axis=1)
        for j in range(len(sxx)):
            p1_rad, p_rad, p2_rad = np.deg2rad([phi1[j], phi_cap[j], phi2[j]])
            rot_matrix = euler_to_rotation_matrix(p1_rad, p_rad, p2_rad)
            all_X.append(np.concatenate((strain_vectors[j], rot_matrix.flatten())))
        all_Y.append(stress_vectors)

    X_final = np.array(all_X);
    Y_final = np.concatenate(all_Y, axis=0)
    print(f"数据处理完成。总有效数据点数: {len(X_final)}")
    return torch.tensor(X_final, dtype=torch.float32), torch.tensor(Y_final,
                                                                    dtype=torch.float32), original_grains_with_euler


# ----------------- 模型训练与评估 -----------------
def train_fcnn_model(X_scaled, Y_scaled, epochs, learning_rate):
    dataset = VectorDataset(X_scaled, Y_scaled);
    train_size = int(0.8 * len(dataset));
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True);
    val_loader = DataLoader(val_dataset, batch_size=256)
    model = AnisoFCNN();
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5);
    criterion = nn.MSELoss()
    print("开始训练FCNN模型(已加入Dropout和L2正则化)...")
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_Y in train_loader:
            optimizer.zero_grad();
            outputs = model(batch_X);
            loss = criterion(outputs, batch_Y)
            loss.backward();
            optimizer.step()
    print("模型训练完成。");
    return model, val_loader


def evaluate_model_quantitatively(model, val_loader, scaler_y):
    print("\n--- 开始对模型进行量化评估 ---");
    model.eval();
    all_targets, all_predictions = [], []
    with torch.no_grad():
        for batch_X, batch_Y_scaled in val_loader:
            prediction_scaled = model(batch_X);
            targets_unscaled = scaler_y.inverse_transform(batch_Y_scaled.numpy());
            predictions_unscaled = scaler_y.inverse_transform(prediction_scaled.numpy())
            all_targets.append(targets_unscaled);
            all_predictions.append(predictions_unscaled)
    Y_actual = np.concatenate(all_targets, axis=0);
    Y_predicted = np.concatenate(all_predictions, axis=0)
    print("评估指标 (基于20%的测试集):")
    for i, stress_name in enumerate(['Sxx', 'Syy', 'Sxy']):
        mae = mean_absolute_error(Y_actual[:, i], Y_predicted[:, i]);
        rmse = np.sqrt(mean_squared_error(Y_actual[:, i], Y_predicted[:, i]));
        r2 = r2_score(Y_actual[:, i], Y_predicted[:, i])
        print(f"  {stress_name}: MAE={mae:.2f} MPa, RMSE={rmse:.2f} MPa, R² Score={r2:.3f}")
    print("--------------------------------\n")


def plot_prediction_vs_actual(model, original_grains_data, scaler_x, scaler_y):
    print("开始为真实晶粒绘制'预测vs实际'对比图...")
    for i, grain_data in enumerate(original_grains_data):
        if len(grain_data['exx']) == 0:
            print(f"晶粒 {i + 1} 数据为空，跳过绘图。")
            continue
        plt.figure(figsize=(10, 8))
        plt.scatter(grain_data['exx'], grain_data['sxx'], s=15, alpha=0.4, color='blue', label='Actual Data')
        avg_euler = np.mean(grain_data['euler_angles'], axis=0)
        strain_range_actual = (np.min(grain_data['exx']), np.max(grain_data['exx']))
        print(f"  为Grain {i + 1} 在应变范围 [{strain_range_actual[0]:.4f}, {strain_range_actual[1]:.4f}] 内进行预测。")
        strain_range, stress_predicted = predict_stress_strain_curve(model, avg_euler, scaler_x, scaler_y,
                                                                     strain_range=strain_range_actual)
        plt.plot(strain_range, stress_predicted, color='red', linewidth=3, label=f'Model Prediction')
        plt.title(f'Prediction vs. Actual for Grain {i + 1}', fontsize=16)
        plt.xlabel('Strain (Exx)', fontsize=12);
        plt.ylabel('Stress (Sxx) [MPa]', fontsize=12)
        plt.legend();
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(f"prediction_vs_actual_grain_{i + 1}_final.png", dpi=300);
        plt.show()


# ----------------- 主程序入口 -----------------
if __name__ == '__main__':
    # --- 参数设置 ---
    MAT_FILE = 'AZ31_Grains_Data.mat'  # 确保您使用的是新数据源处理后生成的文件
    EPOCHS = 200
    LEARNING_RATE = 1e-4
    PERFORM_CLEANING = False  # <--- 重要开关！设置为False来使用您观察到的“好”数据

    # 1. 加载数据，可选择是否进行清理
    X_tensor, Y_tensor, original_grains = load_and_prepare_fcnn_data(MAT_FILE, perform_cleaning=PERFORM_CLEANING)

    if X_tensor is not None and len(X_tensor) > 0:
        # 2. 数据标准化
        scaler_x = StandardScaler().fit(X_tensor);
        scaler_y = StandardScaler().fit(Y_tensor)
        X_scaled = torch.tensor(scaler_x.transform(X_tensor), dtype=torch.float32)
        Y_scaled = torch.tensor(scaler_y.transform(Y_tensor), dtype=torch.float32)

        # 3. 训练模型
        model, val_loader = train_fcnn_model(X_scaled, Y_scaled, epochs=EPOCHS, learning_rate=LEARNING_RATE)

        # 4. 进行量化评估
        evaluate_model_quantitatively(model, val_loader, scaler_y)

        # 5. 进行最终的、优化的定性可视化评估
        plot_prediction_vs_actual(model, original_grains, scaler_x, scaler_y)
    else:
        print("没有加载到有效数据，程序终止。")