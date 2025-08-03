import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.preprocessing import StandardScaler


# ----------------- 辅助函数：物理计算与模型定义 -----------------

def euler_to_rotation_matrix(phi1, Phi, phi2):
    c1, s1 = np.cos(phi1), np.sin(phi1);
    c_phi, s_phi = np.cos(Phi), np.sin(Phi);
    c2, s2 = np.cos(phi2), np.sin(phi2)
    g1 = np.array([[c1, s1, 0], [-s1, c1, 0], [0, 0, 1]]);
    g_phi = np.array([[1, 0, 0], [0, c_phi, s_phi], [0, -s_phi, c_phi]]);
    g2 = np.array([[c2, s2, 0], [-s2, c2, 0], [0, 0, 1]])
    return np.dot(g2, np.dot(g_phi, g1))


class AnisoFCNN(nn.Module):
    def __init__(self, input_size=12, output_size=3):
        super(AnisoFCNN, self).__init__();
        self.network = nn.Sequential(nn.Linear(input_size, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(),
                                     nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, output_size))

    def forward(self, x): return self.network(x)


class VectorDataset(Dataset):
    def __init__(self, X, Y): self.X, self.Y = X, Y

    def __len__(self): return len(self.X)

    def __getitem__(self, idx): return self.X[idx], self.Y[idx]


def calculate_von_mises_stress_2d(sxx, syy, sxy): return np.sqrt(sxx ** 2 - sxx * syy + syy ** 2 + 3 * sxy ** 2)


def calculate_effective_strain_2d(exx, eyy, exy): return np.sqrt(
    (2 / 3) * (exx ** 2 + eyy ** 2 - exx * eyy + 2 * exy ** 2))


def calculate_principal_values(xx, yy, xy):
    p1 = 0.5 * (xx + yy) + np.sqrt((0.5 * (xx - yy)) ** 2 + xy ** 2)
    p2 = 0.5 * (xx + yy) - np.sqrt((0.5 * (xx - yy)) ** 2 + xy ** 2)
    return p1, p2


# ----------------- 数据加载 (已更新) -----------------
def load_and_prepare_fcnn_data(mat_file):
    try:
        data = scipy.io.loadmat(mat_file)
        grain_data_list = data['Grains_Data_Cell']
        original_grains_full_data = []
    except FileNotFoundError:
        print(f"错误: 找不到文件 {mat_file}。"); return None, None, None

    all_X, all_Y = [], []
    for i in range(len(grain_data_list)):
        grain_struct = grain_data_list[i, 0]
        sxx = grain_struct['Stress_xx'][0, 0].flatten();
        syy = grain_struct['Stress_yy'][0, 0].flatten();
        sxy = grain_struct['Stress_xy'][0, 0].flatten()
        exx = grain_struct['Strain_xx'][0, 0].flatten();
        eyy = grain_struct['Strain_yy'][0, 0].flatten();
        exy = grain_struct['Strain_xy'][0, 0].flatten()
        phi1 = grain_struct['Euler_phi1'][0, 0].flatten();
        phi_cap = grain_struct['Euler_Phi'][0, 0].flatten();
        phi2 = grain_struct['Euler_phi2'][0, 0].flatten()

        original_grains_full_data.append({'sxx': sxx, 'syy': syy, 'sxy': sxy, 'exx': exx, 'eyy': eyy, 'exy': exy,
                                          'euler_angles': np.stack((phi1, phi_cap, phi2), axis=1)})
        strain_vectors = np.stack((exx, eyy, exy), axis=1);
        stress_vectors = np.stack((sxx, syy, sxy), axis=1)

        for j in range(len(sxx)):
            p1_rad, p_rad, p2_rad = np.deg2rad([phi1[j], phi_cap[j], phi2[j]])
            rot_matrix = euler_to_rotation_matrix(p1_rad, p_rad, p2_rad)
            all_X.append(np.concatenate((strain_vectors[j], rot_matrix.flatten())))
        all_Y.append(stress_vectors)

    X_final = np.array(all_X);
    Y_final = np.concatenate(all_Y, axis=0)
    return torch.tensor(X_final, dtype=torch.float32), torch.tensor(Y_final,
                                                                    dtype=torch.float32), original_grains_full_data


# ----------------- 模型训练 -----------------
def train_fcnn_model(X_scaled, Y_scaled, epochs, learning_rate):
    dataset = VectorDataset(X_scaled, Y_scaled);
    train_size = int(0.8 * len(dataset));
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    model = AnisoFCNN();
    optimizer = optim.Adam(model.parameters(), lr=learning_rate);
    criterion = nn.MSELoss()
    print("开始训练FCNN模型...");
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_Y in train_loader:
            optimizer.zero_grad();
            outputs = model(batch_X);
            loss = criterion(outputs, batch_Y)
            loss.backward();
            optimizer.step()
    print("模型训练完成。");
    return model


# ----------------- 预测与可视化 (已重构) -----------------
def get_predictions_for_curve(model, euler_angles_deg, strain_range, scaler_x, scaler_y, steps=100, mode='principal'):
    """为特定取向和应变范围生成预测曲线所需的所有数据。"""
    model.eval()
    rot_matrix_flat = euler_to_rotation_matrix(*np.deg2rad(euler_angles_deg)).flatten()

    # 根据模式确定加载路径
    if mode == 'principal':
        # 我们需要找到与主应变范围对应的加载路径
        # 为简化，我们继续沿Exx方向加载，但范围由主应变决定
        # 这是一个近似，但在单向拉伸为主的情况下是合理的
        strain_xx_range = np.linspace(strain_range[0], strain_range[1], steps)
        strain_yy_range = -0.3 * strain_xx_range
        strain_xy_range = np.zeros_like(strain_xx_range)
    else:  # mode == 'equivalent'
        strain_xx_range = np.linspace(strain_range[0], strain_range[1], steps)
        strain_yy_range = -0.3 * strain_xx_range
        strain_xy_range = np.zeros_like(strain_xx_range)

    input_vectors_unscaled = []
    for j in range(len(strain_xx_range)):
        strain_vector = np.array([strain_xx_range[j], strain_yy_range[j], strain_xy_range[j]])
        input_vectors_unscaled.append(np.concatenate((strain_vector, rot_matrix_flat)))

    input_vectors_scaled = scaler_x.transform(np.array(input_vectors_unscaled))
    with torch.no_grad():
        predicted_stress_scaled = model(torch.tensor(input_vectors_scaled, dtype=torch.float32))
    predicted_stress = scaler_y.inverse_transform(predicted_stress_scaled.numpy())

    pred_sxx, pred_syy, pred_sxy = predicted_stress[:, 0], predicted_stress[:, 1], predicted_stress[:, 2]

    # 返回一个包含所有计算结果的字典
    return {
        'principal_stress': calculate_principal_values(pred_sxx, pred_syy, pred_sxy)[0],
        'principal_strain': calculate_principal_values(strain_xx_range, strain_yy_range, strain_xy_range)[0],
        'von_mises_stress': calculate_von_mises_stress_2d(pred_sxx, pred_syy, pred_sxy),
        'effective_strain': calculate_effective_strain_2d(strain_xx_range, strain_yy_range, strain_xy_range)
    }


def plot_final_comparison_curves(model, original_grains_data, scaler_x, scaler_y):
    print("开始为真实晶粒绘制'主应力/等效应力'对比图(已修正)...")

    for i, grain_data in enumerate(original_grains_data):
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle(f'Prediction vs. Actual for Grain {i + 1}', fontsize=18)

        # --- 准备真实数据 ---
        actual_principal_stress, _ = calculate_principal_values(grain_data['sxx'], grain_data['syy'], grain_data['sxy'])
        actual_principal_strain, _ = calculate_principal_values(grain_data['exx'], grain_data['eyy'], grain_data['exy'])
        actual_von_mises = calculate_von_mises_stress_2d(grain_data['sxx'], grain_data['syy'], grain_data['sxy'])
        actual_effective_strain = calculate_effective_strain_2d(grain_data['exx'], grain_data['eyy'], grain_data['exy'])

        avg_euler = np.mean(grain_data['euler_angles'], axis=0)

        # --- 绘图：主应力 vs 主应变 ---
        strain_range_principal = (np.min(actual_principal_strain), np.max(actual_principal_strain))
        pred_data_principal = get_predictions_for_curve(model, avg_euler, strain_range_principal, scaler_x, scaler_y,
                                                        mode='principal')

        axes[0].scatter(actual_principal_strain, actual_principal_stress, s=10, alpha=0.3, color='blue',
                        label='Actual Data')
        # 排序确保曲线正确绘制
        sort_indices = np.argsort(pred_data_principal['principal_strain'])
        axes[0].plot(pred_data_principal['principal_strain'][sort_indices],
                     pred_data_principal['principal_stress'][sort_indices], color='red', linewidth=3,
                     label='Model Prediction')
        axes[0].set_title('Max. Principal Stress vs. Max. Principal Strain', fontsize=14)
        axes[0].set_xlabel('Max. Principal Strain', fontsize=12);
        axes[0].set_ylabel('Max. Principal Stress (MPa)', fontsize=12)
        axes[0].legend();
        axes[0].grid(True, linestyle='--')

        # --- 绘图：等效应力 vs 等效应变 ---
        strain_range_effective = (np.min(actual_effective_strain), np.max(actual_effective_strain))
        pred_data_effective = get_predictions_for_curve(model, avg_euler, strain_range_effective, scaler_x, scaler_y,
                                                        mode='equivalent')

        axes[1].scatter(actual_effective_strain, actual_von_mises, s=10, alpha=0.3, color='blue', label='Actual Data')
        sort_indices = np.argsort(pred_data_effective['effective_strain'])
        axes[1].plot(pred_data_effective['effective_strain'][sort_indices],
                     pred_data_effective['von_mises_stress'][sort_indices], color='red', linewidth=3,
                     label='Model Prediction')
        axes[1].set_title('Equivalent Stress vs. Equivalent Strain', fontsize=14)
        axes[1].set_xlabel('Effective Strain', fontsize=12);
        axes[1].set_ylabel('Von Mises Stress (MPa)', fontsize=12)
        axes[1].legend();
        axes[1].grid(True, linestyle='--')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]);
        plt.savefig(f"final_comparison_grain_{i + 1}.png", dpi=300)
        plt.show()


# ----------------- 主程序入口 -----------------
if __name__ == '__main__':
    MAT_FILE = 'AZ31_Grains_Data.mat'
    EPOCHS = 200;
    LEARNING_RATE = 1e-4

    X_tensor, Y_tensor, original_grains = load_and_prepare_fcnn_data(MAT_FILE)

    if X_tensor is not None:
        scaler_x = StandardScaler().fit(X_tensor);
        scaler_y = StandardScaler().fit(Y_tensor)
        X_scaled = torch.tensor(scaler_x.transform(X_tensor), dtype=torch.float32)
        Y_scaled = torch.tensor(scaler_y.transform(Y_tensor), dtype=torch.float32)

        model = train_fcnn_model(X_scaled, Y_scaled, epochs=EPOCHS, learning_rate=LEARNING_RATE)

        # 调用新的可视化函数
        plot_final_comparison_curves(model, original_grains, scaler_x, scaler_y)