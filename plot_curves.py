import scipy.io
import numpy as np
import matplotlib.pyplot as plt


# ----------------- 数据加载与准备 (已修正) -----------------

def load_separated_grain_data(mat_file):
    """
    加载由新版MATLAB脚本生成的、已分好组的Grains_Data_Separated.mat文件。
    增加了对深度嵌套数据结构的处理。
    """
    try:
        data = scipy.io.loadmat(mat_file)
        # data['Grains_Data_Cell'] 是一个 (3, 1) 的cell array
        grain_data_list = []
        # 遍历cell array, 提取每个晶粒的数据
        for i in range(len(data['Grains_Data_Cell'])):
            grain_struct = data['Grains_Data_Cell'][i, 0]

            # --- 终极修正：直接从嵌套结构中提取核心数据矩阵 ---
            # 通过 [0, 0] 索引访问嵌套在最深处的数值矩阵
            grain_dict = {
                'sxx': grain_struct['Stress_xx'][0, 0].flatten(),
                'syy': grain_struct['Stress_yy'][0, 0].flatten(),
                'sxy': grain_struct['Stress_xy'][0, 0].flatten(),
                'exx': grain_struct['Strain_xx'][0, 0].flatten(),
                'eyy': grain_struct['Strain_yy'][0, 0].flatten(),
                'exy': grain_struct['Strain_xy'][0, 0].flatten()
            }
            grain_data_list.append(grain_dict)

        print(f"成功加载 {mat_file}，共找到 {len(grain_data_list)} 个晶粒的数据。")
        return grain_data_list

    except FileNotFoundError:
        print(f"错误: 找不到文件 {mat_file}。")
        return None
    except Exception as e:
        print(f"加载或处理 .mat 文件时发生未知错误: {e}")
        print("这可能表示 .mat 文件的内部结构比预期的更复杂。")
        return None


# ----------------- 计算与可视化 (保持不变) -----------------

def calculate_von_mises_stress_2d(sxx, syy, sxy):
    return np.sqrt(sxx ** 2 - sxx * syy + syy ** 2 + 3 * sxy ** 2)


def calculate_effective_strain_2d(exx, eyy, exy):
    return np.sqrt((2 / 3) * (exx ** 2 + eyy ** 2 - exx * eyy + 2 * exy ** 2))


def plot_detailed_grain_curves(grain_data_list):
    """
    绘制详细的、区分不同晶粒的应力-应变曲线图。
    """
    print("开始绘制详细的应力-应变曲线图...")

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 蓝、橙、绿
    n_grains = len(grain_data_list)

    # --- 第一张图: Sxx vs. Exx ---
    fig1, axes1 = plt.subplots(1, 4, figsize=(24, 6), sharey=True)
    fig1.suptitle('Principal Stress (Sxx) vs. Principal Strain (Exx)', fontsize=16)

    for i in range(n_grains):
        grain_data = grain_data_list[i]
        axes1[i].scatter(grain_data['exx'], grain_data['sxx'],
                         s=5, alpha=0.5, color=colors[i])
        axes1[i].set_title(f'Grain {i + 1}')
        axes1[i].grid(True, linestyle='--', alpha=0.6)
        axes1[i].set_xlabel('Strain (Exx)')

    axes1[0].set_ylabel('Stress (Sxx) [MPa]')

    axes1[3].set_title('All Grains Superimposed')
    for i in range(n_grains):
        grain_data = grain_data_list[i]
        axes1[3].scatter(grain_data['exx'], grain_data['sxx'],
                         s=5, alpha=0.5, color=colors[i], label=f'Grain {i + 1}')
    axes1[3].legend()
    axes1[3].grid(True, linestyle='--', alpha=0.6)
    axes1[3].set_xlabel('Strain (Exx)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("principal_stress_strain_curves.png", dpi=300)
    print("主应力-主应变曲线图已保存。")
    plt.show()

    # --- 第二张图: Von Mises Stress vs. Effective Strain ---
    fig2, axes2 = plt.subplots(1, 4, figsize=(24, 6), sharey=True)
    fig2.suptitle('Equivalent Stress vs. Equivalent Strain', fontsize=16)

    for i in range(n_grains):
        grain_data = grain_data_list[i]
        von_mises = calculate_von_mises_stress_2d(grain_data['sxx'], grain_data['syy'], grain_data['sxy'])
        effective_strain = calculate_effective_strain_2d(grain_data['exx'], grain_data['eyy'], grain_data['exy'])

        axes2[i].scatter(effective_strain, von_mises,
                         s=5, alpha=0.5, color=colors[i])
        axes2[i].set_title(f'Grain {i + 1}')
        axes2[i].grid(True, linestyle='--', alpha=0.6)
        axes2[i].set_xlabel('Effective Strain')

    axes2[0].set_ylabel('Von Mises Stress [MPa]')

    axes2[3].set_title('All Grains Superimposed')
    for i in range(n_grains):
        grain_data = grain_data_list[i]
        von_mises = calculate_von_mises_stress_2d(grain_data['sxx'], grain_data['syy'], grain_data['sxy'])
        effective_strain = calculate_effective_strain_2d(grain_data['exx'], grain_data['eyy'], grain_data['exy'])
        axes2[3].scatter(effective_strain, von_mises,
                         s=5, alpha=0.5, color=colors[i], label=f'Grain {i + 1}')
    axes2[3].legend()
    axes2[3].grid(True, linestyle='--', alpha=0.6)
    axes2[3].set_xlabel('Effective Strain')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("equivalent_stress_strain_curves.png", dpi=300)
    print("等效应力-等效应变曲线图已保存。")
    plt.show()


# ----------------- 主程序入口 -----------------
if __name__ == '__main__':
    MAT_FILE = 'AZ31_Grains_Data.mat'  # 请确保您使用的是新版MATLAB脚本生成的文件

    list_of_grains = load_separated_grain_data(MAT_FILE)

    if list_of_grains:
        plot_detailed_grain_curves(list_of_grains)