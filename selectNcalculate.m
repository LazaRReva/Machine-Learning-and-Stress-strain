% =========================================================================
% 项目：第二期（2.5）- 新数据源处理与虚拟应力计算脚本
% 目标：1. 加载新的AZ31应变数据。
%       2. 可视化弹性应变场以便用户选择已知晶粒。
%       3. 根据用户输入的欧拉角计算所选晶粒的应力场。
%       4. 将数据打包成与Python FCNN模型兼容的格式。
% =========================================================================

clear; clc; close all;

%% --- 步骤 1: 设置与加载初始数据 ---
disp('步骤 1: 正在加载新数据源...');

% --- 物理常数定义 ---
% AZ31镁合金 (HCP) 的单晶弹性常数 (单位: GPa)
% 文献参考值，您也可以根据需要修改
C11 = 59.3;  C12 = 25.7;  C13 = 21.4;
C33 = 61.5;  C44 = 16.4;
% 根据HCP晶体对称性构建6x6的Voigt表示法刚度矩阵
C0_crystal = [
    C11 C12 C13 0   0   0;
    C12 C11 C13 0   0   0;
    C13 C13 C33 0   0   0;
    0   0   0   C44 0   0;
    0   0   0   0   C44 0;
    0   0   0   0   0   (C11-C12)/2
];
C0_crystal = C0_crystal * 1e3; % 从 GPa 转换为 MPa

% --- 加载.mat文件 ---
try
    load('AZ3_Data_Strain_100c.mat', 'Data_Strain');
    % 提取应变场数据，假设为 HxWx3x3 的四维数组
    epsilon_p = Data_Strain{1}; % 塑性应变
    epsilon_t = Data_Strain{2}; % 总应变
    epsilon_e = Data_Strain{3}; % 弹性应变
    disp('AZ31应变数据加载成功。');
catch e
    error('无法加载 AZ3_Data_Strain_100c.mat 或其内部变量不正确: %s', e.message);
end

% 获取数据尺寸
[h, w, ~, ~] = size(epsilon_e);

%% --- 步骤 2: 计算等效弹性应变并可视化 ---
disp('步骤 2: 正在计算等效弹性应变用于可视化...');

epsilon_e_eff = zeros(h, w);
for i = 1:h
    for j = 1:w
        % 从 (i,j) 点提取3x3的弹性应变张量
        e_tensor = squeeze(epsilon_e(i, j, :, :));
        % 计算Von Mises等效应变
        e_dev = e_tensor - (1/3)*trace(e_tensor)*eye(3); % 偏应变张量
        epsilon_e_eff(i, j) = sqrt((2/3) * sum(sum(e_dev .* e_dev)));
    end
end

% 绘制云图作为选择向导
figure('Name', '请根据此图选择晶粒区域', 'NumberTitle', 'off', 'WindowState', 'maximized');
imagesc(epsilon_e_eff);
axis equal tight;
colormap('jet');
colorbar;
caxis([0, 0.005]); % 将颜色范围强制设置为0到0.2%，这样可以看清低应变区的细节
title('等效弹性应变 (Von Mises) 分布图');

%% --- 步骤 3: 交互式选择、输入欧拉角并计算应力 ---

% 弹出对话框，询问用户要选择几个晶粒
prompt = {'您要选择几个晶粒?'};
dlgtitle = '设置';
dims = [1 35];
definput = {'3'};
answer = inputdlg(prompt, dlgtitle, dims, definput);
num_grains_to_select = str2double(answer{1});

% 初始化最终的数据容器
Grains_Data_Cell = cell(num_grains_to_select, 1);

for i = 1:num_grains_to_select
    fprintf('\n--- 正在处理第 %d / %d 个晶粒 ---\n', i, num_grains_to_select);
    
    % --- 交互式框选 ---
    fprintf('请用鼠标在图上拖动以选择第 %d 个晶粒区域...\n', i);
    rect = getrect;
    x_start = floor(rect(1)); x_end = ceil(rect(1) + rect(3));
    y_start = floor(rect(2)); y_end = ceil(rect(2) + rect(4));
    x_indices = max(1, x_start):min(w, x_end);
    y_indices = max(1, y_start):min(h, y_end);
    
    hold on;
    rectangle('Position', rect, 'EdgeColor', 'r', 'LineWidth', 2);
    text(x_start, y_start - 10, sprintf('晶粒 %d', i), 'Color', 'white', 'FontSize', 12, 'FontWeight', 'bold');
    hold off;
    
    % --- 弹出对话框，让用户输入欧拉角 ---
    prompt_euler = {sprintf('请输入晶粒 %d 的三个欧拉角 (phi1, Phi, phi2)，用空格隔开:', i)};
    dlgtitle_euler = '输入晶体取向 (角度制)';
    answer_euler = inputdlg(prompt_euler, dlgtitle_euler, [1 50]);
    euler_angles_deg = str2num(answer_euler{1});
    
    if numel(euler_angles_deg) ~= 3
        error('输入无效，必须是三个数字。请重新运行脚本。');
    end
    fprintf('晶粒 %d 的欧拉角: (%.1f, %.1f, %.1f)\n', i, euler_angles_deg(1), euler_angles_deg(2), euler_angles_deg(3));
    
    % --- 计算该取向的刚度矩阵 ---
    C_sample = transform_stiffness(C0_crystal, euler_angles_deg);
    
    % --- 提取数据并计算应力 ---
    fprintf('正在为晶粒 %d 计算应力场...\n', i);
    
    % 提取所选区域的应变张量
    epsilon_e_grain = epsilon_e(y_indices, x_indices, :, :);
    epsilon_t_grain = epsilon_t(y_indices, x_indices, :, :);
    
    [h_grain, w_grain, ~, ~] = size(epsilon_e_grain);
    stress_grain = zeros(h_grain, w_grain, 3, 3);
    
    % 逐点计算应力
    for y = 1:h_grain
        for x = 1:w_grain
            e_tensor = squeeze(epsilon_e_grain(y, x, :, :));
            e_voigt = [e_tensor(1,1); e_tensor(2,2); e_tensor(3,3); 2*e_tensor(2,3); 2*e_tensor(1,3); 2*e_tensor(1,2)];
            s_voigt = C_sample * e_voigt;
            s_tensor = [s_voigt(1) s_voigt(6) s_voigt(5);
                        s_voigt(6) s_voigt(2) s_voigt(4);
                        s_voigt(5) s_voigt(4) s_voigt(3)];
            stress_grain(y, x, :, :) = s_tensor;
        end
    end
    
    %% --- 步骤 4: 按照Python脚本的格式打包数据 ---
    current_grain_data = struct();
    
    % 提取并展平各个分量
    current_grain_data.Stress_xx = squeeze(stress_grain(:,:,1,1));
    current_grain_data.Stress_yy = squeeze(stress_grain(:,:,2,2));
    current_grain_data.Stress_xy = squeeze(stress_grain(:,:,1,2));
    
    current_grain_data.Strain_xx = squeeze(epsilon_t_grain(:,:,1,1)); % 使用总应变
    current_grain_data.Strain_yy = squeeze(epsilon_t_grain(:,:,2,2));
    current_grain_data.Strain_xy = squeeze(epsilon_t_grain(:,:,1,2));
    
    % 添加恒定的欧拉角信息
    num_points = numel(current_grain_data.Stress_xx);
    current_grain_data.Euler_phi1 = ones(size(current_grain_data.Stress_xx)) * euler_angles_deg(1);
    current_grain_data.Euler_Phi  = ones(size(current_grain_data.Stress_xx)) * euler_angles_deg(2);
    current_grain_data.Euler_phi2 = ones(size(current_grain_data.Stress_xx)) * euler_angles_deg(3);
    
    Grains_Data_Cell{i} = current_grain_data;
    fprintf('晶粒 %d 的数据已处理并打包完成。\n', i);
end

%% --- 步骤 5: 保存最终数据包 ---
save('AZ31_Grains_Data.mat', 'Grains_Data_Cell');

fprintf('\n========================================================\n');
fprintf('处理完成！\n');
fprintf('最终数据集已成功保存为 AZ31_Grains_Data.mat\n');
fprintf('该文件可以直接用于您之前的Python FCNN训练脚本。\n');
fprintf('========================================================\n');


%% --- 辅助函数 ---

function C_sample = transform_stiffness(C0_crystal, euler_angles_deg)
    % 将欧拉角从角度转换为弧度
    phi1 = deg2rad(euler_angles_deg(1));
    Phi  = deg2rad(euler_angles_deg(2));
    phi2 = deg2rad(euler_angles_deg(3));
    
    % Bunge ZXZ 约定下的取向矩阵 g
    c1 = cos(phi1); s1 = sin(phi1);
    cP = cos(Phi); sP = sin(Phi);
    c2 = cos(phi2); s2 = sin(phi2);
    
    g1 = [c1 s1 0; -s1 c1 0; 0 0 1];
    gP = [1 0 0; 0 cP sP; 0 -sP cP];
    g2 = [c2 s2 0; -s2 c2 0; 0 0 1];
    g = g2 * gP * g1; % 从晶体坐标系到样品坐标系的旋转
    
    % 计算Bond变换矩阵 M
    M = zeros(6,6);
    M(1,1) = g(1,1)^2; M(1,2) = g(1,2)^2; M(1,3) = g(1,3)^2;
    M(1,4) = 2*g(1,2)*g(1,3); M(1,5) = 2*g(1,3)*g(1,1); M(1,6) = 2*g(1,1)*g(1,2);
    
    M(2,1) = g(2,1)^2; M(2,2) = g(2,2)^2; M(2,3) = g(2,3)^2;
    M(2,4) = 2*g(2,2)*g(2,3); M(2,5) = 2*g(2,3)*g(2,1); M(2,6) = 2*g(2,1)*g(2,2);

    M(3,1) = g(3,1)^2; M(3,2) = g(3,2)^2; M(3,3) = g(3,3)^2;
    M(3,4) = 2*g(3,2)*g(3,3); M(3,5) = 2*g(3,3)*g(3,1); M(3,6) = 2*g(3,1)*g(3,2);

    M(4,1) = g(2,1)*g(3,1); M(4,2) = g(2,2)*g(3,2); M(4,3) = g(2,3)*g(3,3);
    M(4,4) = g(2,2)*g(3,3)+g(2,3)*g(3,2); M(4,5) = g(2,1)*g(3,3)+g(2,3)*g(3,1); M(4,6) = g(2,1)*g(3,2)+g(2,2)*g(3,1);

    M(5,1) = g(1,1)*g(3,1); M(5,2) = g(1,2)*g(3,2); M(5,3) = g(1,3)*g(3,3);
    M(5,4) = g(1,2)*g(3,3)+g(1,3)*g(3,2); M(5,5) = g(1,1)*g(3,3)+g(1,3)*g(3,1); M(5,6) = g(1,1)*g(3,2)+g(1,2)*g(3,1);

    M(6,1) = g(1,1)*g(2,1); M(6,2) = g(1,2)*g(2,2); M(6,3) = g(1,3)*g(2,3);
    M(6,4) = g(1,2)*g(2,3)+g(1,3)*g(2,2); M(6,5) = g(1,1)*g(2,3)+g(1,3)*g(2,1); M(6,6) = g(1,1)*g(2,2)+g(1,2)*g(2,1);

    % 计算旋转后的刚度矩阵
    C_sample = M * C0_crystal * M';
end