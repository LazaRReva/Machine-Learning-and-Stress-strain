%% The main code for machine learning process 2025.07.31
% Extract strain 22, remove semicircle and 最后计算得到Z_filtered_all是对序列中所有DIC数据去除半圆并裁剪的数据，用于ML的前期数据

% 应变场初步可视化 %
% figure, imagesc(Data_Strain{5}(:,:,2,2)),caxis([-0.02 0.02]),axis off, axis image, colormap jet,colorbar

% ==== 参数设置 ====
% 圆心和弧顶（用于半圆剔除）
xc = 32;
yc = 170;
xt = 62;
yt = 170;

% ==== 提取并处理每个循环的 ε_yy 分布 ====
Z_filtered_all = cell(1, 6);  % 预分配结果

for i = 1:19
    
    % Step 1: 获取整张 ε_yy 分布取出第 i 个循环的 ε_yy，截取
    strain_tensor = Data.Strain{1, i};
    Z_full = strain_tensor(:, :, 2, 2);  
    
    % Step 2: 剔除右半圆区域（用的是全局坐标）
    [Z_filtered, ~] = remove_semicircle(Z_full, xc, yc, xt, yt);
    
    % Step 3: 截取 
    Z_local = Z_filtered(35:300, 62:400); 

    % Step 4: 存储结果
    Z_filtered_all{i} = Z_local;

end



% % ==== 画图简单可视 ====
% 
figure;
colormap('jet');

for i = 1:6
    subplot(2, 5, i);  % 2 行 5 列子图排布
    imagesc(Z_filtered_all{i});  % 显示应变分布
    colorbar;
    title(sprintf('Cycle %d', i));
    axis equal tight;
    set(gca, 'YDir', 'normal');  % 保证y轴方向向上
end

sgtitle('ε_{yy} Distribution (After Semicircle Removal)');



% ==== 截断区域可视化 (可选）====
% figure;
% imagesc(active_mask);
% set(gca, 'YDir', 'normal');
% axis image off;
% title('Region with ε_{pl}^{nonlocal} ≥ 0.002');
% colormap hot;
% colorbar;
% 
% 
% fatigue_strain_22_avg
% fatigue_strain_22_max