%% This is a function to remove the data in a semicircle for DIC data process (保留圆弧顶点）

function [Z_filtered, filtered_points] = remove_semicircle(Z, xc, yc, xt, yt)
%REMOVE_SEMICIRCLE 从应变场中剔除半圆区域
%
% 输入参数：
%   Z         - 原始二维应变矩阵
%   xc, yc    - 半圆圆心坐标（像素点）
%   xt, yt    - 半圆弧顶坐标（像素点）
%
% 输出参数：
%   Z_filtered       - 剔除后的应变矩阵（NaN 表示被剔除）
%   filtered_points  - 保留的非NaN点，[x, y, strain] 三列

    % 获取矩阵尺寸
    [rows, cols] = size(Z);
    
    % 构建网格坐标
    [X, Y] = meshgrid(1:cols, 1:rows);  % 注意 X 为列，Y 为行
    
    % 计算半径
    R = sqrt((xt - xc)^2 + (yt - yc)^2);
    
    % 判断方向
    if abs(xt - xc) > abs(yt - yc)
        % 左右半圆（以x轴为主）
        if xt > xc
            half_mask = X >= xc;  % 右半圆
        else
            half_mask = X <= xc;  % 左半圆
        end
    else
        % 上下半圆（以y轴为主）
        if yt > yc
            half_mask = Y >= yc;  % 上半圆
        else
            half_mask = Y <= yc;  % 下半圆
        end
    end

    % 距离圆心的平方
    dist2 = (X - xc).^2 + (Y - yc).^2;
    in_circle = dist2 <= R^2;
    
    % 最终屏蔽区域
    mask = in_circle & half_mask;
    
    % ——保留弧顶像素（xt, yt）——
    xi = round(xt);
    yi = round(yt);
    if xi >= 1 && xi <= cols && yi >= 1 && yi <= rows
        mask(yi, xi) = false;   % 注意：行=Y，列=X
    end
    
    % 创建屏蔽后的矩阵
    Z_filtered = Z;
    Z_filtered(mask) = NaN;
    
    % 提取保留下来的点云（非NaN）
    valid_mask = ~isnan(Z_filtered);
    x_valid = X(valid_mask);
    y_valid = Y(valid_mask);
    strain_valid = Z_filtered(valid_mask);
    filtered_points = [x_valid(:), y_valid(:), strain_valid(:)];
end
