% 本代码用来进行应力计算的第二种方法，将应变旋转到晶体坐标系，然后进行应力计算，再将应力旋转到样品坐标系
% 并将三维旋转矩阵投影到二维平面上
% 以通过DIC测量的塑性应变来计算应力
% 要使用该代码，应先获取EBSD中的欧拉角、加载最高点和产生塑性应变后卸载的DIC应变分布
% 在readtable位置输入欧拉角的.txt文件，并直接打开Ncorr的变量即可使用
% 该代码使用的样品坐标系就是EBSD中的样品坐标，因为欧拉角是直接读出的
% 确保DIC坐标与样品坐标对应，可以先用AZtec把EBSD坐标转过来

% 第一步
% 定义HCP结构的弹性常数
correct = false;

while ~correct
 % 输入弹性矩阵
%  elastic_matrix=zeros(6,6);
%  elastic_matrix(1,1) = input('请输入弹性常数 C11: ');
%  elastic_matrix(2,2) = elastic_matrix(1,1);
%  elastic_matrix(1,2) = input('请输入弹性常数 C12: ');
%  elastic_matrix(2,1) = elastic_matrix(1,2);
%  elastic_matrix(1,3) = input('请输入弹性常数 C13: ');
%  elastic_matrix(3,1) = elastic_matrix(1,3);
%  elastic_matrix(2,3) = elastic_matrix(1,3);
%  elastic_matrix(3,2) = elastic_matrix(1,3);
%  elastic_matrix(3,3) = input('请输入弹性常数 C33: ');
%  elastic_matrix(4,4) = input('请输入弹性常数 C44: ');
%  elastic_matrix(5,5) = input('请输入弹性常数 C55: ');
%  elastic_matrix(6,6) = elastic_matrix(5,5);
 
 elastic_matrix=zeros(6,6);
 elastic_matrix(1,1) = 162400;
 elastic_matrix(2,2) = elastic_matrix(1,1);
 elastic_matrix(1,2) = 92000;
 elastic_matrix(2,1) = elastic_matrix(1,2);
 elastic_matrix(1,3) = 69000;
 elastic_matrix(3,1) = elastic_matrix(1,3);
 elastic_matrix(2,3) = elastic_matrix(1,3);
 elastic_matrix(3,2) = elastic_matrix(1,3);
 elastic_matrix(3,3) = 180700;
 elastic_matrix(4,4) = 35200;
 elastic_matrix(5,5) = 11700;
 elastic_matrix(6,6) = elastic_matrix(5,5);
    
    % 显示弹性矩阵以确认
    disp('您输入的弹性矩阵为:');
    disp(elastic_matrix);
    
    % 提示用户确认
    confirmation = input('请确认弹性矩阵是否正确 (输入y确认，n重新输入): ', 's');
    
    if strcmpi(confirmation, 'y')
        disp('弹性矩阵输入正确，继续程序...');
        correct = true;  % 退出循环
    elseif strcmpi(confirmation, 'n')
        % 提示重新输入
        disp('请重新输入弹性矩阵...');
    else
        % 处理无效输入，询问是否继续或退出
        choice = input('无效输入。是否要退出程序？(输入y退出，n重新输入): ', 's');
        if strcmpi(choice, 'y')
            disp('程序退出...');
            return;  % 退出程序
        else
            disp('请重新输入弹性矩阵...');
        end
    end
end

% 显示弹性矩阵
disp('弹性矩阵 C 为：');
disp(elastic_matrix);

% 第三步
% 将弹性矩阵根据欧拉角进行旋转变换
% 读取每个像素点的欧拉角
% Euler = readtable('StressCalArea_correct-E1 + E2 + E3');
% Euler(1, :) = [];

phi1 = Euler.Euler1;
PHI = Euler.Euler2;
phi2 = Euler.Euler3;
% 根据步长来读取EBSD图片尺寸
EBSDInterval = 3;
EBSDSize = [max(Euler.X)/EBSDInterval+1 max(Euler.Y)/EBSDInterval+1];
% 读取应变数据，通过ROI大小来读取图片尺寸，目前先读取一张图片
% 通过第一个非零元素和最后一个非零元素来定位矩形ROI，预留了一个可以处理i张图片的参数i
zerosindex = find(data_dic_save.strains(1).plot_exx_ref_formatted ~= 0);
firstindex = zerosindex(1);
lastindex = zerosindex(numel(zerosindex));
[firstr, firstc] = ind2sub(size(data_dic_save.strains(1).plot_exx_ref_formatted), firstindex);
[lastr, lastc] = ind2sub(size(data_dic_save.strains(1).plot_exx_ref_formatted), lastindex);
ROISize = [lastr-firstr+1 lastc-firstc+1];
% 用ROI大小初始化重新索引的应变矩阵
Exx = zeros(ROISize(1), ROISize(2)); 
Eyy = zeros(ROISize(1), ROISize(2)); 
Exy = zeros(ROISize(1), ROISize(2)); 
% 将非零元素填入新矩阵
for r = firstr:lastr
    for c = firstc:lastc
        Exx(r-firstr+1,c-firstc+1) = data_dic_save.strains(1).plot_exx_ref_formatted(r,c);
    end
end

% 重新排列Eyy
for r = firstr:lastr
    for c = firstc:lastc
        Eyy(r-firstr+1,c-firstc+1) = data_dic_save.strains(1).plot_eyy_ref_formatted(r,c);
    end
end
% 重新排列Exy
for r = firstr:lastr
    for c = firstc:lastc
        Exy(r-firstr+1,c-firstc+1) = data_dic_save.strains(1).plot_exy_ref_formatted(r,c);
    end
end

% 将ROI图片尺寸和EBSD图片尺寸进行对应，用DIC尺寸来适应EBSD尺寸
Exxq = imresize(Exx, [EBSDSize(2), EBSDSize(1)],'bilinear');    %imresize直接调整大小，比interp2好用很多，interp2不能扩大图像像素，只能插值平滑
Eyyq = imresize(Eyy, [EBSDSize(2), EBSDSize(1)],'bilinear');
Exyq = imresize(Exy, [EBSDSize(2), EBSDSize(1)],'bilinear');
% 用EBSD尺寸对应力变量进行初始化
Sxx = zeros(EBSDSize(2),EBSDSize(1));
Syy = zeros(EBSDSize(2),EBSDSize(1));
Sxy = zeros(EBSDSize(2),EBSDSize(1));
sigma = zeros(6,1);
phi1_matrix = reshape(phi1, EBSDSize(1), EBSDSize(2));
PHI_matrix = reshape(PHI, EBSDSize(1), EBSDSize(2));
phi2_matrix = reshape(phi2, EBSDSize(1), EBSDSize(2));

phi1_matrix = phi1_matrix';
PHI_matrix = PHI_matrix';
phi2_matrix = phi2_matrix';


for i = 1:(max(Euler.Index))
% 定义旋转矩阵，该矩阵为晶体在样品坐标系中的取向，那么，只要将该矩阵倒转，就能将弹性矩阵，从晶体坐标系变回样品坐标系_杨平：电子背散射衍射技术及其应用
Rphi1= [cosd(phi1_matrix(i)) sind(phi1_matrix(i)) 0;
    -sind(phi1_matrix(i)) cosd(phi1_matrix(i))  0;
    0 0 1    
       ];

RPHI= [1 0 0 ;
       0 cosd(PHI_matrix(i)) sind(PHI_matrix(i));
       0 -sind(PHI_matrix(i)) cosd(PHI_matrix(i))
       ];

Rphi2= [cosd(phi2_matrix(i)) sind(phi2_matrix(i)) 0;
    -sind(phi2_matrix(i)) cosd(phi2_matrix(i))  0;
    0 0 1
       ];

R = Rphi2*RPHI*Rphi1; %这里的R实质上就是晶体取向的矩阵表达，将样品坐标变换为晶体坐标

% 旋转矩阵为正交矩阵，用转置即可求逆
RC = R';
% 将晶体坐标的弹性矩阵转回样品坐标系，定义弹性矩阵的旋转矩阵
SElasticRotation = [RC(1,1)^2, RC(1,2)^2, RC(1,3)^2, 2*RC(1,1)*RC(1,2), 2*RC(1,2)*RC(1,3), 2*RC(1,1)*RC(1,3);
                   RC(2,1)^2, RC(2,2)^2, RC(2,3)^2, 2*RC(2,1)*RC(2,2), 2*RC(2,2)*RC(2,3), 2*RC(2,1)*RC(2,3);
                   RC(3,1)^2, RC(3,2)^2, RC(3,3)^2, 2*RC(3,1)*RC(3,2), 2*RC(3,2)*RC(3,3), 2*RC(3,1)*RC(3,3);
                   RC(1,1)*RC(2,1), RC(1,2)*RC(2,2), RC(1,3)*RC(2,3), (RC(1,1)*RC(2,2) + RC(1,2)*RC(2,1)), (RC(1,3)*RC(2,2) + RC(1,2)*RC(2,3)), (RC(1,3)*RC(2,1) + RC(1,1)*RC(2,3));
                   RC(3,1)*RC(2,1), RC(3,2)*RC(2,2), RC(3,3)*RC(2,3), (RC(3,2)*RC(2,1) + RC(3,1)*RC(2,2)), (RC(3,3)*RC(2,2) + RC(3,2)*RC(2,3)), (RC(3,3)*RC(2,1) + RC(3,1)*RC(2,3));
                   RC(3,1)*RC(1,1), RC(3,2)*RC(1,2), RC(3,3)*RC(1,3), (RC(3,2)*RC(1,1) + RC(3,1)*RC(1,2)), (RC(3,3)*RC(1,2) + RC(3,2)*RC(1,3)), (RC(3,3)*RC(1,1) + RC(3,1)*RC(1,3))];

EElasticRotation = [
    R(1,1)^2, R(1,2)^2, R(1,3)^2, R(1,1)*R(1,2), R(1,2)*R(1,3), R(1,1)*R(1,3);
    R(2,1)^2, R(2,2)^2, R(2,3)^2, R(2,1)*R(2,2), R(2,2)*R(2,3), R(2,1)*R(2,3);
    R(3,1)^2, R(3,2)^2, R(3,3)^2, R(3,1)*R(3,2), R(3,2)*R(3,3), R(3,1)*R(3,3);
    2*R(2,1)*R(1,1), 2*R(2,2)*R(1,2), 2*R(2,3)*R(1,3), (R(1,1)*R(2,2) + R(1,2)*R(2,1)), (R(1,2)*R(2,3) + R(1,3)*R(2,2)), (R(1,1)*R(2,3) + R(1,3)*R(2,1));
    2*R(3,1)*R(2,1), 2*R(3,2)*R(2,2), 2*R(3,3)*R(2,3), (R(2,1)*R(3,2) + R(2,2)*R(3,1)), (R(2,2)*R(3,3) + R(2,3)*R(3,2)), (R(2,1)*R(3,3) + R(2,3)*R(3,1));
    2*R(3,1)*R(1,1), 2*R(3,2)*R(1,2), 2*R(3,3)*R(1,3), (R(1,1)*R(3,2) + R(1,2)*R(3,1)), (R(1,2)*R(3,3) + R(1,3)*R(3,2)), (R(1,1)*R(3,3) + R(1,3)*R(3,1))
];


Csample = SElasticRotation*elastic_matrix*EElasticRotation;
% 第四步
% 根据平面应力假设进行计算，即把跟Z向相关的应力设成0，因为也没有Z向应变数据，假设Z向应变相对XY很小情况下才能进行计算，把Z向相关应变也设成0
E = [Exxq(i) Eyyq(i) 0 Exyq(i) 0 0]';
% 第五步
% 通过二维本构方程计算每一点的应力
sigma = Csample*E;
Sxx(i) = sigma(1);
Syy(i) = sigma(2);
Sxy(i) = sigma(4);
end




