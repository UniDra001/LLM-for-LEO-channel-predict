clc;
clear;

sample_loc = 128; % 对sample_loc个地点生成卫星信道
num_per_loc = 256; % 每个地点采样num_per_loc个数据
R = 6871; % 低轨卫星离地心距离，单位km
prev_len = 20;
pred_len = 4;
total_len = prev_len + pred_len;
subCarrier_num = 48; % 子载波数
fc = 2e9; % S-band,4GHz,中心频率
Delta_f = 15000; % 子载波频率 1.5kHz
M = 48; % 子载波数
H_U_prev = zeros(sample_loc, num_per_loc, prev_len, subCarrier_num);
H_U_pred = zeros(sample_loc, num_per_loc, pred_len, subCarrier_num);
linear_vel = 7.62;  % 卫星的线速度，单位km/s
angle_vel = linear_vel/R; % 卫星角速度，单位rad/s
time_interval = 1; % 0.5*10e-3; % 信道的采样间隔，单位s
angle_interval = time_interval * angle_vel; % 角度间隔
isTrain = 0;
for iter_sample = 1:sample_loc
    % 随机生成中国地区的某个经纬度坐标
    [lon, lat] = random_china_coordinates();
    for iter_loc = 1: num_per_loc
        angle_start = 40 + (45 - 40) * rand(); % 根据卫星在中国地区的过境时间，计算大致的离心角范围是110~129
        angle_start = angle_start * pi / 180; % 转换为弧度
        % 获得卫星轨迹的距离和仰角序列，序列跟时间轴有关
        % 距离和大尺度有关，仰角和小尺度有关
        [dis_seq, ele_seq] = gen_dis_ele_seq(lat, lon, angle_start, angle_interval, total_len); 
%         TODO
%         1. 计算大尺度
        A_large_scale_seq = large_scal_by_dis(dis_seq, fc, total_len);
%         2. 计算小尺度
        tf_small_channel = small_scale_channel(M,total_len,Delta_f,fc, ele_seq(1));
%         3. 叠加
        tf_channel = A_large_scale_seq.*tf_small_channel;
        H_U_prev(iter_sample, iter_loc,:,:) = tf_channel(:, 1:prev_len).';
        H_U_pred(iter_sample, iter_loc,:,:) = tf_channel(:, prev_len+1:total_len).';
    end
end

if isTrain == 1
    save('..\train_data\H_U_train_large.mat', 'H_U_prev','H_U_pred');
else
    save('..\test_data\H_U_test_large.mat', 'H_U_prev','H_U_pred');
end
        
        
        
        
        
        