function [dis_seq, ele_seq] = gen_dis_ele_seq(lat, lon, angle_start, angle_interval, seq_len)
    angle_seq = angle_start:angle_interval:angle_start+angle_interval*(seq_len-1);
    ans_seq = leo_satellite(angle_seq); % 卫星轨迹的地心坐标序列
    dis_seq = zeros(1, seq_len); % 卫星轨迹离地面终端的距离序列
    ele_seq = zeros(1, seq_len); % 卫星轨迹离地面终端的仰角序列
    % 画图
%     plotOnFig;
    for iter = 1:seq_len
        x1 = ans_seq(1, iter);
        y1 = ans_seq(2, iter);
        z1 = ans_seq(3, iter);
        [dis, ele] = calc_distance_elevation(x1, y1, z1, lat, lon);
        dis_seq(iter) = dis;
        ele_seq(iter) = ele;
    end
end