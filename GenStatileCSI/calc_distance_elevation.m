function [distance, elevation] = calc_distance_elevation(x1, y1, z1, lat, lon)
    % 地球半径
    R_earth = 6371;
    
    % 将纬度和经度转换为弧度
    lat = lat * pi / 180;
    lon = lon * pi / 180;
    
    % 地面终端的地心坐标
    x_ground = R_earth * cos(lat) * cos(lon);
    y_ground = R_earth * cos(lat) * sin(lon);
    z_ground = R_earth * sin(lat);
    
    % 卫星与地面的距离
    dx = x1 - x_ground;
    dy = y1 - y_ground;
    dz = z1 - z_ground;
    
    distance = sqrt(dx^2 + dy^2 + dz^2);
    
    % 地面终端的法向量
    N_ground = [x_ground, y_ground, z_ground];
    
    % 地面指向卫星的向量
    V_sat = [dx, dy, dz];
    
    % 计算仰角
    cos_theta = dot(N_ground, V_sat) / (norm(N_ground) * norm(V_sat));
    elevation = asin(cos_theta) * 180 / pi;  % 转换为度数
    
    % 仰角小于0时，设置为NaN（不在地平线以上）
    if elevation < 0
        elevation = NaN;
    end
end