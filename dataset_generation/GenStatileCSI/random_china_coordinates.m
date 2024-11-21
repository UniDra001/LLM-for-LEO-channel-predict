function [random_lon, random_lat] = random_china_coordinates()
    % RANDOM_CHINA_COORDINATES 生成中国地区的随机经纬度坐标
    %
    % 输出:
    % random_lon - 随机生成的经度（73°E 到 135°E）
    % random_lat - 随机生成的纬度（18°N 到 54°N）
    
    % 中国大致的经纬度范围
    lon_min = 110;   % 最西经度
    lon_max = 120;  % 最东经度
    lat_min = 40;   % 最南纬度
    lat_max = 45;   % 最北纬度
    
    % 随机生成一个经度和一个纬度
    random_lon = lon_min + (lon_max - lon_min) * rand();
    random_lat = lat_min + (lat_max - lat_min) * rand();
    
    % 打印经纬度
%     fprintf('随机生成的中国地区经纬度: 经度 %.4f, 纬度 %.4f\n', random_lon, random_lat);
end