function HTF_response = small_scale_channel(M,N,Delta_f,fc, Elevation)
    speed_C = 299792458;
    max_speed = 1000000;   %m/s

    % NTN-TDL-A
    path_num = 3;
    power_dB = [0, -4.675, -6.482];   %dB
    Norm_Delay = [0,1.0811,2.8416];  %NTN-TDL-A

    % Dense Urban Scenario (NLOS) in S band
    u_lgDS = [-6.76 -6.84	-6.81	-6.94	-7.14	-7.34	-7.53	-7.67	-7.82	-7.84];
    sigma_lgDS = [0.86 0.82	0.61	0.49	0.49	0.51	0.47	0.44	0.42	0.55];

    % 插值计算
    x_known = 0:10:90;
    u_interp = @(x_query) interp1(x_known, u_lgDS, x_query, 'linear');
    sigma_interp = @(x_query) interp1(x_known, sigma_lgDS, x_query, 'linear');


    % Calculate delay(ns)
    Delay_Spread = 10^(u_interp(Elevation) + sigma_interp(Elevation) * rand(1));
    Delay_scaled = Delay_Spread * Norm_Delay;
    delay_taps = round(Delay_scaled * (M * Delta_f));

    % delay_taps = [0 1 2];

    Doppler_taps = zeros(1,path_num);
    for path_idex = 1 : path_num
        Doppler_offset = cos(((rand(1) - 0.5) * 2) * pi) * max_speed / 3600 / speed_C * fc;  %假设最大多普勒频偏为100kHz
        % Doppler_offset = rand(1) * 10000;
        Doppler_index = round(Doppler_offset/(Delta_f/N));
        Doppler_taps(1,path_idex) = [Doppler_index];
    end

    pow_prof = 10.^(0.1 * power_dB);
    chan_coef = sqrt(pow_prof/sum(pow_prof)).*(sqrt(1/2) * ((randn(1,path_num)+1i*randn(1,path_num))));
    % chan_coef = [1 0 0];
    %生成莱斯信道
    % chan_coef = sqrt(K_rice/(K_rice+1)) + sqrt(1/(K_rice+1)) * chan_coef;
    % chan_coef = 1;
    % sum_power = sum(abs(chan_coef).^2)
    HDD_response = zeros(M,N);
    for i = 1:path_num
        HDD_response(M/2+delay_taps(i), N/2+Doppler_taps(i)) = chan_coef(i);
    end
    HTF_response = ifft(fft(HDD_response).').';
end
