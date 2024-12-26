gpt= [0.5338136, 0.2638754, 0.1022743, 0.0372618, 0.0143291, 0.0065691, 0.0036334, 0.0033042];
trans = [0.5861256, 0.2986582, 0.1329517, 0.065289, 0.036197, 0.0310127, 0.0296217, 0.0248906];
gru = [0.557471, 0.3077122, 0.1558207, 0.1027015, 0.0772435, 0.0675815, 0.057795, 0.0650952];
rnn = [0.5576773, 0.2935448, 0.1298874, 0.0578804, 0.0346641, 0.0216114, 0.0190415, 0.0163837];
lstm = [0.632943, 0.4115701, 0.1881105, 0.2185857, 0.1902801, 0.1871289, 0.1884665, 0.1771432];

snr = [0, 5, 10, 15, 20, 25, 30, 35];

% 绘制图形
figure;
semilogy(snr, gpt, '-^r', 'DisplayName', 'GPT');
hold on;
semilogy(snr, trans, '-vg', 'DisplayName', 'Transformer');
semilogy(snr, gru, '-oy', 'DisplayName', 'GRU');
semilogy(snr, rnn, '-xb', 'DisplayName', 'RNN');
semilogy(snr, lstm, '-*c', 'DisplayName', 'LSTM');

% 设置坐标轴
xlabel('SNR(dB)');
ylabel('NMSE');
legend('show', 'Location', 'southwest');
grid on;
ylim([2e-3, 1]);
hold off;