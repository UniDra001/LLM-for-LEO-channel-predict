function A_large_scale_seq = large_scal_by_dis(dis_seq, fc, total_len)
    A_large_scale_seq = zeros(1, total_len);
    for iter = 1:total_len
        A_large_scale_db = 20*log10(fc/1000000) + 20*log10(dis_seq(iter)) +32.4;
        A_large_scale_seq(iter) = 10^(-1*A_large_scale_db/10);
    end
end