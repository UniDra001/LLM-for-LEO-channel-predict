import os
import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from transformers import AutoModelForCausalLM
from einops import rearrange
from Embed import DataEmbedding

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class Res_block(nn.Module):
    def __init__(self, in_planes):
        super(Res_block, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, in_planes, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_planes, in_planes, 3, 1, 1)
        self.ca = ChannelAttention(in_planes=in_planes, ratio=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        rs1 = self.relu(self.conv1(x))
        rs1 = self.conv2(rs1)
        channel_attn = self.ca(rs1)
        output = channel_attn * rs1
        rs = torch.add(x, output)
        return rs


class QwenModel(nn.Module):

    def __init__(self, d_ff=1536, d_model=1536,
                 pred_len=4, prev_len=20, use_gpu=1, gpu_id=0, mlp=0, res_layers=4,
                 enc_in = 96, c_out = 96,
                 patch_size=4, stride=1, res_dim=64,
                 embed='timeF', freq='h', dropout=0.1):
        super(QwenModel, self).__init__()
        self.device = torch.device('cuda:{}'.format(gpu_id))
        self.mlp = mlp
        self.res_layers = res_layers
        self.pred_len = pred_len
        self.prev_len = prev_len
        self.patch_size = patch_size
        self.stride = stride
        self.d_ff = d_ff
        self.d_model = d_model

        # self.K = K
        # self.UQh = UQh
        # self.UQv = UQv
        # self.BQh = BQh
        # self.BQv = BQv
        # self.Nt = UQh * UQv
        # self.Nr = BQh * BQv
        # self.mul = prev_len * K * UQh * UQv * BQh * BQv
        self.enc_in = enc_in
        self.c_out = c_out

        self.enc_embedding1 = DataEmbedding(self.enc_in, self.d_model, embed, freq, dropout)

        self.qwen2 = AutoModelForCausalLM.from_pretrained('./qwen2', output_attentions=True, output_hidden_states=True)

        for i, (name, param) in enumerate(self.qwen2.named_parameters()):
            if 'embed' in name:  # or 'mlp' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        if use_gpu:
            device = torch.device('cuda:{}'.format(gpu_id))
            self.qwen2.to(device=device)

        self.patch_layer = nn.Linear(self.patch_size, self.patch_size)
        self.patch_layer_fre = nn.Linear(self.patch_size, self.patch_size)
        self.predict_linear_pre = nn.Linear(self.prev_len, self.prev_len)
        self.out_layer_dim = nn.Linear(d_ff, self.c_out)
        self.output_layer_time = nn.Sequential(
            nn.Linear(self.prev_len, self.pred_len)
        )

        self.RB_e = nn.Sequential(nn.Conv2d(2, res_dim, 3, 1, 1))
        self.RB_f = nn.Sequential(nn.Conv2d(2, res_dim, 3, 1, 1))
        for i in range(self.res_layers):
            self.RB_e.append(Res_block(res_dim))
            self.RB_f.append(Res_block(res_dim))
        self.RB_e.append(nn.Conv2d(res_dim, 2, 3, 1, 1))
        self.RB_f.append(nn.Conv2d(res_dim, 2, 3, 1, 1))

    def forward(self, x_enc, x_mark_enc = None):
        mean = torch.mean(x_enc)
        std = torch.std(x_enc)
        x_enc = (x_enc - mean) / std
        B, L, enc_in = x_enc.shape  # [B, L, D]
        # process in delay domain
        x_enc_r = rearrange(x_enc, 'b l (k o) -> b l k o', o=2)
        x_enc_complex = torch.complex(x_enc_r[:, :, :, 0], x_enc_r[:, :, :, 1])
        x_enc_delay = torch.fft.ifft(x_enc_complex, dim=2)
        x_enc_delay = torch.cat([torch.real(x_enc_delay), torch.imag(x_enc_delay)], dim=2)
        x_enc_delay = x_enc_delay.reshape(B, L // self.patch_size, self.patch_size, enc_in)
        x_enc_delay = self.patch_layer(x_enc_delay.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        x_enc_delay = x_enc_delay.reshape(B, L, enc_in)
        x_enc_delay = rearrange(x_enc_delay, 'b l (k o) -> b o l k', o=2)
        x_enc_delay = self.RB_f(x_enc_delay)
        # process in frequency domain
        x_enc_fre = x_enc.reshape(B, L // self.patch_size, self.patch_size, enc_in)
        x_enc_fre = self.patch_layer(x_enc_fre.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        x_enc_fre = x_enc_fre.reshape(B, L, enc_in)
        x_enc_fre = rearrange(x_enc_fre, 'b l (k o) -> b o l k', o=2)
        x_enc_fre = self.RB_e(x_enc_fre)

        x_enc = x_enc_fre + x_enc_delay
        x_enc = rearrange(x_enc, 'b o l k -> b l (k o)', o=2)  # [B, L, D]

        enc_out = self.enc_embedding1(x_enc, x_mark_enc)  # [B, L, 768]

        enc_out = self.predict_linear_pre(enc_out.permute(0, 2, 1)).permute(0, 2, 1)
        enc_out = torch.nn.functional.pad(enc_out, (0, self.d_model - enc_out.shape[-1]))

        dec_out = self.qwen2(inputs_embeds=enc_out).hidden_states[-1]  # [B, L, 768]
        dec_out = dec_out[:, :, :self.d_ff]

        dec_out = self.out_layer_dim(dec_out)
        dec_out = self.output_layer_time(dec_out.permute(0, 2, 1)).permute(0, 2, 1)

        # 反归一化
        dec_out = dec_out * std + mean

        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
