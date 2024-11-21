# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": "../data/train_tag_news.json",
    "valid_data_path": "../data/valid_tag_news.json",
    "vocab_path":"chars.txt",
    "model_type":"cnn",
    "max_length": 30,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 15,
    "batch_size": 128,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path":r"D:\bert-base-chinese",
    "seed": 987,
    # 新增待确定参数
    "enc_in": 123,
    "dec_in": 123,
    "c_out": 123,
    "out_len": 123,
    "input_size": 123,
    "pred_len": 4,
    "label_len":20,
}

