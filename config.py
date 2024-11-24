# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "vocab_path":"chars.txt",
    "model_type":"cnn",
    "max_length": 30,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 5,
    "batch_size": 128,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path":r"D:\bert-base-chinese",
    "seed": 987,
    # 新增待确定参数
    "train_file_path": "./train_data/H_U_train.mat",
    "valid_file_path": "./test_data/H_U_test.mat",
    "enc_in": 96,
    "dec_in": 96,
    "c_out": 96,
    "out_len": 123,
    "input_size": 123,
    "pred_len": 4,
    "prev_len":20,
    "label_len":16,
    "model_out_path": "Weights/",
    "features": 96,
    "hidden_size": 192,
    "input_size": 96,
    "SNR": 15,
}

