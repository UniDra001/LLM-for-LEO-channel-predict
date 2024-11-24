# 低轨卫星通信场景的信道预测算法

## 已完成

1. 低轨卫星运动学建模

2. 端到端数据集生成

3. RNN、GRU、LSTM、CNN、Transformer模型支持

## todo

1. LLM系列模型支持和微调

2. 信道预测算法性能评估（包括SE、NMSE、Loss随着epoch和SNR的变化）

3. 信道预测算法性能可视化分析

4. 单独的测试模型脚本

5. 模型性能优化

6. 考虑TDD、FDD场景的问题

7. 考虑信道估计和信道预测算法的评估指标

## Run Code

1. 运行gen_satelite_channel_data.m生成所需数据集

2. 运行 train.py 训练模型

3. config.py保存了模型相关的所有参数

