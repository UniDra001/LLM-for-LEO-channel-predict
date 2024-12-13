import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import Dataset_Pro
import scipy.io as sio
from models.GPT4CP import GPTModel
import numpy as np
import logging
from models.model import TorchModel, choose_optimizer
from config import Config
import shutil
from metrics import NMSELoss, SE_Loss
from datetime import datetime
#[DEBUG, INFO, WARNING, ERROR, CRITICAL]
# 获取当前时间并格式化为字符串（如：2024-11-24_15-30-45）
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

log_filename = f"logs/log_{current_time}.txt"

# 配置日志记录
logging.basicConfig(
    filename=log_filename,          # 使用动态生成的日志文件名
    level=logging.INFO,             # 设置日志级别
    format="%(asctime)s - %(levelname)s - %(message)s",  # 日志格式
    datefmt="%Y-%m-%d %H:%M:%S"     # 时间格式
)
# ============= HYPER PARAMS(Pre-Defined) ==========#


###################################################################
# ------------------- Main Train (Run second)----------------------------------
###################################################################
def train(config):
    
    #创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    # global epochs, best_loss
    model = TorchModel(config)
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        model = model.cuda()
    #加载优化器
    optimizer = choose_optimizer(config, model)
    epochs= config["epoch"]
    # 加载损失函数
    criterion = NMSELoss().cuda()
    
    # 显示模型参数
    show_model_parament(model)
    
    best_loss = 100
    
    model_message = f"Start training {model.model_type} ..."
    printAndLog(model_message)
    train_loss = []
    valid_loss = []
    for epoch in range(epochs):
        epoch_train_loss, epoch_val_loss = [], []
        # ============Epoch Train=============== #
        model.train()

        for iteration, batch in enumerate(training_data_loader, 1):
            pred_t, prev = Variable(batch[0]).cuda(), \
                           Variable(batch[1]).cuda()
            optimizer.zero_grad()  # fixed
            pred_m = model(prev)
            loss = criterion(pred_m, pred_t)  # compute loss
            epoch_train_loss.append(loss.item())  # save all losses into a vector for one epoch

            loss.backward()
            optimizer.step()

        t_loss = np.nanmean(np.array(epoch_train_loss))  # compute the mean value of all losses, as one epoch loss
        train_loss.append(t_loss)
        train_epoch_message = f"Epoch: {epoch+1}/{epochs} training loss: {t_loss:.7f}"
        printAndLog(train_epoch_message)  # logging loss for each epoch

        # ============Epoch Validate=============== #
        model.eval()
        with torch.no_grad():
            for iteration, batch in enumerate(validate_data_loader, 1):
                pred_t, prev = Variable(batch[0]).cuda(), \
                               Variable(batch[1]).cuda()
                optimizer.zero_grad()  # fixed
                pred_m = model(prev)
                loss = criterion(pred_m, pred_t)  # compute loss
                epoch_val_loss.append(loss.item())  # save all losses into a vector for one epoch
            v_loss = np.nanmean(np.array(epoch_val_loss))
            valid_loss.append(v_loss)
            valid_epoch_message = f"validate loss: {v_loss:.7f}"
            printAndLog(valid_epoch_message)
            if v_loss < best_loss:
                best_loss = v_loss
                save_best_checkpoint(model, config)
    # 记录训练和验证的损失
    train_loss = [f"{item:.7f}" for item in train_loss]
    valid_loss = [f"{item:.7f}" for item in valid_loss]
    train_loss_message = f"Training loss: {train_loss}"
    valid_loss_message = f"Validate loss: {valid_loss}"
    printAndLog(train_loss_message)
    printAndLog(valid_loss_message)
    return best_loss

def generate_data(config):
    train_set = Dataset_Pro(config["train_file_path"], is_train=1, SNR=config["SNR"])  # creat data for training
    validate_set = Dataset_Pro(config["valid_file_path"], is_train=0, SNR=config["SNR"])  # creat data for validation
        
    training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=config["batch_size"], 
                                    shuffle=True,
                                    pin_memory=True,
                                    drop_last=True)  # put training data to DataLoader for batches
    validate_data_loader = DataLoader(dataset=validate_set, num_workers=0, batch_size=config["batch_size"],
                                    shuffle=True,
                                    pin_memory=True,
                                    drop_last=True)  # put training data to DataLoader for batches
    return training_data_loader, validate_data_loader

def printAndLog(message):
    print(message)
    logging.info(message)

def show_model_parament(model):
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.5fM" % (total / 1e6))
    total_learn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of learnable parameter: %.5fM" % (total_learn / 1e6))
    
def save_best_checkpoint(model, config):  # save model function
    model_out_path = config["model_out_path"] + config["model_type"] + ".pth"
    torch.save(model, model_out_path)
    
# ------------------- Main Function (Run first) -------------------
if __name__ == "__main__":
    # ['gpt', 'transformer', 'cnn', 'gru', 'lstm', 'rnn']
    model_list = ['qwen', 'gpt', 'cnn', 'gru', 'rnn', 'lstm'] # 'gpt', 'transformer', 
    snr_list = [0, 5, 10, 15, 20, 25, 30, 35]
    nmse_model_list = []
    for model in model_list:
        Config["model_type"] = model
        nmse_list = []
        for snr in snr_list:
            Config["SNR"] = snr
            Config["model_out_path"] += f"SNR_{snr}/"
            os.makedirs(Config["model_out_path"], exist_ok=True)
            training_data_loader, validate_data_loader = generate_data(Config)
            nmse_list.append(train(Config))
        nmse_model_list.append(f"{model} : {nmse_list}")
    printAndLog(f"best loss:\n nmse_model_list")
