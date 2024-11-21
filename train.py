import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import Dataset_Pro
import scipy.io as sio
from models.GPT4CP import Model
import numpy as np
import logging
from models.model import TorchModel, choose_optimizer
from config import Config
import shutil
from metrics import NMSELoss, SE_Loss
#[DEBUG, INFO, WARNING, ERROR, CRITICAL]
logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# ============= HYPER PARAMS(Pre-Defined) ==========#
# lr = 0.0001
# epochs = 500
# batch_size = 1024
# device = torch.device('cuda')

# best_loss = 100
# save_path = "Weights/U2U_LLM4CP.pth"
# train_TDD_r_path = "./train_data/H_U_his_train.mat"
# train_TDD_t_path = "./train_data/H_U_pre_train.mat"
# key = ['H_U_his_train', 'H_U_pre_train', 'H_D_pre_train']
# train_set = Dataset_Pro(train_TDD_r_path, train_TDD_t_path, is_train=1, is_U2D=0, is_few=0)  # creat data for training
# validate_set = Dataset_Pro(train_TDD_r_path, train_TDD_t_path, is_train=0, is_U2D=0)  # creat data for validation

# model = Model(gpu_id=0,
#               pred_len=4, prev_len=16,
#               UQh=1, UQv=1, BQh=1, BQv=1).to(device)
# if os.path.exists(save_path):
#     model = torch.load(save_path, map_location=device)





###################################################################
# ------------------- Main Train (Run second)----------------------------------
###################################################################
def train(config):
    training_data_loader, validate_data_loader = generate_data(config)
    #创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    # global epochs, best_loss
    model = TorchModel(config)
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    #加载优化器
    optimizer = choose_optimizer(config, model)
    epochs= config["epoch"]
    # 加载损失函数
    criterion = NMSELoss().cuda()
    
    # 显示模型参数
    show_model_parament(model)

    print('Start training...')
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
        print('Epoch: {}/{} training loss: {:.7f}'.format(epoch+1, epochs, t_loss))  # print loss for each epoch

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
            print('validate loss: {:.7f}'.format(v_loss))
            if v_loss < best_loss:
                best_loss = v_loss
                save_best_checkpoint(model, config)

def generate_data(config):
    train_set = Dataset_Pro(config["train_file_path"], is_train=1, is_U2D=0, is_few=0)  # creat data for training
    validate_set = Dataset_Pro(config["valid_file_path"], is_train=0, is_U2D=0)  # creat data for validation
        
    training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=config["batch_size"], shuffle=True,
                                    pin_memory=True,
                                    drop_last=True)  # put training data to DataLoader for batches
    validate_data_loader = DataLoader(dataset=validate_set, num_workers=0, batch_size=["batch_size"],
                                    shuffle=True,
                                    pin_memory=True,
                                    drop_last=True)  # put training data to DataLoader for batches
    return training_data_loader, validate_data_loader

def show_model_parament(model):
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.5fM" % (total / 1e6))
    total_learn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of learnable parameter: %.5fM" % (total_learn / 1e6))
    
def save_best_checkpoint(model, config):  # save model function
    model_out_path = save_path
    torch.save(model, model_out_path)
    
# ------------------- Main Function (Run first) -------------------
if __name__ == "__main__":

    train(Config)  # call train function (