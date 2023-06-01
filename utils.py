import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

from scipy import io
from torch.utils.data import DataLoader

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def ManualSeed(seed:int, deterministic=False):
    # random seed 고정
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if deterministic: # True면 cudnn seed 고정 (정확한 재현 필요한거 아니면 제외)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def doTrain(model:nn.Module, 
            train_loader:DataLoader, 
            num_epoch:int, 
            optimizer:optim.Optimizer):
    criterion = nn.BCELoss()
    tr_acc = np.zeros(num_epoch)
    tr_loss = np.zeros(num_epoch)
    model.train()
    for epoch in range(num_epoch):
        correct, total, trn_loss = (0, 0, 0.0)
        for i, (x, y) in enumerate(train_loader,0):
            x, y = (a.to(DEVICE) for a in [x,y])
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y.float())
            loss.backward()
            optimizer.step()

            pred = (out>0.5).int()
            total += y.size(0)
            correct += (pred==y).sum().item()
            trn_loss += loss.item()

        tr_loss[epoch] = round(trn_loss/len(train_loader), 4)
        tr_acc[epoch] = round(100 * correct / total,4)
    return tr_acc, tr_loss

def doTest(model:nn.Module, test_loader:DataLoader):
    sigmoid = nn.Sigmoid()
    total = 0
    correct = 0
    preds = np.array([])
    targets = np.array([])
    with torch.no_grad():
        model.eval()
        for x, y in test_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            pred = model(x)
            pred = sigmoid(torch.squeeze(pred.data))
            predicted = (pred > 0.5).int()
            correct += (predicted==y).sum().item()
            total += y.size(0)
            preds = np.append(preds,pred.to('cpu').numpy())
            targets = np.append(targets,y.to('cpu').numpy())
    acc = round(100 * correct / total, 4)
    return acc, preds, targets

def SaveResults_mat(filepath, test_acc, test_preds, test_targets, tr_acc, tr_loss, num_batch, num_epoch, lr):
    path = './results/' + filepath + '.mat'
    io.savemat(path, {'acc':test_acc,'preds':test_preds,'targets':test_targets,
                          'tr_acc':tr_acc,'tr_loss':tr_loss,
                          'info':f'batch_{num_batch}, epoch_{num_epoch}, lr_{lr}'})