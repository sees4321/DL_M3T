import torch
import torch.optim as optim
import datetime

from utils import *
from models import M3T_model
from torch.utils.data import DataLoader

# Jang, Jinseong, and Dosik Hwang. 
# "M3T: three-dimensional Medical image classifier using Multi-plane and Multi-slice Transformer." 
# Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.

# set random seed
ManualSeed(2222)

# hyper-parameters
num_batch = 4
num_epoch = 50
learning_rate = 5e-5

# data => this code block need to be edited with actual data
data = torch.rand((100,128,128,128))
label = torch.zeros((100))
train_loader = DataLoader(data[:70],batch_size=num_batch,shuffle=False)
test_loader = DataLoader(data[70:],batch_size=num_batch,shuffle=False)

# train-test
model = M3T_model().to(DEVICE)
tr_acc, tr_loss = doTrain(model=model,
                          train_loader=train_loader,
                          num_epoch=num_epoch,
                          optimizer=optim.Adam(model.parameters(),lr=learning_rate,betas=(0.9,0.999)))
acc, predictions, targets = doTest(model,test_loader)

# save result
cur_time = datetime.datetime.now().strftime('%m%d_%H%M')
SaveResults_mat(f'result_{cur_time}',acc,predictions,targets,tr_acc,tr_loss,num_batch,num_epoch,learning_rate)