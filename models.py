import torch
import torch.nn as nn
import torch.functional as F

from torchvision.models import resnet50, ResNet50_Weights
from einops import repeat
from utils import DEVICE
class CNN_3D(nn.Module):
    def __init__(self) :
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(1,8,5,padding=5//2),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Conv3d(8,32,5,padding=5//2),
            nn.BatchNorm3d(32),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.block(x)
    
class M3T_model(nn.Module):
    def __init__(self) :
        super().__init__()

        # 3D CNN
        self.cnn3d = CNN_3D()

        # 2D CNN
        self.cnn2d = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.cnn2d.conv1 = nn.Conv2d(32, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.cnn2d.fc = nn.Sequential(
            nn.Linear(2048,512),
            nn.ReLU(),
            nn.Linear(512,256)
        )

        # Position and Plane Embedding
        self.cls_token = nn.Parameter(torch.rand(1,1,256))
        self.sep_token = nn.Parameter(torch.rand(1,1,256))

        # Transformer Encoder
        encoder = nn.TransformerEncoderLayer(256,8,768,activation='gelu',batch_first=True)
        self.transformer_enc = nn.TransformerEncoder(encoder,8)
        self.register_buffer("pos_idx",torch.arange(388))
        self.register_buffer("pln_idx",torch.tensor([0]+[x//129 for x in range(387)]))
        self.pos_emb = nn.Embedding(388,256)
        self.pln_emb = nn.Embedding(3,256)

        # Classification
        self.fc = nn.Sequential(
            nn.Linear(388*256,1),
            nn.Sigmoid()
        )

    def forward(self, x:torch.Tensor):
        # initial_input_shape = B*128*128*128

        # 1. 3D CNN
        x = x.unsqueeze(1)      # B*1*128*128*128
        x = self.cnn3d(x)       # B*32*128*128*128

        # 2. Multi-Slices & 2D CNN
        c = self.cnn2d(x[:,:,0,:,:]).unsqueeze(1)
        s = self.cnn2d(x[:,:,:,0,:]).unsqueeze(1)
        a = self.cnn2d(x[:,:,:,:,0]).unsqueeze(1)
        for i in range(1,128):
            c = torch.concat([c,self.cnn2d(x[:,:,i,:,:]).unsqueeze(1)],dim=1)   # 2dcnn out: [B*256]
            s = torch.concat([s,self.cnn2d(x[:,:,:,i,:]).unsqueeze(1)],dim=1)   # 
            a = torch.concat([a,self.cnn2d(x[:,:,:,:,i]).unsqueeze(1)],dim=1)   # c,saa: [B*128*256]
        
        # 3. Position and Plane Embedding
        cls_tokens = repeat(self.cls_token, '() n e -> b n e',b=x.shape[0])
        sep_tokens = repeat(self.sep_token, '() n e -> b n e',b=x.shape[0])
        out = torch.concat([cls_tokens,c,sep_tokens,s,sep_tokens,a,sep_tokens],dim=1)   # B*388*256
        pos_emb = self.pos_emb(self.pos_idx)    # 388*256
        pln_emb = self.pln_emb(self.pln_idx)    # 388*256
        out += pos_emb + pln_emb                # B*388*256

        # 4. Transformer Encoder
        out = self.transformer_enc(out)   # B*388*256

        # 5. Classification
        out = out.flatten(1)    # B*99328
        out = self.fc(out)      # B*1
        return torch.squeeze(out)