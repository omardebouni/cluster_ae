import torch
import torch.nn as nn
import torch.nn.functional as F
from models import *

device = 'cpu' #if torch.backends.mps.is_available() else 'cpu'
        
class PartSharedLoss(nn.Module):
    def __init__(self, in_feature, fst_embed, scnd_embed, reg, num_clusters=2, linear=True, CNN=False, sig=False):
        super(PartSharedLoss, self).__init__()
        print("partly shared architecture")
        self.n_clust = num_clusters
        self.CNN = CNN

        # Shared part
        if CNN:
            self.shared_encoder = CNN_Encoder(fst_embed)
            self.shared_decoder = CNN_Decoder(fst_embed)
        else: 
            self.shared_encoder = Encoder(in_feature, fst_embed, linear)
            self.shared_decoder = Decoder(fst_embed, in_feature,sig=sig)

        # Tensorized part
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(num_clusters):
            self.encoders.append(Encoder(fst_embed, scnd_embed, linear))
            self.decoders.append(Decoder(scnd_embed, fst_embed))


        self.mse = nn.MSELoss()
        self.reg = reg
        
    def forward(self, X, centers, i, clust_assign, X_out=None):
        if X_out == None:
            X_out = X

        loss = 0
        loss_clust_idx = -1
        loss_clust = torch.inf


        for j in range(self.n_clust):
            x = X - centers[j]
            x_out = X_out - centers[j]
            
            
            if self.CNN:
                # this can def be optimized
                x = x.reshape(28, 28)
                x = torch.unsqueeze(x, dim=0)
                x = torch.unsqueeze(x, dim=0)

                x_out = x_out.reshape(28, 28)
                x_out = torch.unsqueeze(x_out, dim=0)
                x_out = torch.unsqueeze(x_out, dim=0)



            encoded1 = self.shared_encoder(x)
            encoded2 = self.encoders[j](encoded1)
            reconstructed1 = self.decoders[j](encoded2)
            reconstructed2 = self.shared_decoder(reconstructed1)


            l = self.mse(reconstructed2, x_out) + (self.reg * torch.square(torch.norm(encoded2)))

            if loss_clust > l:
                loss_clust = l
                loss_clust_idx = j
            l = clust_assign[j][i] * l

            loss += l
        return loss, loss_clust_idx


class New_AE(nn.Module):
    def __init__(self, in_feature, fst_embed, scnd_embed, linear=True):
        super(New_AE, self).__init__()
        # encoder
        self.enc1 = nn.Linear(in_features=in_feature, out_features=fst_embed, bias=False).to(device)
        self.enc2 = nn.Linear(in_features=fst_embed, out_features=scnd_embed, bias=False).to(device)
        # decoder
        self.dec1 = nn.Linear(in_features=scnd_embed, out_features=fst_embed, bias=False).to(device)
        self.dec2 = nn.Linear(in_features=fst_embed, out_features=in_feature, bias=False).to(device)
        self.linear = linear
        # self.double()

    def forward(self, x):
        x = self.enc1(x).to(device)
        x = self.enc2(x).to(device)
        if self.linear == False:
            x = F.relu(x).to(device)
        x = self.dec1(x).to(device)
        x = self.dec2(x).to(device)
        return x

    
class Encoder(nn.Module):
    def __init__(self, in_feature, embed, linear=True):
        super(Encoder, self).__init__()
        self.enc1 = nn.Linear(in_features=in_feature, out_features=embed, bias=False)
        self.linear = linear
        nn.init.orthogonal_(self.enc1.weight)

    def forward(self, x):
        x = self.enc1(x).to(device)
        if not self.linear:
            x = F.relu(x).to(device)
        return x

class Decoder(nn.Module):
    def __init__(self, embed, out_feature, sig=False):
        super(Decoder, self).__init__()
        self.dec1 = nn.Linear(in_features=embed, out_features=out_feature, bias=False).to(device)
        self.sig=sig

    def forward(self, x):
        x = self.dec1(x).to(device)
        if self.sig:
            x = F.sigmoid(x).to(device)
        return x
    
class CNN_Encoder(nn.Module):
    def __init__(self, embed):
        super(CNN_Encoder,self).__init__()
        self.encoder = nn.Sequential(
            # 28 x 28
            nn.Conv2d(1, 4, kernel_size=5),
            # 4 x 24 x 24
            nn.ReLU(True),
            nn.Conv2d(4, 8, kernel_size=5),
            nn.ReLU(True),
            # 8 x 20 x 20 = 3200
            nn.Flatten(),
            nn.Linear(3200, embed),
            # 10
            # nn.Softmax(),
            )
    def forward(self, x):
            enc = self.encoder(x)
            return enc
    
class CNN_Decoder(nn.Module):
    def __init__(self, embed):
        super(CNN_Decoder,self).__init__()
        self.decoder = nn.Sequential(
            # 10
            nn.Linear(embed, 400),
            # 400
            nn.ReLU(True),
            nn.Linear(400, 4000),
            # 4000
            nn.ReLU(True),
            nn.Unflatten(1, (10, 20, 20)),
            # 10 x 20 x 20
            nn.ConvTranspose2d(10, 10, kernel_size=5),
            # 24 x 24
            nn.ConvTranspose2d(10, 1, kernel_size=5),
            # 28 x 28
            nn.Sigmoid(),
            )
    def forward(self, x):
            dec = self.decoder(x)
            return dec
 