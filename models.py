import torch
import torch.nn as nn
import torch.nn.functional as F
from new_models import New_AE

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class Autoencoder(nn.Module):
    def __init__(self, in_feature, embed, linear=True):
        super(Autoencoder, self).__init__()
        # encoder
        self.enc1 = nn.Linear(in_features=in_feature, out_features=embed, bias=False).to(device)
        # decoder
        self.dec1 = nn.Linear(in_features=embed, out_features=in_feature, bias=False).to(device)
        self.linear = linear
        # self.double()

    def forward(self, x):
        x = self.enc1(x).to(device)
        if self.linear == False:
            x = F.relu(x).to(device)
        x = self.dec1(x).to(device)
        return x
      
    
class TensorisedAEloss(nn.Module):
# Added the shared variable, which specifies if the encoder or decoder are shared between the clusters
    def __init__(self, in_feature, fst_embed, scnd_embed, reg, num_clusters=2, linear=True, CNN=False):
        super(TensorisedAEloss, self).__init__()
        self.AE = nn.ModuleList()
        # add num_clusters AE
        self.n_clust = num_clusters
        for i in range(num_clusters):
            # note: this should be written better so that the network is passed
            if CNN:
                self.AE.append(CNN_Autoencoder(embed))
            else:
                if fst_embed==scnd_embed:
                    self.AE.append(Autoencoder(in_feature,scnd_embed, linear))
                else:
                    self.AE.append(New_AE(in_feature, fst_embed,scnd_embed, linear))

        self.mse = nn.MSELoss()
        self.reg = reg
        self.CNN = CNN
        self.fst_embed = fst_embed
        self.scnd_embed = scnd_embed

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

                l = self.mse(self.AE[j](x), x_out) + (self.reg * torch.square(torch.norm(self.AE[j].enc2(self.AE[j].enc1(x)))))
            else:
                if self.fst_embed==self.scnd_embed: 
                    l = self.mse(self.AE[j](x), x_out) + (self.reg * torch.square(torch.norm(self.AE[j].enc1(x))))
                else:
                    l = self.mse(self.AE[j](x), x_out) + (self.reg * torch.square(torch.norm(self.AE[j].enc2(self.AE[j].enc1(x)))))
    
            if loss_clust > l:
                loss_clust = l
                loss_clust_idx = j
            l = clust_assign[j][i] * l
            loss += l
        return loss, loss_clust_idx

class CNN_Autoencoder(nn.Module):
    def __init__(self,embed):
        super(CNN_Autoencoder,self).__init__()
        self.enc1 = nn.Sequential(
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
        self.dec1 = nn.Sequential(
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
        # x = x.reshape(28, 28)
        # x = torch.unsqueeze(x, dim=0)
        # x = torch.unsqueeze(x, dim=0)
        enc = self.enc1(x)
        # print("encoded shape", enc.shape())
        dec = self.dec1(enc)
        return dec
    
    