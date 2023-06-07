import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, in_size, n_hidden_layers, hidden_size, out_size):
        super(Generator,self).__init__()
    
        def block(in_feat, out_feat):
            layers = [nn.Linear(in_feat, out_feat)]
            layers.append(nn.LeakyReLU(0.1, inplace=True))
            return layers
    
        self.model = nn.Sequential(*block(in_size,hidden_size),
        *n_hidden_layers*block(hidden_size,hidden_size),
        nn.Linear(hidden_size,out_size),
        nn.Softmax(dim=-1))
    
    def forward(self,noise,prop):
        input = torch.cat((noise,prop),-1)
        x = self.model(input)
        return x

class Discriminator(nn.Module):
    def __init__(self, in_size, n_hidden_layers, hidden_size, out_size):
        super(Discriminator,self).__init__()

        def block(in_feat,out_feat):
            layers = [nn.Linear(in_feat,out_feat)]
            layers.append(nn.LeakyReLU(0.1, inplace=True))
            return layers
    
        self.model = nn.Sequential(*block(in_size,hidden_size),
        *n_hidden_layers*block(hidden_size,hidden_size),
        nn.Linear(hidden_size,out_size),
        nn.Sigmoid())
    
    def forward(self,comp,prop):
        input = torch.cat((comp,prop),-1)
        x = self.model(input)
        return x


