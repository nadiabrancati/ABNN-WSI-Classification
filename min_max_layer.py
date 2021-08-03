"""
Created on July 2021

@author: Nadia Brancati

"""
import torch.nn as nn
import torch

class min_max_layer(nn.Module):
    def __init__(self, pool_size,device):
        super().__init__()
        self._pool_size = pool_size
        self.device=device

    def forward(self, x):
        #computation of min pool
        #inversion of input
        x_min = -x
        min_pool = nn.MaxPool2d(self._pool_size, self._pool_size, return_indices=True)
        min_unpool = nn.MaxUnpool2d(self._pool_size, self._pool_size)
        x_min_unsqueeze = x_min.unsqueeze(0)
        x_min_res, index = min_pool(x_min_unsqueeze.type(torch.FloatTensor))
        x_min_fin = min_unpool(x_min_res, index,output_size=torch.Size([1,x_min.shape[0],x_min.shape[1], x_min.shape[2]]))
        x_min_fin=-x_min_fin
        x_min_fin = x_min_fin.type(torch.FloatTensor)
        #computation of maxpool
        max_pool = nn.MaxPool2d(self._pool_size, self._pool_size, return_indices=True)
        max_unpool = nn.MaxUnpool2d(self._pool_size, self._pool_size)
        x_max_unsqueeze = x.unsqueeze(0)
        x_max_res, index = max_pool(x_max_unsqueeze.type(torch.FloatTensor))
        x_max_fin = max_unpool(x_max_res, index,output_size=torch.Size([1,x.shape[0],x.shape[1], x.shape[2]]))
        x_max_fin = x_max_fin.type(torch.FloatTensor)
        return x_max_fin.cuda(self.device),x_min_fin.cuda(self.device)

