"""
Created on July 2021

@author: Nadia Brancati

"""
import torch.nn.functional as F
from min_max_layer import *

class AttentionModel(nn.Module):
    def __init__(self,num_classes,filters_out,filters_in,dropout,device):
        super(AttentionModel, self).__init__()
        self.filters_out=filters_out
        self.filters_in=filters_in
        self.dropout=dropout
        self.num_classes=num_classes
        self.device=device
        #3D convolution of the tensor
        self.conv = nn.Conv3d(in_channels=1, out_channels=self.filters_out, kernel_size=(512, 3, 3),
                               padding=(0, 1, 1))
        #computation of max and min pool layer
        self.min_max_layer=min_max_layer(2,device)
        #linear layer for the classification
        self.linear = nn.Linear(in_features=self.filters_in *self.filters_out*2,out_features=num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.drop_out=nn.Dropout(dropout)


    def _featureExtraction(self,X,AttentionMap):
        #reshape of tensors for the computation of the product between AttentionMap and input tensor X
        AttentionMap = AttentionMap.reshape((AttentionMap.shape[1], AttentionMap.shape[0], AttentionMap.shape[2], AttentionMap.shape[3], AttentionMap.shape[4]))
        X = X.reshape((X.shape[1], X.shape[0], X.shape[2], X.shape[3], X.shape[4]))

        ris = F.conv3d(input=X, weight=AttentionMap, bias=None, stride=1, padding=0, groups=1)

        ris = torch.squeeze(ris)
        ris = ris.reshape((ris.shape[0] * ris.shape[1]))
        return ris



    def forward(self, x):
        x1=self.conv(x)
        x1 = torch.squeeze(x1)
        #max and min pool computation
        M_mat,m_mat=self.min_max_layer(x1)
        #computation of Attention Map for Max and Min
        M_mat = F.softmax(M_mat.view((M_mat.shape[1], M_mat.shape[2] * M_mat.shape[3])), dim=1).view(M_mat.shape)
        m_mat = F.softmax(m_mat.view((m_mat.shape[1], m_mat.shape[2] * m_mat.shape[3])), dim=1).view(m_mat.shape)

        M_mat = M_mat.unsqueeze(2)
        m_mat = m_mat.unsqueeze(2)
        #product between Attention Map of Max and input tensor and between Attention Map of Min and input tensor
        x2 = self.relu(self._featureExtraction(x,M_mat))
        x3 = self.relu(self._featureExtraction(x, m_mat))
        #concat of the extracted features
        x4=torch.cat((x2, x3),0)
        #final classification
        output = self.drop_out(self.linear(x4))
        return output


