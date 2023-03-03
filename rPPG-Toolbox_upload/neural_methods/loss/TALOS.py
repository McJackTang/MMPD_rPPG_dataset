from __future__ import print_function, division
import torch
import matplotlib.pyplot as plt
import argparse, os
import pandas as pd
import numpy as np
import random
import math
from torchvision import transforms
from torch import nn


class TALOS(nn.Module):
    """
    The Neg_Pearson Module is from the orignal author of Physnet.
    Code of 'Remote Photoplethysmograph Signal Measurement from Facial Videos Using Spatio-Temporal Networks' 
    source: https://github.com/ZitongYu/PhysNet/blob/master/NegPearsonLoss.py
    """
    
    def __init__(self):
        super(TALOS, self).__init__()
        return

    def possibility(self,k):
        #-15<k<16,int class
        k_length = 10
        poss = 1/k_length
        return poss
    
    def padding(self,k,labels):
        if k>0:
            y_pad = nn.functional.pad(labels,(0,0,0,k))
            y_pad = y_pad[k:]
        elif k ==0:
            return labels
        else:
            y_pad = nn.functional.pad(labels,(0,0,abs(k),0))
            y_pad = y_pad[:k]
        return y_pad

    def forward(self, preds, labels):     
        MSE_LOSS =   nn.MSELoss()
        loss = 0
        k_length = 10
        loss_list = []
        for k in range(-4,6):
            shift_labels = self.padding(k,labels)
            # print(len(labels),len(shift_labels))
            # print(type(labels),labels.shape,labels)
            loss_list.append(MSE_LOSS(preds,shift_labels))
            # loss += MSE_LOSS(preds,shift_labels)*self.possibility(k)         
        # loss = loss/k_length
        loss = min(loss_list)
        return loss




