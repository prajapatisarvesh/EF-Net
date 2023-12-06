'''
LAST UPDATE: 2023.09.20
Course: CS7180
AUTHOR: Sarvesh Prajapati (SP), Abhinav Kumar (AK), Rupesh Pathak (RP)

E-MAIL: prajapati.s@northeastern.edu, kumar.abhina@northeastern.edu, pathal.r@northeastern.edu
DESCRIPTION: 


'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod

class BaseModel(nn.Module):
    '''
    '''
    @abstractmethod
    def forward(self, *inputs):
        '''
        To be implemented by module
        '''
        raise NotImplementedError
    

    def __str__(self):
        '''
            Prints out model's trainable parameter
        '''
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_params])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)