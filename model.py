import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

class InputEmbedding(nn.Module):
    def __init__(self, d_model, input_shape)-> None:
        super().__init__()
        self.d_model = d_model
        self.input_shape = input_shape
        self.linear = torch.randn(size=[self.input_shape[1], self.d_model], dtype = torch.float32, requires_grad= True)
    def forward(self,x):
        return torch.matmul(x, self.linear)     

class FullyConnected(nn.Module):
    def __init__(self, input_shape, num_of_l2, output_shape, type_active = "softmax"):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.num_of_l2 = num_of_l2
        self.type_active = type_active
        self.linear1 = nn.Linear(in_features= self.input_shape, out_features= self.num_of_l2, dtype= torch.float32)
        self.linear2 = nn.Linear(in_features= self.num_of_l2, out_features=self.output_shape, dtype=torch.float32)
        if self.type_active == "softmax":
            self.active = nn.Softmax(dim = 2)
        else:
            self.active = nn.Sigmoid()
    def forward(self, x):
        tmp = self.linear1(x)
        tmp = F.relu(tmp)
        tmp = self.linear2(tmp)
        return tmp
        #return self.active(tmp)
    
class MultiHeadAtten(nn.Module):
    def __init__(self, input_shape, d_atten, num_of_head, FullyConect = True):
        super().__init__()
        self.input_shape = input_shape
        self.d_atten = d_atten
        self.num_of_head = num_of_head
        self.fullconnect = FullyConect
        self.linear_q = torch.randn(size = [self.input_shape[1], self.d_atten * self.num_of_head],dtype = torch.float32, requires_grad= True)
        self.linear_k = torch.randn(size = [self.input_shape[1], self.d_atten * self.num_of_head],dtype = torch.float32, requires_grad= True)   
        self.linear_v = torch.randn(size = [self.input_shape[1], self.d_atten * self.num_of_head],dtype = torch.float32, requires_grad= True)
        self.linear = torch.randn(size = [self.d_atten * self.num_of_head, self.input_shape[1]], dtype = torch.float32, requires_grad= True)
        self.layernorm = nn.LayerNorm(self.input_shape[1])
        if self.fullconnect:
            self.linear1 = torch.randn(size = [self.input_shape[1], self.input_shape[1]], dtype = torch.float32, requires_grad = True)
            self.linear2 = torch.randn(size = [self.input_shape[1], self.input_shape[1]], dtype = torch.float32, requires_grad = True)
        
    
    def forward(self,x):
        Q = torch.matmul(x, self.linear_q).transpose(1,2).view(x.shape[0],self.num_of_head,self.d_atten,self.input_shape[0]).transpose(2,3)
        K = torch.matmul(x, self.linear_k).transpose(1,2).view(x.shape[0],self.num_of_head,self.d_atten,self.input_shape[0]).transpose(2,3)
        V = torch.matmul(x, self.linear_v).transpose(1,2).view(x.shape[0],self.num_of_head,self.d_atten,self.input_shape[0]).transpose(2,3)
        #print(torch.matmul(torch.matmul(Q, K.transpose(1,2)).softmax(dim = 2), V))
        V = torch.matmul((torch.matmul(Q, K.transpose(2,3))/ np.sqrt(self.d_atten)).softmax(dim = 3), V).transpose(1,2).reshape(x.shape[0],self.input_shape[0], self.num_of_head * self.d_atten)
        V = torch.matmul(V, self.linear)
        V = V + x
        V = self.layernorm(V)
        if self.fullconnect:
            V1 = F.relu(torch.matmul(V, self.linear1))
            return self.layernorm(torch.matmul(V1, self.linear2) + V)
        else:
            return V

class MainModel(nn.Module):
    def __init__(self, input_shape, d_model, d_atten, num_of_head, num_of_l2, num_of_multi, output_shape, type_active = "softmax"):
        super().__init__()
        self.input_shape = input_shape
        self.d_model = d_model
        self.d_atten = d_atten
        self.num_of_head = num_of_head
        self.num_of_multi = num_of_multi
        self.output_shape = output_shape
        self.type_active = type_active
        self.num_of_l2 = num_of_l2
        self.InEm = InputEmbedding(self.d_model, self.input_shape)
        self.FuCon = FullyConnected(self.d_model, type_active= self.type_active, output_shape= self.output_shape,num_of_l2=self.num_of_l2)
        self.Stack_MultiHead = nn.ModuleList([MultiHeadAtten([self.input_shape[0], self.d_model], d_atten= self.d_atten, num_of_head= self.num_of_head) for _ in range(self.num_of_multi)])
        self.AvgPooling = nn.AvgPool2d(kernel_size= [self.input_shape[0],1], padding = 0, stride=[1,1])   
           
    
    def forward(self,x):                     
        inputem = self.InEm(x)
        for layer in self.Stack_MultiHead:
            inputem = layer(inputem)
        return self.FuCon(self.AvgPooling(inputem)).reshape(inputem.shape[0], self.output_shape)
                    