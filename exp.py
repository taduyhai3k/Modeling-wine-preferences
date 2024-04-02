import pandas as pd
import numpy as np 
import torch 
import model
import processdata as proda

class Exp_Basic(object):
    def __init__(self, agrs):
        self.agrs = agrs
    def train(self):
        pass
    def test(self):
        pass    
    def save(self):
        pass
    def load(self):
        pass
    def getdata(self):
        pass
    
class Exp_Wine(Exp_Basic):
    def __init__(self, agrs):
        super().__init__(agrs)
        self.model = model.MainModel(self.agrs.input_shape, self.agrs.d_model, self.agrs.d_atten, self.agrs.num_of_head,
                                     self.agrs.num_of_l2, self.agrs.num_of_multi, self.agrs.output_shape, self.type_active)
        self.data = []
        self.train = []
        self.test = [] 
    def getdata(self):
        raw_data = pd.read_csv(self.agrs.filepath, sep= ",") 
        target = raw_data[raw_data.columns[self.agrs.target]]
        raw_data = raw_data.to_numpy(np.float32)
        list_quality = {int(i) : [] for i in target.unique()}
        for i in range(len(target)):
            list_quality[int(target[i])].append(raw_data[i])
        self.data = proda.GetDataTrainTest(list_quality, 5)    
    def train(self):
        self.model.train()
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters, lr = 0.001, betas = (0.9, 0.999))    
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)    
        sum_loss = []
        for i in range(20):            
            for feature, label in self.train:
                optimizer.zero_grad()                
                output = self.model(feature)
                loss = loss_fn(output, label)
                sum_loss.append(loss) 
                loss.backward()
                optimizer.step()
            scheduler.step()    
        return sum_loss    
    def test(self):
        self.model.eval()
        loss = []
        for feature, label in self.test:
            output = self.model(feature)
            output = np.array([torch.argmax(i).item() for i in output], dtype = np.int16, ndmin= 1)
            label = label.detach().numpy() 
            loss.append(np.sum(np.abs(output - label)) / len(output))
        return np.mean(np.array(loss))    