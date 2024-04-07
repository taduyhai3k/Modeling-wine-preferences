import pandas as pd
import numpy as np 
import torch 
import model
import processdata as proda
import datetime
import random

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
        
    def build_model(self):
         self.model = model.MainModel(self.agrs.input_shape, self.agrs.d_model, self.agrs.d_atten, self.agrs.num_of_head,
                                     self.agrs.num_of_l2, self.agrs.num_of_multi, self.agrs.output_shape, self.agrs.type_active)  
                  
    def getdata(self):
        raw_data = pd.read_csv(self.agrs.filepath, sep= ",") 
        target = raw_data[raw_data.columns[self.agrs.target]]
        raw_data = raw_data.to_numpy(np.float32)
        list_quality = {int(i) : [] for i in target.unique()}
        for i in range(len(target)):
            list_quality[int(target[i])].append(raw_data[i])
        max1 = max([len(list_quality[i]) for i in target.unique()])            
        if self.agrs.balance_data:
            for i in target.unique():
                list_quality[i] += random.choices(list_quality[i], k = max1 - len(list_quality[i]))     
        self.data = proda.GetDataTrainTest(list_quality, 5)    
    def train(self):
        self.build_model()
        self.model.train()
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr = 0.001, betas = (0.9, 0.999))               
        sum_loss = []
        for i in range(self.agrs.epoch):     
            self.model.train()       
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)             
            for feature, label in self.train_data:
                optimizer.zero_grad()                
                output = self.model(feature)
                loss = loss_fn(output, label)
                sum_loss.append(loss.item()) 
                loss.backward()
                optimizer.step()
            scheduler.step()    
            if i % 4 == 0:
                print(f"Vòng lặp thứ {i}, giá trị hàm train loss {np.mean(np.array(sum_loss))}")
                self.test()   
        return sum_loss    
    def test(self):
        self.model.eval()
        matrix = np.zeros(shape=(self.agrs.output_shape, self.agrs.output_shape), dtype= np.uint8)
        loss = []
        with torch.no_grad():
            for feature, label in self.test_data:
                output = self.model(feature)
                output = np.array([torch.argmax(i).item() for i in output], dtype = np.int16, ndmin= 1)
                label = label.detach().numpy()
                for i in range(len(label)):
                    matrix[output[i]][label[i]] += 1 
                loss.append(np.sum(np.abs(output - label)) / len(output))
            print(f"Giá trị hàm test loss {np.mean(np.array(loss))}")    
            print(matrix)
        return np.mean(np.array(loss))   
    
    def save(self, epoch):
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_dict = self.model.state_dict()  
        checkpoint_path = f'checkpoint_epoch_{epoch}_time_{current_time}.pth'
        checkpoint = {
            'model_state_dict': model_dict,
            'fold': epoch,
            'training_time': current_time,
        }
        torch.save(checkpoint, checkpoint_path)
    
    def train_test_loop(self):
        array_loss_train = []
        array_loss_test = []
        for i in range(self.agrs.k_fold):
            data_tmp = next(self.data)
            self.train_data =  proda.DataLoader(proda.MyData(data_tmp[0]), batch_size= self.agrs.batch_size, shuffle = True)
            self.test_data = proda.DataLoader(proda.MyData(data_tmp[1]),batch_size= self.agrs.batch_size, shuffle = True)
            array_loss_train.append(self.train())
            #array_loss_test.append(self.test())
            self.save(i)
            