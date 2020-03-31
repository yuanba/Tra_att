from config import *
import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import LabelEncoder
import random
import pickle
import torch
import torch.utils.data as Data
from torch import nn
import torch.optim as optim
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"

sys.path.append("..")

from core.utils.geohash import bin_geohash
from core.utils.metrics import *

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch <= 1:
        lr = LR
    elif epoch>1 and epoch<=25:
        lr = LR * (0.87 ** ((epoch-1)))
    else:
        lr = LR * (0.85 ** (epoch-1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_trajectories(train_file, test_file, tid_col='tid',
                     label_col='label',geo_precision=8):
    df_train = pd.read_csv(train_file)[[tid_col,label_col,"poi"]]
    df_test = pd.read_csv(test_file)[[tid_col,label_col,"poi"]]
    df = df_train.copy().append(df_test)

    le1 = LabelEncoder()
    le1.fit(df["poi"])
    df_train["poi"] = le1.transform(df_train["poi"])
    df_test["poi"] = le1.transform(df_test["poi"])

    poi_max = max(df_train["poi"].max(),df_test["poi"].max())+10
    max_len = 0

    le2 = LabelEncoder()
    le2.fit(df[label_col])
    df_train[label_col] = le2.transform(df_train[label_col])
    df_test[label_col] = le2.transform(df_test[label_col])
    classes = df_test[label_col].max()+1

    x_train = []
    x_test = []
    y_train = []
    y_test = []

    for idx, tid in enumerate(set(df_train[tid_col])):
        traj = df_train.loc[df_train[tid_col].isin([tid])]
        max_len = max(max_len,len(list(traj["poi"])))
        x_train.append(list(traj["poi"]))
        y_train.append(list(traj[label_col])[0])

    for idx, tid in enumerate(set(df_test[tid_col])):
        traj = df_test.loc[df_test[tid_col].isin([tid])]
        max_len = max(max_len,len(list(traj["poi"])))
        x_test.append(list(traj["poi"]))
        y_test.append(list(traj[label_col])[0])

    x_train = torch.from_numpy(np.array([[0]*(max_len-len(x))+x for x in x_train])).long() #padding,(train_size,max_len)
    x_test = torch.from_numpy(np.array([[0]*(max_len-len(x))+x for x in x_test])).long() 
    y_train = torch.from_numpy(np.array(y_train)).long() 
    y_test = torch.from_numpy(np.array(y_test)).long() 

    return x_train,y_train,x_test,y_test,poi_max,max_len,classes

class GRU_Net(nn.Module):
    def __init__(self,x_train,y_train,x_test,y_test,poi_max,max_len,classes):
        super(GRU_Net, self).__init__()
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.poi_max = poi_max
        self.max_len = max_len
        self.classes = classes

        self.hidden_size = HIDDEN_SIZE
        self.embed_size = EMBEDDING_SIZE
        self.rnn = nn.GRU(     # LSTM 效果要比 nn.RNN() 好多了
            input_size=self.embed_size,      # 图片每行的数据像素点
            hidden_size=self.hidden_size,     # rnn hidden unit
            num_layers=1,       # 有几层 RNN layers
            batch_first=True,   # input & output 会是以 batch size 为第一维度的特征集 e.g. (batch, time_step, input_size)
            bidirectional=False
        )

        self.emb = nn.Embedding(self.poi_max,self.embed_size)
        self.mlp = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(self.hidden_size, self.classes),
        )

    def forward(self, x):
        emb_out = self.emb(x) #in:(batch_size,seq_len)   out:(batch_size,seq_len,emb_size)
        rnn_out,h_n = self.rnn(emb_out)  #in:(batch_size,seq_len,emb_size)  out:(batch_size,seq_len,hidden_size)  h_n:(batch_size,1,hidden_size)
        out = self.mlp(rnn_out[:, -1, :])
        # print(x.shape,emb_out.shape,rnn_out.shape,out.shape)
        return out

    def train_and_test(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=LR, weight_decay=LAMDA)   # optimize all parameters
        loss_func = nn.CrossEntropyLoss()   # the target label is not one-hotted
        if torch.cuda.is_available():
            self = self.cuda()

        train_dataset = Data.TensorDataset(self.x_train,self.y_train)
        train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_dataset = Data.TensorDataset(self.x_test,self.y_test)
        test_loader = Data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)

        for epoch in range(EPOCH):

            #adjust_learning_rate(optimizer, epoch)
            print("===== Training Epoch %d =====" % (epoch + 1))
            running_loss = 0.0

            self.train()
            for step, (batch_x, batch_y) in enumerate(train_loader):   # gives batch data
                # mod = 100
                # if step%mod==0:
                #     print("Trainging: EPOCH {}, finished {} batch, total {}".format(epoch, step, len(x_train)))

                fea,label = (batch_x, batch_y)
                if torch.cuda.is_available():
                    fea = fea.cuda()
                    label = label.cuda()
                output = self(fea)               # rnn output

                #record output and prediction for evaluation    
                if step==0:
                    y_true = label.cpu().detach().numpy().copy()
                    y_pred = output.cpu().detach().numpy().copy()
                else:
                    y_true = np.append(y_true,label.cpu().detach().numpy().copy())
                    y_pred = np.vstack((y_pred,output.cpu().detach().numpy().copy()))  

                loss = loss_func(output, label)   # cross entropy loss
                running_loss += loss.item()

                optimizer.zero_grad()           
                loss.backward()                 
                optimizer.step() 

            print('Finish {} epoch, Loss: {:.6f}'.format(
                epoch + 1, running_loss / self.x_train.shape[0] ))

            compute_acc_acc5_f1_prec_rec(y_true,y_pred,print_pfx="TRAIN")

            self.eval()
            with torch.no_grad():
                for step, (batch_x, batch_y) in enumerate(test_loader):   # gives batch data
                    fea,label = (batch_x, batch_y)
                    if torch.cuda.is_available():
                        fea = fea.cuda()
                        label = label.cuda()
                    output = self(fea)               # rnn output

                #record output and prediction for evaluation    
                if step==0:
                    y_true = label.cpu().detach().numpy().copy()
                    y_pred = output.cpu().detach().numpy().copy()
                else:
                    y_true = np.append(y_true,label.cpu().detach().numpy().copy())
                    y_pred = np.vstack((y_pred,output.cpu().detach().numpy().copy()))
                
                compute_acc_acc5_f1_prec_rec(y_true,y_pred,print_pfx="TEST")

if __name__=="__main__":
    x_train,y_train,x_test,y_test,poi_max,max_len,classes = get_trajectories(TRAIN_FILE,TEST_FILE)
    gru = GRU_Net(x_train,y_train,x_test,y_test,poi_max,max_len,classes)
    gru.train_and_test()