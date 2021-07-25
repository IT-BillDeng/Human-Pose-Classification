import os
import math
import pandas as pd
import numpy as np


#数据读入
#统一按350帧读入 不足350帧的用均值插帧
#最后读成n*350*68与n*5的列表 n为数组个数
#
class Data_load:
    def __init__(self, File="./data/train/00"):
        super().__init__()
        self.data = []
        self.labels = []
        self.seqlen = []
        flag = False
        for i in range(5):
            for root, dirs, files in os.walk(File+str(i)):
                for file in files:
                    # print(file)
                    S = []
                    X = np.load(root+"/"+file,"r")
                    Len = len(X[0,0,:,0,0])
                    L = math.ceil(350/Len)
                    cnt = 350-Len
                    for m in range(len(X[0,0,:,0,0])):
                        for j in range(17):
                            for k in range(2):
                                if not j and not k:
                                    AT = np.hstack((X[0,0,m,j,k],X[0,2,m,j,k]))                                
                                else:
                                    AT = np.hstack((AT, X[0,0,m,j,k]))
                                    AT = np.hstack((AT, X[0,2,m,j,k]))
                        if m==0:
                            S += AT.reshape(1,68).tolist()
                            AO = AT
                        else:
                            for j in range(L-1):
                                LS = AO+((j+1)*(AT-AO))/L
                                if cnt:
                                    S += LS.reshape(1,68).tolist()
                                    cnt -= 1
                                else:
                                    break
                            S += AT.reshape(1,68).tolist()
                            AO = AT
                    for j in range(350-len(S)):
                        S += AT.reshape(1,68).tolist()
                    # print(np.array(S).shape) #(350,68)
                    self.data.append(S)
                    if X[0,:,:,:,1].any() :
                        print(i)
                    cnt = [0,0,0,0,0]
                    cnt[i] = 1
                    self.labels.append(cnt)
                    self.seqlen.append(350)
    
    def returnData(self):
        Data = self.data
        Labels = self.labels
        Seqlen = self.seqlen
        return Data, Labels, Seqlen
