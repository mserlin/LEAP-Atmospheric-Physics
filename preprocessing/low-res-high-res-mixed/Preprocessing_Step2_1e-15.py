'''
Reads the subsampled low-res csv dataset to calculate standardization parameters for the inputs

Then reads the full low-res and subsampled high-res data, standardizes it, and resaves it as hickel files
'''

import gc
import os
import random
import time
import torch
import datetime
import numpy as np
import pandas as pd
import polars as pl
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchmetrics.regression import R2Score
from tqdm import tqdm
import torch.nn.functional as F
import hickle as hkl
from torch.nn import AvgPool1d 
import torch.nn as nn

from torch.nn import LSTM, Conv1d, TransformerEncoder, TransformerEncoderLayer
from torch.nn import LayerNorm

from matplotlib.pyplot import plot

def format_time(elapsed):
    """Take a time in seconds and return a string hh:mm:ss."""
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

#Data path for the subsampled low-res data
DATA_PATH = 'C:/LEAP/'

df = pd.read_parquet(DATA_PATH+'train.csv')
df_test = pd.read_csv(DATA_PATH+'test.csv')

#Define the features
seq_fea_list = ['state_t','state_q0001','state_q0002','state_q0003','state_u','state_v','pbuf_ozone','pbuf_CH4','pbuf_N2O']
num_fea_list = ['state_ps','pbuf_SOLIN','pbuf_LHFLX','pbuf_SHFLX','pbuf_TAUX','pbuf_TAUY','pbuf_COSZRS','cam_in_ALDIF','cam_in_ALDIR','cam_in_ASDIF','cam_in_ASDIR','cam_in_LWUP','cam_in_ICEFRAC','cam_in_LANDFRAC','cam_in_OCNFRAC','cam_in_SNOWHLAND']

seq_y_list = ['ptend_t','ptend_q0001','ptend_q0002','ptend_q0003','ptend_u','ptend_v']
num_y_list = ['cam_out_NETSW','cam_out_FLWDS','cam_out_PRECSC','cam_out_PRECC','cam_out_SOLS','cam_out_SOLL','cam_out_SOLSD','cam_out_SOLLD']

seq_fea_expand_list = []
for i in seq_fea_list:
    for j in range(60):
        seq_fea_expand_list.append(i+'_'+str(j))

seq_y_expand_list = []
for i in seq_y_list:
    for j in range(60):
        seq_y_expand_list.append(i+'_'+str(j))
        
TARGET_COLS = seq_y_expand_list + num_y_list
FEAT_COLS = seq_fea_expand_list + num_fea_list



ts = time.time()

#Get the submission weights from the sample submission file
weights = pd.read_csv(DATA_PATH + "sample_submission.csv", nrows=1)
del weights['sample_id'] #where the sample_id can be ignored
weights = weights.T
weights = weights.to_dict()[0]

#Apply the weights to the dataset before determining the standardization parameters
for target in tqdm(weights):
    # print(target)
    df[target] = (df[target]*weights[target])

print("Time to read dataset:", format_time(time.time()-ts), flush=True)

#Apply a log transform to the following columns
LOG_COLS = ['state_q0001','state_q0002','state_q0003','pbuf_ozone','pbuf_CH4','pbuf_N2O']
for i in tqdm(LOG_COLS):
    for j in range(60):
        df[i+'_'+str(j)] = (np.log(df[i+'_'+str(j)]+1e-7)) #Add 1e-7 to avoid log(0) 
        df_test[i+'_'+str(j)] = (np.log(df_test[i+'_'+str(j)]+1e-7)) #Add 1e-7 to avoid log(0) 

gc.collect()

#Initialize a dictionary to store all the normalization values
norm_dict = dict() 

#Loop through all the features and calculate mean and std
for i in seq_fea_list:
    inter_list = []
    for j in range(60):
        inter_list.append(i+'_'+str(j))
    mean_value = df[inter_list].values.mean()
    std_value = df[inter_list].values.std()
    norm_dict[i] = [mean_value,std_value]
    
for i in num_fea_list:
    mean_value = df[i].values.mean()
    std_value = df[i].values.std()
    norm_dict[i] = [mean_value,std_value]



#Minimum standardization values
MIN_STD = 1e-10

#Path to the low res data
DATA_PATH = 'C:/kaggle/CD 1/'

#Loop through all the years (index i) and months (index j) of data
for i1 in tqdm(range(1,9)):
    if i1 == 1:
        lower = 2
    else:
        lower = 1
    for j1 in range(lower,13):
        df_i_j = pd.read_parquet(DATA_PATH+'c_data_'+str(i1)+'_'+str(j1)+'.parquet')
        for target in tqdm(weights):
            # print(target)
            df_i_j[target] = (df_i_j[target]*weights[target])

        #log transform the appropriate columns
        LOG_COLS = ['state_q0001','state_q0002','state_q0003','pbuf_ozone','pbuf_CH4','pbuf_N2O']
        for i in tqdm(LOG_COLS):
            for j in range(60):
                df_i_j[i+'_'+str(j)] = (np.log(df_i_j[i+'_'+str(j)]+1e-7))

        #Apply the computed standardization
        for i in seq_fea_list:
            inter_list = []
            for j in range(60):
                inter_list.append(i+'_'+str(j))

            df_i_j[inter_list] = ((df_i_j[inter_list]-norm_dict[i][0])/(MIN_STD+norm_dict[i][1])).astype('float32')

        for i in num_fea_list:
            df_i_j[i] = (df_i_j[i]-norm_dict[i][0])/(MIN_STD+norm_dict[i][1]).astype('float32')

        #Save the data as hickel files
        x_train_i_j = df_i_j[seq_fea_expand_list+num_fea_list].values.astype(np.float32)
        y_train_i_j = df_i_j[seq_y_expand_list+num_y_list].values.astype(np.float32)
        hkl.dump(x_train_i_j, DATA_PATH+'c_data_x_'+str(i1)+'_'+str(j1)+'_v1_1e-20.hkl')
        hkl.dump(y_train_i_j, DATA_PATH+'c_data_y_'+str(i1)+'_'+str(j1)+'_v1_1e-20.hkl')


DATA_PATH = 'C:/kaggle/CD 1/'

for i1 in tqdm(range(1,9)):
    if i1 == 1:
        lower = 2
    else:
        lower = 1
    for j1 in range(lower,13):
        df_i_j = pd.read_parquet(DATA_PATH+'c_data_'+str(i1)+'_'+str(j1)+'.parquet')
        for target in tqdm(weights):
            # print(target)
            df_i_j[target] = (df_i_j[target]*weights[target])
                
        LOG_COLS = ['state_q0001','state_q0002','state_q0003','pbuf_ozone','pbuf_CH4','pbuf_N2O']
        for i in tqdm(LOG_COLS):
            for j in range(60):
                df_i_j[i+'_'+str(j)] = (np.log(df_i_j[i+'_'+str(j)]+1e-7))

        for i in seq_fea_list:
            inter_list = []
            for j in range(60):
                inter_list.append(i+'_'+str(j))

            df_i_j[inter_list] = ((df_i_j[inter_list]-norm_dict[i][0])/(MIN_STD+norm_dict[i][1])).astype('float32')

        for i in num_fea_list:
            df_i_j[i] = (df_i_j[i]-norm_dict[i][0])/(MIN_STD+norm_dict[i][1]).astype('float32')

        x_train_i_j = df_i_j[seq_fea_expand_list+num_fea_list].values.astype(np.float32)
        y_train_i_j = df_i_j[seq_y_expand_list+num_y_list].values
        y_train_i_j = (y_train_i_j - my.reshape(1,-1)) / sy.reshape(1,-1)
        y_train_i_j = y_train_i_j.astype(np.float32)
        hkl.dump(x_train_i_j, DATA_PATH+'c_data_x_'+str(i1)+'_'+str(j1)+'_v1_1e-20.hkl')
        hkl.dump(y_train_i_j, DATA_PATH+'c_data_y_'+str(i1)+'_'+str(j1)+'_v1_1e-20.hkl')
