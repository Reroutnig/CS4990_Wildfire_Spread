import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler

from utils.tools import StandardScaler
from utils.timefeatures import time_features

import warnings
warnings.filterwarnings('ignore')

from datetime import datetime, timedelta


import ast

class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12*30*24 - self.seq_len, 12*30*24+4*30*24 - self.seq_len]
        border2s = [12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTm1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='t', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12*30*24*4 - self.seq_len, 12*30*24*4+4*30*24*4 - self.seq_len]
        border2s = [12*30*24*4, 12*30*24*4+4*30*24*4, 12*30*24*4+8*30*24*4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        
        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols=cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    '''from scratch!!'''
    def __read_data__(self):
        self.scaler = StandardScaler()

        # Load tensor files from directory
        path = 'tensorData/'
        pt_files = sorted([f for f in os.listdir(path) if f.endswith('.pt')])
        tensor_list = []
        num = 1
        for file in pt_files:
            if num == 365:
                break
            else:
                # if num < 366:
                #     tensor = torch.load(os.path.join(path + 'burn_2023_' + str(num) + '.pt'))  # Expecting shape (340, 220)
                # else:
                #     tensor = torch.load(os.path.join(path + 'burn_2024_' + str(num) + '.pt'))  # Expecting shape (340, 220)
                tensor = torch.load(os.path.join(path + 'burn_2024_' + str(num) + '.pt'))

                if tensor.shape != (340, 220):
                    raise ValueError(f"Unexpected shape {tensor.shape} in {file}")

                tensor_list.append(tensor)
            num +=1 

        # Stack tensors into shape (num_samples, 340, 220)
        combined_tensor = torch.stack(tensor_list, dim=0)  # torch.Tensor
        num_samples = combined_tensor.shape[0]
        print("numo fsameples: ", num_samples)

        # Split boundaries
        num_train = int(num_samples * 0.8)
        num_test = int(num_samples * 0.1)
        num_vali = num_samples - num_train - num_test

        border1s = [0, num_train - self.seq_len, num_samples - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, num_samples]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # print('comined anything?', combined_tensor[border1:border2])
        self.data_x = combined_tensor[border1:border2]
        # print(f"shape? 3? {self.data_x.shape}")
        if self.inverse:
            self.data_y = combined_tensor[border1:border2]
        else:
            self.data_y = combined_tensor[border1:border2]
        # self.data_stamp = np.arange(border1, border2)
        start_date = datetime(2024, 1, 1)  # Start date is the first day of 2024

        # Generate the corresponding dates for the days in the range
        dates = [start_date + timedelta(days=int(day_num)-1) for day_num in np.arange(border1, border2)]

        # Convert the dates into a pandas dataframe (this mimics df_stamp['date'])
        df_stamp = pd.DataFrame({'date': dates})

        # Now call your time_features function (assuming it's similar to time_features.py in the original repo)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        # data_stamp should now contain the same output as before, but using day numbers instead of the actual dates.
        self.data_stamp = data_stamp


    def __getitem__(self, index):
        # print('are you in here?')
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        # print(f'brother wat? {seq_x.shape}')
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    
    def __len__(self):
        print("hrlp", len(self.data_x))
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
    
class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols=cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        print("in here?")
        self.scaler = StandardScaler()

        # Load tensor files from directory
        path = 'tensorData/'
        pt_files = sorted([f for f in os.listdir(path) if f.endswith('.pt')])
        tensor_list = []

        for file in pt_files:
            tensor_path = os.path.join(path, file)
            tensor = torch.load(tensor_path)  # Expecting shape (340, 220)

            if tensor.shape != (340, 220):
                raise ValueError(f"Unexpected shape {tensor.shape} in {file}")

            tensor_list.append(tensor)

        # Stack tensors into shape (num_samples, 340, 220)
        combined_tensor = torch.stack(tensor_list, dim=0)  # torch.Tensor
        num_samples = combined_tensor.shape[0]

        # Split boundaries
        num_train = int(num_samples * 0.8)
        num_test = int(num_samples * 0.1)
        num_vali = num_samples - num_train - num_test

        border1s = [0, num_train - self.seq_len, num_samples - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, num_samples]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # print('comined anything?', combined_tensor[border1:border2])
        self.data_x = combined_tensor[border1:border2]
        # print(f"shape? 3? {self.data_x.shape}")
        if self.inverse:
            self.data_y = combined_tensor[border1:border2]
        else:
            self.data_y = combined_tensor[border1:border2]
        # self.data_stamp = np.arange(border1, border2)
        start_date = datetime(2024, 1, 1)  # Start date is the first day of 2024

        # Generate the corresponding dates for the days in the range
        dates = [start_date + timedelta(days=int(day_num)-1) for day_num in np.arange(border1, border2)]

        # Convert the dates into a pandas dataframe (this mimics df_stamp['date'])
        df_stamp = pd.DataFrame({'date': dates})

        # Now call your time_features function (assuming it's similar to time_features.py in the original repo)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        # data_stamp should now contain the same output as before, but using day numbers instead of the actual dates.
        self.data_stamp = data_stamp

    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin+self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin+self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols=cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        print("in here?")
        self.scaler = StandardScaler()

        # Load tensor files from directory
        path = 'tensorData/'
        pt_files = sorted([f for f in os.listdir(path) if f.endswith('.pt')])
        tensor_list = []
        num = 20
        for file in pt_files:
            if num == 365:
                break
            else:
                tensor = torch.load(os.path.join(path + 'burn_2024_' + str(num) + '.pt'))

                if tensor.shape != (340, 220):
                    raise ValueError(f"Unexpected shape {tensor.shape} in {file}")

                tensor_list.append(tensor)
            num +=1 

        # Stack tensors into shape (num_samples, 340, 220)
        combined_tensor = torch.stack(tensor_list, dim=0)  # torch.Tensor
        num_samples = combined_tensor.shape[0]
        print("numo fsameples: ", num_samples)

        # Split boundaries

        border1 = len(combined_tensor[0])-self.seq_len
        border2 = len(combined_tensor[0])

        # print('comined anything?', combined_tensor[border1:border2])
        self.data_x = combined_tensor[border1:border2]
        # print(f"shape? 3? {self.data_x.shape}")
        if self.inverse:
            self.data_y = combined_tensor[border1:border2]
        else:
            self.data_y = combined_tensor[border1:border2]

        start_date = datetime(2024, 1, 1)  # Start date is the first day of 2024
        dates = [start_date + timedelta(days=int(day_num)-1) for day_num in np.arange(border1, border2)]
        last_date = dates[-1]
        # Generate additional dates after the last date
        additional_dates = [last_date + timedelta(days=i) for i in range(1, self.seq_len + 1)]

        # Combine original dates with the additional dates
        all_dates = dates + additional_dates
        df_stamp = pd.DataFrame({'date': all_dates})
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_stamp = data_stamp

    
    def __getitem__(self, index):
        print(f"ARE YOU EVER IN HERE? -o- {index}")
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin+self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin+self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
