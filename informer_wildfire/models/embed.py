import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


import math

class PositionalEmbedding(nn.Module):
    # positioning each sequence, each row
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float() # the number of sequence, 96, i think right now, by 512 embedded values for the data
        pe.require_grad = False # not learnable

        position = torch.arange(0, max_len).float().unsqueeze(1) # we are going to get the column of 0 to the squence length
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return self.pe[:, :x.size(1)]


# used in the temporal for fixed data or not, i dont think that we need/are using
class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()

# may or may not be used for the temporal embedding, depends on the parameter
class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4; hour_size = 24
        weekday_size = 7; day_size = 32; month_size = 13

        Embed = FixedEmbedding if embed_type=='fixed' else nn.Embedding
        if freq=='t':
            self.minute_embed = Embed(minute_size, d_model)
        # mapping each of the sequences for the day to year or month or more to the dimension of our model
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)
    
    def forward(self, x):
        x = x.long()
        
        minute_x = self.minute_embed(x[:,:,4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:,:,3])
        weekday_x = self.weekday_embed(x[:,:,2])
        day_x = self.day_embed(x[:,:,1])
        month_x = self.month_embed(x[:,:,0])
        
        return hour_x + weekday_x + day_x + month_x + minute_x

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h':4, 't':5, 's':6, 'm':1, 'a':1, 'w':2, 'd':3, 'b':3}
        # this goes back to the time features (timefeatures.py) to see how the times are being used and calculated for our model
        d_inp = freq_map[freq]
        # this is 4 and 512
        self.embed = nn.Linear(d_inp, d_model)
    
    def forward(self, x):
        # print("what is this x", x)
        return self.embed(x)
    
class FireImageEmbedder(nn.Module):
    def __init__(self, d_model):
        super(FireImageEmbedder, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),  # (B*S, 16, 170, 110)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # (B*S, 16, 85, 55)
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # (B*S, 32, 43, 28)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # (B*S, 32, 1, 1)
        )

        self.fc = nn.Linear(32, d_model)

    def forward(self, x):
        # x shape: (B, S, H, W)
        B, S, H, W = x.shape
        x = x.view(B * S, 1, H, W)  # Add channel dim: (B*S, 1, H, W)
        x = self.cnn(x)             # -> (B*S, 32, 1, 1)
        x = x.view(B * S, -1)       # -> (B*S, 32)
        x = self.fc(x)              # -> (B*S, d_model)
        x = x.view(B, S, -1)        # -> (B, S, d_model)
        # print("fireIMageembed",x.shape)
        return x


# the main function that we call
class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = FireImageEmbedder(d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model) # getting the position of the sequence
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) if embed_type!='timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        # calling time feature embedding since we are dealing with time
        

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
        
        return self.dropout(x)