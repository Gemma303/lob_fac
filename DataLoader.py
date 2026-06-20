import torch
import torch.utils.data as torch_data
from torch.distributions import Normal

class NormalYDataset(torch_data.Dataset):
    def __init__(self, df,df_ret, feature_dim=40, length=16,horizon=5, interval=1, quantile=None):
        self.dates = df.index.get_level_values(0).drop_duplicates()
        self.interval = interval
        self.horizon = horizon
        self.quantile=quantile
        self.length=length
        self.feature_dim=feature_dim
        
        self.data = df
        self.data_y=df_ret
        self.df_length = len(self.dates)
        self.x_idx = self.get_x_idx()
        self.standard_normal=Normal(loc=0., scale=1.)

    def __getitem__(self, index):
        x_i = self.x_idx[index]
        date_i = self.dates[x_i]
        
        x_data = self.data.loc[date_i]
        y_data = self.data_y[x_i+2:x_i + self.horizon+2]
        
        secus=x_data.index.get_level_values(0).drop_duplicates()
        y_data = y_data.loc[:,secus].mean(0)

        if self.quantile:
            secus=secus[y_data>y_data.quantile(self.quantile)]
        else:
            secus = secus[y_data.notna()]
        x_data=x_data.loc[secus]
        y_data=y_data.loc[secus]
        
        mask = torch.tensor(x_data.notna().values).reshape(-1,self.length,self.feature_dim)
        x = torch.tensor(x_data.fillna(0).values,dtype=torch.float).reshape(-1,self.length,self.feature_dim)
        y = torch.tensor(y_data.values,dtype=torch.float)
        yrank=torch.zeros_like(y,dtype=torch.float)
        yrank[torch.argsort(y)]=self.standard_normal.icdf(torch.arange(1,len(y)+1,dtype=torch.float)/(len(y)+1))
        
        return x.unsqueeze(1), yrank, mask.unsqueeze(1)

    def __len__(self):
        return len(self.x_idx)

    def get_x_idx(self):
        x_index_set = range(0, self.df_length - self.horizon - 1, self.interval)
        x_idx = [x_index_set[j] for j in range(len(x_index_set))]
        return x_idx


class TestDataset(torch_data.Dataset):
    def __init__(self, df,feature_dim=40, length=16,horizon=5, interval=1):
        self.dates = df.index.get_level_values(0).drop_duplicates()
        self.interval = interval
        self.horizon = horizon
        self.length=length
        self.feature_dim=feature_dim
        
        self.data = df
        self.df_length = len(self.dates)
        self.x_idx = self.get_x_idx()
       
    def __getitem__(self, index):
        x_i = self.x_idx[index]
        date_i = self.dates[x_i]
        
        x_data = self.data.loc[date_i]
        secus=x_data.index.get_level_values(0).drop_duplicates()
        x_data=x_data.loc[secus]
        
        mask = torch.tensor(x_data.notna().values).reshape(-1,self.length,self.feature_dim)
        x = torch.tensor(x_data.fillna(0).values,dtype=torch.float).reshape(-1,self.length,self.feature_dim)
        
        return x.unsqueeze(1), mask.unsqueeze(1),secus,date_i

    def __len__(self):
        return len(self.x_idx)

    def get_x_idx(self):
        x_index_set = range(0, self.df_length - self.horizon - 1, self.interval)
        x_idx = [x_index_set[j] for j in range(len(x_index_set))]
        return x_idx