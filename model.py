import torch
import torch.nn as nn
import torch.nn.functional as F

class deeplob(nn.Module):
    def __init__(self, y_len,conv_dropout=0.3,inp_dropout=0.2,conv_c=4,inp_c=8,gru_k=16,gru_layers=1):
        super().__init__()
        self.y_len = y_len
        self.conv_dropout=conv_dropout
        self.inp_dropout=inp_dropout
        
        # convolution blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=conv_c, kernel_size=(1,2), stride=(1,2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(conv_c),
            nn.Conv2d(in_channels=conv_c, out_channels=conv_c, kernel_size=(2,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(conv_c),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=conv_c, out_channels=conv_c, kernel_size=(1,2), stride=(1,2)),
            nn.Tanh(),
            nn.BatchNorm2d(conv_c),
            nn.Conv2d(in_channels=conv_c, out_channels=conv_c, kernel_size=(2,1)),
            nn.Tanh(),
            nn.BatchNorm2d(conv_c),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=conv_c, out_channels=conv_c, kernel_size=(1,10)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(conv_c),
            nn.Conv2d(in_channels=conv_c, out_channels=conv_c, kernel_size=(2,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(conv_c),
        )
        
        # inception modules
        self.inp1 = nn.Sequential(
            nn.Conv2d(in_channels=conv_c, out_channels=inp_c, kernel_size=(1,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(inp_c),

            nn.Dropout2d(p=self.inp_dropout),
            nn.Conv2d(in_channels=inp_c, out_channels=inp_c, kernel_size=(3,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(inp_c),
        )
        self.inp2 = nn.Sequential(
            nn.Conv2d(in_channels=conv_c, out_channels=inp_c, kernel_size=(1,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(inp_c),
            
            nn.Dropout2d(p=self.inp_dropout),
            nn.Conv2d(in_channels=inp_c, out_channels=inp_c, kernel_size=(5,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(inp_c),
        )
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(in_channels=conv_c, out_channels=inp_c, kernel_size=(1,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(inp_c),
        )
        
        # GRU layer & Linear Transformation
        self.gru = nn.GRU(input_size=3*inp_c, hidden_size=gru_k, num_layers=gru_layers, batch_first=True)
        self.fc1 = nn.Linear(gru_k, self.y_len)
        self.dropout = nn.Dropout(p=self.conv_dropout)

    def forward(self, x, mask):

        x = x * mask
    
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.conv3(x)
        
        x_inp1 = self.inp1(x)
        x_inp2 = self.inp2(x)
        x_inp3 = self.inp3(x)  
        
        x = torch.cat((x_inp1, x_inp2, x_inp3), dim=1)
        
        x = x.permute(0, 2, 1, 3)
        x = torch.reshape(x, (-1, x.shape[1], x.shape[2]))
        
        x, _ = self.gru(x)
        x = x[:, -1, :]
        forecast_y = self.fc1(x).squeeze()
        
        return forecast_y