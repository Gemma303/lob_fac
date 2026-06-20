import numpy as np
import pandas as pd
import time
import os

import torch
import torch.nn as nn
import torch.utils.data as torch_data
from DataLoader import NormalYDataset,TestDataset
from model import deeplob

class CorrLoss(nn.Module):
    def __init__(self):
        super(CorrLoss, self).__init__()

    def forward(self, output, target):
        output_dm=output-torch.mean(output)
        target_dm=target-torch.mean(target)
        return -torch.sum(output_dm*target_dm)/torch.sqrt(torch.sum(output_dm**2)*torch.sum(target_dm**2))

def train(train_valid_xdata_path,train_valid_return_path,model_state_path,args):
    
    random_seed=args.random_seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    train_valid_xdata=pd.read_parquet(train_valid_xdata_path)
    train_valid_return=pd.read_parquet(train_valid_return_path)

    train_valid_return_std=train_valid_return.subtract(train_valid_return.mean(1),0).divide(train_valid_return.std(1),0)
    train_valid_set=NormalYDataset(train_valid_xdata,train_valid_return_std,feature_dim=args.feature_dim, length=args.lob_length,horizon=args.horizon, interval=args.interval)
    train_set,valid_set=torch_data.random_split(train_valid_set,[int(len(train_valid_set)*0.7),len(train_valid_set)-int(len(train_valid_set)*0.7)])

    train_indices = train_set.indices
    long_data = NormalYDataset(train_valid_xdata,train_valid_return_std,feature_dim=args.feature_dim, length=args.lob_length,horizon=args.horizon, interval=args.interval,quantile=1-args.long_ratio)
    train_set_long = torch_data.Subset(long_data, train_indices)

    train_loader = torch_data.DataLoader(train_set,shuffle=True)
    valid_loader = torch_data.DataLoader(valid_set)
    train_loader_long = torch_data.DataLoader(train_set_long,shuffle=True)

    model = deeplob(y_len = args.y_len,conv_dropout=args.conv_dropout,inp_dropout=args.inp_dropout,conv_c=args.conv_c,inp_c=args.inp_c,gru_k=args.gru_k,gru_layers=args.gru_layers).to(args.device)
    forecast_loss = CorrLoss()
    my_optim = torch.optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.w_decay)
    my_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=my_optim, T_max=args.epoch//args.lr_step_epoch,eta_min=args.lr_min) 

    model.train()
    for epoch in range(args.epoch_long):
        epoch_start_time = time.time()
        total_loss = 0
    
        for inputs, target, mask in train_loader_long:    
            inputs=inputs.squeeze(0).to(args.device) 
            target=target.squeeze(0).to(args.device)
            mask=mask.squeeze(0).to(args.device)
        
            my_optim.zero_grad()
            forecast=model(inputs,mask)
            loss = forecast_loss(forecast, target)
            loss.backward()       
            my_optim.step()
            total_loss += float(loss)
                
        average_loss_train=total_loss/len(train_set_long)
        print(f'LongEndTraining Epoch {epoch}, time{time.time()-epoch_start_time}, trainset loss{average_loss_train}')

    train_loss_epochs=[]
    valid_loss_epochs=[]

    last_valid_loss = np.inf
    best_valid_loss = np.inf
    valid_loss_non_decrease_count = 0  

    for epoch in range(args.epoch):
        model.train()
        epoch_start_time = time.time()
        total_loss_train = 0
        total_loss_valid = 0
        for inputs, target, mask in train_loader:    
            inputs=inputs.squeeze(0).to(args.device) 
            target=target.squeeze(0).to(args.device)
            mask=mask.squeeze(0).to(args.device)
        
            my_optim.zero_grad()
            forecast=model(inputs,mask)
            loss = forecast_loss(forecast, target)
            loss.backward()       
            my_optim.step()
            total_loss_train += float(loss)
                
        average_loss_train=total_loss_train/len(train_set)
        train_loss_epochs.append(average_loss_train)
        
        model.eval()
        with torch.no_grad():
            for inputs, target, mask in valid_loader:    
                inputs=inputs.squeeze(0).to(args.device)
                target=target.squeeze(0).to(args.device)
                mask=mask.squeeze(0).to(args.device)
                
                forecast=model(inputs,mask)
                loss = forecast_loss(forecast, target)
                total_loss_valid += float(loss)
        average_loss_valid=total_loss_valid/len(valid_set)
        valid_loss_epochs.append(average_loss_valid)
        print(f'AllTraining Epoch {epoch}, time{time.time()-epoch_start_time}, trainset loss{average_loss_train}, validset loss{average_loss_valid}')
            
        if (args.lr_step_epoch>0) and ((epoch+1) % args.lr_step_epoch == 0):
            my_lr_scheduler.step()

        if average_loss_valid < last_valid_loss:
            valid_loss_non_decrease_count = 0
        else:
            valid_loss_non_decrease_count += 1 
        last_valid_loss=average_loss_valid

        if average_loss_valid<best_valid_loss:
            torch.save(model.state_dict(), model_state_path)
            best_valid_loss=average_loss_valid
        
        if valid_loss_non_decrease_count >= args.early_stop_epoch:
            return train_loss_epochs,valid_loss_epochs
        
    return train_loss_epochs,valid_loss_epochs


def test(test_xdata_path,return_data_path,model_state_path,test_result_path,test_portfolio_path,args):
    return_data=pd.read_parquet(return_data_path)
    weekly_return=(np.exp(np.log(1+return_data).rolling(args.horizon,min_periods=1).sum())-1).stack().dropna().sort_index().to_frame().rename(columns={0:'WeeklyReturn'})
    if args.use_existing_prediction and os.path.isfile(test_result_path):
        signal_df=pd.read_parquet(test_result_path)
    else:
        test_xdata=pd.read_parquet(test_xdata_path)
        test_set = TestDataset(test_xdata)

        model = deeplob(y_len = args.y_len,conv_dropout=args.conv_dropout,inp_dropout=args.inp_dropout,conv_c=args.conv_c,inp_c=args.inp_c,gru_k=args.gru_k,gru_layers=args.gru_layers).to(args.device)
        model.load_state_dict(torch.load(model_state_path,weights_only=True))
        model.eval()

        test_list=[]
        with torch.no_grad():
            for i in range(len(test_set)):   
                curr_test_data=test_set[i]
                inputs,mask,secucodes,date=curr_test_data[0].squeeze(0).to(device),curr_test_data[1].squeeze(0).to(device),curr_test_data[2].tolist(),curr_test_data[3]

                forecast = model(inputs,mask)
                forecast = pd.DataFrame(forecast.cpu().numpy(),columns=['Signal']).assign(SecuCode=secucodes).assign(DataDate=date)
                test_list.append(forecast)
        signal_df=pd.concat(test_list).assign(DataDate=lambda x:pd.to_datetime(x.DataDate)).set_index(['DataDate','SecuCode']).sort_index()
        signal_df.to_parquet(test_result_path)
    
    signal_df=signal_df.unstack().reindex(return_data.index).shift(1+args.horizon).stack()
    compare=signal_df.merge(weekly_return,left_index=True,right_index=True)
    RankCorr=compare.groupby('DataDate').apply(lambda df:df.corr(method='spearman').iloc[0,1])
    RankIC=RankCorr.mean()
    RankICIR=RankIC/RankCorr.std()
    compare['Group']=compare.groupby('DataDate')['Signal'].apply(lambda x:pd.qcut(x,10,[f'Group{i}' for i in range(1,11)]))
    group_weekly_return=compare.groupby(['DataDate','Group'])['WeeklyReturn'].mean().unstack()
    group_weekly_return.to_pickle(test_portfolio_path)
    
    return RankIC,RankICIR,group_weekly_return

def dd_analysis(portfolio,k=3):
    portfolio['NAV']=(1+portfolio['Return']).cumprod()
    portfolio['DD']=(portfolio['NAV']-portfolio['NAV'].cummax())/portfolio['NAV'].cummax()
    portfolio['DD_Flag']=0
    
    begins=[]
    ends=[]
    depths=[]
    for _ in range(k):
        if (portfolio['DD']<0).sum()==0:
            break
        midx=portfolio.loc[~portfolio['DD_Flag'].astype(bool)]['DD'].idxmin()
        begin=portfolio.loc[:midx].query("DD>=0").index[-1]
        end_df=portfolio.loc[midx:].query("DD>=0")
        if len(end_df)>0:
            end=end_df.index[0]
        else:
            end=len(portfolio)-1
        portfolio.loc[begin:end,'DD_Flag']=1
        begins.append(portfolio.loc[begin]['DataDate'])
        ends.append(portfolio.loc[end]['DataDate'])
        depths.append(portfolio.loc[midx]['DD'])
    output=pd.DataFrame([begins,ends,depths],index=['StartDate','EndDate','DrawDown']).T
    return portfolio,output