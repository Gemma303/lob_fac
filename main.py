import os
import pandas as pd
import numpy as np
from datetime import datetime
from utils import train,test,dd_analysis
import argparse
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

parser = argparse.ArgumentParser()
parser.add_argument('--random_seed', type=int, default=92)
parser.add_argument('--horizon', type=int, default=5)
parser.add_argument('--interval', type=int, default=1)
parser.add_argument('--long_ratio', type=int, default=0.25)
parser.add_argument('--y_len', type=int, default=1)
parser.add_argument('--conv_dropout', type=float, default=0.3)
parser.add_argument('--inp_dropout', type=float, default=0.2)
parser.add_argument('--conv_c', type=int, default=4)
parser.add_argument('--inp_c', type=int, default=8)
parser.add_argument('--gru_k', type=int, default=16)
parser.add_argument('--gru_layers', type=int, default=1)
parser.add_argument('--feature_dim', type=int, default=40)
parser.add_argument('--lob_length', type=int, default=16)

parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_min', type=float, default=1e-5)
parser.add_argument('--w_decay', type=float, default=1e-3)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--epoch_long', type=int, default=5)
parser.add_argument('--lr_step_epoch', type=int, default=5)
parser.add_argument('--early_stop_epoch', type=int, default=10)
parser.add_argument('--top_drawdown_periods', type=int, default=3)

parser.add_argument('--lob_data_train', type=str, default='lob_data_train.parquet')
parser.add_argument('--lob_data_test', type=str, default='lob_data_test.parquet')
parser.add_argument('--return_data_train', type=str, default='dailyreturn_train.parquet')
parser.add_argument('--return_data_test', type=str, default='dailyreturn_test.parquet')

parser.add_argument('--model_state', type=str, default='deeplob.pth')
parser.add_argument('--test_results', type=str, default='test_forecast.parquet')
parser.add_argument('--test_portfolio', type=str, default='test_portfolio.pickle')

parser.add_argument('--train', type=bool, default=True)
parser.add_argument('--evaluate', type=bool, default=True)
parser.add_argument('--use_existing_prediction', type=bool, default=False)


args = parser.parse_args()
if __name__ == '__main__':
    
    model_state_path=os.path.join('results','trained_model',args.model_state)
    test_result_path=os.path.join('results','test_results',args.test_results)
    test_portfolio_path=os.path.join('results','test_results',args.test_portfolio)

    train_valid_xdata_path=os.path.join('data',args.lob_data_train)
    train_valid_return_path=os.path.join('data',args.return_data_train)

    test_xdata_path=os.path.join('data',args.lob_data_test)
    test_return_path=os.path.join('data',args.return_data_test)
    
    if args.train:
        print("Training begins.")
        start_time = datetime.now().timestamp()
        train_loss_epochs,valid_loss_epochs = train(train_valid_xdata_path,train_valid_return_path,model_state_path,args)
       

        fig = make_subplots(rows=1, cols=2)
        fig.add_trace(go.Scatter(x=list(range(1,len(train_loss_epochs)+1)), y=train_loss_epochs,mode='lines'), row=1, col=1)
        fig.add_trace(go.Scatter(x=list(range(1,len(valid_loss_epochs)+1)), y=valid_loss_epochs,mode='lines'), row=1, col=2)
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_yaxes(title_text="Training Loss", row=1, col=1)
        fig.update_yaxes(title_text="Validation Loss", row=1, col=2)
        fig.update_layout(width=1000, height=500,showlegend=False)
        with open("results/training_loss/training_loss.html", "w") as f:
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
        
        end_time = datetime.now().timestamp()
        print(f'Training took {np.ceil((end_time - start_time) / 60)} minutes.')
        
    if args.evaluate:
        print("Evaluation begins.")
        start_time = datetime.now().timestamp()
        RankIC,RankICIR,group_weekly_return=test(test_xdata_path,test_return_path,model_state_path,test_result_path,test_portfolio_path,args)
        LongPortfolio,LSPortfolio=group_weekly_return['Group10'].iloc[::args.horizon],(group_weekly_return['Group10']-group_weekly_return['Group1']).iloc[::args.horizon]

        fig_decile = make_subplots(rows=1, cols=2, subplot_titles=("Decile Portfolio Return", "Decile Portfolio Volatility"))
        fig_decile.add_trace(go.Bar(x=list(range(1,11)), y=group_weekly_return.mean()*252/5), row=1, col=1)
        fig_decile.add_trace(go.Bar(x=list(range(1,11)), y=group_weekly_return.std()*np.sqrt(252/5)), row=1, col=2)
        fig_decile.update_xaxes(title_text="Signal Decile", row=1, col=1)
        fig_decile.update_xaxes(title_text="Signal Decile", row=1, col=2)
        fig_decile.update_yaxes(title_text="Annualized Average Return", row=1, col=1)
        fig_decile.update_yaxes(title_text="Annualized Volatility", row=1, col=2)
        fig_decile.update_layout(width=1000, height=500,showlegend=False)

        LSPortfolio_DD=LSPortfolio.reset_index().rename(columns={0:'Return'})
        LSPortfolio_DD,dd_results=dd_analysis(LSPortfolio_DD,k=args.top_drawdown_periods)
        with open("results/test_results/test_results.html", "w") as f:
            f.write(f"RankIC: {np.round(RankIC,4)}, RankICIR: {np.round(RankICIR,4)}<br>")
            f.write("Long Only Portfolio:<br>")
            f.write(f"Annualized Average Return: {np.round(LongPortfolio.mean()*252/5,4)}, Annualized Sharpe Ratio: {np.round(LongPortfolio.mean()/LongPortfolio.std()*np.sqrt(252/5),4)}<br>")
            f.write("Long Short Portfolio:<br>")
            f.write(f"Annualized Average Return: {np.round(LSPortfolio.mean()*252/5,4)}, Annualized Sharpe Ratio: {np.round(LSPortfolio.mean()/LSPortfolio.std()*np.sqrt(252/5),4)}<br>")
            f.write(fig_decile.to_html(full_html=False, include_plotlyjs='cdn'))

            f.write("Long Short Portfolio DrawDown Analysis:<br>")
            if len(dd_results)==0:
                f.write(f"No drawdown periods.<br>")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=LSPortfolio_DD['DataDate'], y=LSPortfolio_DD['NAV'], mode='lines'))
                fig.update_yaxes(range=[LSPortfolio_DD['NAV'].min(), LSPortfolio_DD['NAV'].max()],title_text="Long Short Portfolio NAV")
                fig.update_xaxes(title_text="Date")
                fig.update_layout(width=1000, height=500)
                f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
            else:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=LSPortfolio_DD['DataDate'], y=LSPortfolio_DD['NAV'], mode='lines'))
                for i in range(len(dd_results)):
                    begin=dd_results['StartDate'].iloc[i]
                    end=dd_results['EndDate'].iloc[i]
                    fig.add_shape(type="rect",x0=begin,x1=end,y0=0, y1=1, yref="paper", fillcolor="pink",opacity=0.5,line_width=0)
                    fig.add_shape(type="line",x0=end,x1=end,y0=0,y1=1,yref="paper",line=dict(color="pink", width=2, dash="solid"),opacity=0.8)
                    fig.add_shape(type="line",x0=begin,x1=begin,y0=0,y1=1,yref="paper",line=dict(color="pink", width=2, dash="solid"),opacity=0.8)
                fig.update_yaxes(range=[LSPortfolio_DD['NAV'].min(), LSPortfolio_DD['NAV'].max()],title_text="Long-Short Portfolio NAV")
                fig.update_xaxes(title_text="Date")
                fig.update_layout(width=1000, height=500)
                
                f.write(f"Top {len(dd_results)} drawdown periods.<br>")
                f.write(dd_results.to_html())
                f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
        end_time = datetime.now().timestamp()
        print(f'Evaluation took {np.ceil((end_time - start_time) / 60)} minutes.')