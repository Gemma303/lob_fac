# lob_fac
This repository presents a weekly stock return forecasting factor constructed by training a CNN-GRU deep learning model on LOB data. It shows that high-frequency, intraday LOB data contains predictive information for longer-term future stock returns.

The results presented in _test_results.html_ are obtained by averaging predictions from 5 independent training runs. For the testing period from January 2023 to April 2026, the factor's average RankIC and RankICIR were 0.0819 and 0.5237, respectively.

The deep convolutional neural networks in the model are based on the architecture from the paper, [DeepLOB: Deep Convolutional Neural Networks for Limit Order Books](https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books/tree/master). Regularization measures such as Dropout layers are added to stabilize the training process.
