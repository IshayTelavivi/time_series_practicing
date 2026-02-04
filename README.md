# Time Series Practicing

Practicing time series forecasting of sales, based on a Kaggle competition.

My practice was on an completed Kaggle competition: https://www.kaggle.com/competitions/sales-time-series-forecasting-tx-afcs2021/overview. 

The data is a daily data of ~5 years, where product hierarchy: item, department, category. The sales resolution is by item.
Also we have data for 3 states, where each state has multiple stores.

In addition to the sales data, we had also calendar dataset with events of different types (National, Cultural, Sporting, Religious) and sell process dataset.

The challenge was to build a model that predicts the next 28 days.

## Approach

I chose to use classic ML where some of my features are previous sequential data, and some are engineered features, based on sequential data aggregation, temporal features, events and price fluctuations. The processing and feature engineering is in `preprocessing.ipynb`.

For training I used two methods:
- Forecasting each one of the 28 days independently (in `lgbm_training_multi_output.ipynb`)
- Chain regressor - forecasting each day based on the prediction of the previous days (in `lgbm_training_chain_regressor.ipynb`)
