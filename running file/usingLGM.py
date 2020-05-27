# coding: utf-8
import os
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import classification
from sklearn.metrics import regression
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn import model_selection


#%%
path = './data'
#%% dataset CLICK

# csv_path = os.path.join(path, 'click.csv')
# if not os.path.exists(csv_path):
#     os.makedirs(path, exist_ok=True)
#     download('https://www.dropbox.com/s/w43ylgrl331svqc/click.csv?dl=1', csv_path)

# data = pd.read_csv(csv_path, index_col=0)
#
# X, y = data.drop(columns=['target']), data['target']
#
# X[['url_hash','ad_id','advertiser_id','query_id','keyword_id','title_id','description_id','user_id']]=X[['url_hash','ad_id','advertiser_id','query_id','keyword_id','title_id','description_id','user_id']].apply(LabelEncoder().fit_transform)
#
# X_train, X_test = X[:-100_000].copy(), X[-100_000:].copy()
# y_train, y_test = y[:-100_000].copy(), y[-100_000:].copy()
#
# X_train_set, X_valid_set, y_train_set, y_valid_set = train_test_split(X_train, y_train, test_size=valid_size, random_state=validation_seed)
#
# y_train = (y_train.values.reshape(-1) == 1).astype('int64')
# y_test = (y_test.values.reshape(-1) == 1).astype('int64')

#%% dataset YearPrediction
data_path = os.path.join(path, 'data.csv')
if not os.path.exists(data_path):
    os.makedirs(path, exist_ok=True)
    download('https://www.dropbox.com/s/l09pug0ywaqsy0e/YearPredictionMSD.txt?dl=1', data_path)

test_size = 51630
n_features = 91
types = {i: (np.float32 if i != 0 else np.int) for i in range(n_features)}



data = pd.read_csv(data_path, header=None, dtype=types)
data_train, data_test = data.iloc[:-test_size], data.iloc[-test_size:]




X_train, y_train = data_train.iloc[:, 1:].values, data_train.iloc[:, 0].values

X_train_set, X_valid_set, y_train_set, y_valid_set =model_selection.train_test_split(X_train, y_train, test_size = 0.3)


X_test, y_test = data_test.iloc[:, 1:].values, data_test.iloc[:, 0].values




#%%

# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train_set, y_train_set)
lgb_eval = lgb.Dataset(X_valid_set, y_valid_set, reference=lgb_train)

# specify your configurations as a dict
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2'},
    'num_leaves': 31,
    'learning_rate': 0.01,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
    'num_thread': 8,
    'random_seed': 1337
}

print('Start training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=200000,
                valid_sets=lgb_eval,
                early_stopping_rounds=10000)

print('Save model...')
# save model to file
gbm.save_model('model.txt')

print('Start predicting...')
# predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
# #%%
# # eval
# # print('The rmse of prediction is:', classification.accuracy_score(y_test, y_pred))
# threshold = 0.5
# for i in range(len(y_pred)):
#     y_pred[i] = int(1) if y_pred[i] > threshold else 0
#%%
print(regression.mean_squared_error(y_test, y_pred))