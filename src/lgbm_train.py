import numpy as np
import pandas as pd
import lightgbm as lgb
import time
from datetime import datetime, timedelta
DIR = '../data/'

start = time.time()

print("GETTING TRAINING FEATURES...")
train_x = pd.read_csv(DIR + 'processed/train_features.csv')
print("\tDone.")

print("GETTING TRAINING LABELS...")
train_labels = train_x['reordered']
print("\tDone.")

print("GETTING TESTING FEATURES...")
test_x = pd.read_csv(DIR + 'processed/test_features.csv')
print("\tDone.")

features = ['order_dow', 'order_hour_of_day', 'days_since_prior_order',
        'reorder_rate', 'order_total','avg_add_to_cart_order', 'day_count', 'hour_count',
        'reorder_total', 'orders_sum', 'days_since_prior_std','avg_basket', 'avg_reorder', 'num_unique_items',
        'aisle_id', 'department_id', 'up_orders', 'up_reorders', 'up_reorder_rate', 'up_add_to_cart_order',
        'up_days_since_prior_order', 'order_ratio', 'delta_dow', 'delta_order_hour_of_day', 'ordered_last_time', 'aisle_orders', 'depart_orders',
        'numbers_since_last_order', 'first_last', 'first_ordered_number', 'ratio_since_first_purchase', 'bought_times',
        'comp_size', 'avg_diff', 'std_diff']
#'comp_size', 'avg_diff', 'std_diff'

# parameter for lgbt#0.38119
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'max_depth': 5,
    'learning_rate': 0.05,
    'num_leaves': 32,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.95,
    'bagging_freq': 5
}
num_round = 100

print('Building dataset...')
# keep features
train_data = lgb.Dataset(train_x[features], label=train_labels, categorical_feature=['aisle_id', 'department_id'])
valid_data = lgb.Dataset(train_x[features], train_labels)

# starting to train
print('Training......')
bst = lgb.train(params, train_data, num_round, valid_sets=valid_data, verbose_eval=5)
del train_x

print('Predicting......')
pred = bst.predict(test_x[features])
print('Prediction Done......')
test_x['confidence'] = pred
test_x[['order_id', 'product_id', 'confidence']].to_csv(DIR + 'predict/lgbm_pred.csv', mode='w+', index=False)
end = time.time()
print(str((end - start) / 60), "minutes")
