import numpy as np
import pandas as pd
import lightgbm as lgb
import csv
import time
from datetime import datetime, timedelta
DIR = '../data/'

start = time.time()

print("GETTING TRAINING FEATURES...")
train_x = pd.read_csv(DIR + 'train_x_features.csv')
print("\tDone.")

print("GETTING TRAINING LABELS...")
train_labels = pd.read_csv(DIR + 'label_train_features.csv',
                           names = ['num', 'reorder'])
train_labels = train_labels[train_labels.columns[1]]
print("\tDone.")

print("GETTING TESTING FEATURES...")
test_x = pd.read_csv(DIR + 'test_x_features.csv')
print("\tDone.")

features = ['order_dow', 'order_hour_of_day', 'days_since_prior_order',
        'reorder_rate', 'order_total','avg_add_to_cart_order', 'day_count', 'hour_count',
        'reorder_total', 'orders_sum', 'days_since_prior_std','avg_basket', 'avg_reorder', 'num_unique_items',
        'aisle_id', 'department_id', 'up_orders', 'up_reorders', 'up_reorder_rate', 'up_add_to_cart_order', 
        'up_days_since_prior_order', 'order_ratio', 'delta_dow', 'delta_order_hour_of_day', 'ordered_last_time', 
        'reorder_total_ratio', 'reorder_total_ratio', 'numbers_since_last_order', 'first_ordered_number',
        'comp_size', 'avg_diff', 'std_diff']
#'comp_size', 'avg_diff', 'std_diff'

# parameter for lgbt
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'num_leaves': 96,
    'max_depth': 10,
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
result = test_x[['order_id', 'product_id', 'confidence']]
del test_x

"""
Selecting product with confidence level above threshold.
Then combine products within the same order together
Write output to out.csv
"""
"""Threshold settings"""
threshold = 0.18
result = result[result['confidence'] >= threshold]
result = result.groupby('order_id')['product_id'].apply(list).reset_index()
result.columns = ['order_id', 'products']
result['products'] = result['products'].apply(lambda x: " ".join(str(num) for num in x))
submission = pd.read_csv(DIR + 'sample_submission.csv')
submission = submission.drop('products', axis=1)
submission = pd.merge(submission, result, on='order_id', how='left').fillna("None")
submission = submission.sort_values('order_id')
print("Writing output")
submission.to_csv('out.csv', index=False, mode='w+', quoting=csv.QUOTE_NONE)

end = time.time()
print(str((end - start) / 60), "minutes")

""" TODO: Implement Cross-validation"""
def cross_validate(feature_vector,labels):

    return None
