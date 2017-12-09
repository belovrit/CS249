import numpy as np
import pandas as pd
import lightgbm as lgb
import csv
import gc
import time

DIR = '../data/'

start = time.time()

priors = pd.read_csv(DIR + 'order_products__prior.csv', dtype={
            'order_id': np.int32,
            'product_id': np.uint16,
            'add_to_cart_order': np.int16,
            'reordered': np.int8})

print("loaded priors")

train = pd.read_csv(DIR + 'order_products__train.csv', dtype={
            'order_id': np.int32,
            'product_id': np.uint16,
            'add_to_cart_order': np.int16,
            'reordered': np.int8})

print("loaded train")

orders = pd.read_csv(DIR + 'orders.csv', dtype={
        'order_id': np.int32,
        'user_id': np.int32,
        'eval_set': 'category',
        'order_number': np.int16,
        'order_dow': np.int8,
        'order_hour_of_day': np.int8,
        'days_since_prior_order': np.float32})

print("loaded orders")

products = pd.read_csv(DIR + 'products.csv', dtype={
        'product_id': np.uint16,
        'product_name': np.str,
        'aisle_id': np.int16,
        'department_id': np.int16},
        usecols=['product_id', 'product_name', 'aisle_id', 'department_id'])

print('loaded products')

print('Joining data')
# prior set
products['word_length_product_name'] = products['product_name'].apply(lambda x: len(x.split()))
products['organic_name'] = products['product_name'].apply(lambda x: x.lower().find('organic'))
products['product_organic'] = products['organic_name'].apply(lambda x: 1 if x >= 0 else 0)
prior_orders = pd.merge(priors, orders, on='order_id')
prior_orders['up_id'] = (prior_orders['user_id'] * 100000 + prior_orders['product_id']  ).astype(np.uint64)
del priors
# retrieve traning set
train = train[train['reordered'] == 1].drop('add_to_cart_order', axis=1)
# generate candidate set based on products to user
print('Getting list of product ordered from each user')
user_products = pd.DataFrame()
user_products = prior_orders.groupby('user_id')['product_id'].apply(set)
user_products = user_products.reset_index()
user_products.columns = ['user_id', 'products']

print('Getting Linzuo')
product_info = pd.DataFrame()
product_info['order_total'] = prior_orders.groupby('product_id')['reordered'].count()
product_info['reorder_total'] = prior_orders.groupby('product_id')['reordered'].sum()
product_info['reorder_rate'] = product_info['reorder_total'] / product_info['order_total']
product_info['avg_add_to_cart_order'] = prior_orders.groupby('product_id')['add_to_cart_order'].mean().astype(np.int8)
product_info = product_info.reset_index()
products = products.merge(product_info, on='product_id', how='left')
del product_info
gc.collect()
# Steven: Add features including relationship between products and days/hours
print('Getting Steven')
#Frequency of the day of product being bought
product_day_freq = prior_orders.groupby(['product_id', 'order_dow'])['order_dow'].count()
product_day_freq = product_day_freq.rename(columns = {'order_dow': 'day_count'}).reset_index()
product_day_freq.columns = ['product_id', 'order_dow', 'day_count']    
# Frequency of the hour of product being bought
product_hour_freq = prior_orders.groupby(['product_id', 'order_hour_of_day'])['order_hour_of_day'].count()
product_hour_freq = product_hour_freq.rename(columns = {'order_hour_of_day': 'hour_count'}).reset_index()
product_hour_freq.columns = ['product_id', 'order_hour_of_day', 'hour_count']
gc.collect()

#user X product feature
#calculating U_P featuers
temp = prior_orders.merge(products, on='product_id', how='left')

print('Generating user X product features')
u_p = pd.DataFrame()
u_p['up_orders'] = prior_orders.groupby('up_id')['reordered'].count()
u_p['up_reorders'] = prior_orders.groupby('up_id')['reordered'].sum()
u_p['up_reorder_rate'] = u_p['up_reorders'] / u_p['up_orders']
u_p['up_add_to_cart_order'] = prior_orders.groupby('up_id')['add_to_cart_order'].mean()
u_p['up_days_since_prior_order'] = prior_orders.groupby('up_id')['days_since_prior_order'].mean()
u_p['up_avg_dow'] = prior_orders.groupby('up_id')['order_dow'].mean()
u_p['up_avg_hour'] = prior_orders.groupby('up_id')['order_hour_of_day'].mean()
u_p['last_ordered_number'] = prior_orders.groupby('up_id')['order_number'].max()
u_p['first_ordered_number'] = prior_orders.groupby('up_id')['order_number'].min()
u_p['first_last'] = u_p['last_ordered_number'] - u_p['first_ordered_number']
u_p['bought_times'] = prior_orders.groupby('up_id').cumcount() + 1
#Alex
u_p['like_organic'] = temp.groupby('up_id')['product_organic'].sum()
u_p = u_p.reset_index()

print('Generating department and aisle features')

user_aisle = temp.groupby(['user_id', 'aisle_id'])['product_id'].count().to_frame().reset_index()
user_aisle.columns = ['user_id', 'aisle_id', 'aisle_orders']

user_depart = temp.groupby(['user_id', 'department_id'])['product_id'].count().to_frame().reset_index()
user_depart.columns = ['user_id', 'department_id', 'depart_orders']

#Alex:
print('Getting Alex')
user_features = pd.DataFrame()
user_features['mean_length_product_name'] = temp.groupby('user_id')['word_length_product_name'].mean()
user_features = user_features.reset_index()
temp2 = pd.DataFrame()
temp3 = pd.DataFrame()
user = pd.DataFrame()
temp2['user_order_sum'] = orders.groupby('user_id')['order_id'].size()
temp2 = temp2.reset_index()
temp3['user_avg_interval'] = orders.groupby('user_id')['days_since_prior_order'].mean()
temp3 = temp3.reset_index()
user = pd.merge(temp2, temp3, on='user_id')
del temp2
del temp3
user_features = pd.merge(user_features,user,on='user_id')
del user
del temp
gc.collect()

print('Getting sets by orders')
users_last_order = prior_orders.groupby(['user_id'])['order_number'].max().to_frame().reset_index()
last_orders = users_last_order.merge(prior_orders[['user_id', 'order_number', 'product_id']], on=['user_id', 'order_number'])
last_orders = last_orders.drop('order_number', axis = 1)
last_orders['ordered_last_time'] = 1
last_orders['up_id'] = (last_orders['user_id'] * 100000 + last_orders['product_id']).astype(np.uint64)
last_orders= last_orders.drop(['user_id', 'product_id'], axis = 1)
del prior_orders
del users_last_order
gc.collect()

print('Creating feature vectors')
# declared used features
feature_vector = pd.DataFrame()
feature_vector['order_id'] = orders['order_id']
feature_vector['user_id'] = orders['user_id']
feature_vector['order_dow'] = orders['order_dow']
feature_vector['order_number'] = orders['order_number']
feature_vector['order_hour_of_day'] = orders['order_hour_of_day']
feature_vector['days_since_prior_order'] = orders['days_since_prior_order']
feature_vector['eval_set'] = orders['eval_set']
feature_vector = feature_vector[feature_vector['eval_set'] != 'prior']
del orders
gc.collect()
print('Generating candidate Set for user')
feature_vector = feature_vector.merge(user_products, on='user_id', how='left')
# convert set to list
feature_vector['products'] = feature_vector['products'].apply(lambda x: list(x))
# expand product list, replicate each row
print('Replicating each row based on list items')
""" !!! This lamda function was copied from https://stackoverflow.com/questions/27263805/pandas-when-cell-contents-are-lists-create-a-row-for-each-element-in-the-list"""
s = feature_vector.apply(lambda x: pd.Series(x['products']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'product_id'
feature_vector = feature_vector.drop('products', axis=1).join(s.astype(np.uint16))
feature_vector['up_id'] = (feature_vector['user_id'] * 100000 + feature_vector['product_id']).astype(np.uint64)
del s
del user_products
gc.collect()
# merge ordered product based on train set
feature_vector = feature_vector.merge(train, on=['order_id', 'product_id'], how='left')
feature_vector['reordered'] = feature_vector['reordered'].fillna(0)
feature_vector['reordered'] = feature_vector['reordered'].astype(np.uint8)
print('Merging features')
feature_vector = feature_vector.merge(products, on='product_id', how='left')
del products

feature_vector = feature_vector.merge(product_day_freq, on=['product_id', 'order_dow'], how='left')
feature_vector = feature_vector.merge(product_hour_freq, on=['product_id', 'order_hour_of_day'], how='left')
feature_vector = feature_vector.merge(user_features, on='user_id', how='left')

del product_day_freq
del product_hour_freq
del user_features
gc.collect()
feature_vector = feature_vector.merge(u_p, on='up_id', how='left')
feature_vector = feature_vector.merge(last_orders, on='up_id', how='left')
feature_vector['ordered_last_time'] = feature_vector['ordered_last_time'].fillna(0)
del u_p
del last_orders
gc.collect()
feature_vector = feature_vector.merge(user_aisle, on=['user_id', 'aisle_id'], how='left')
feature_vector = feature_vector.merge(user_depart, on=['user_id', 'department_id'], how='left')


#feature_vector['order_ratio'] = feature_vector['up_orders'] / feature_vector['orders_sum']
feature_vector['delta_days_since_prior_order'] = abs(feature_vector['up_days_since_prior_order'] - feature_vector['days_since_prior_order'])
feature_vector['delta_order_hour_of_day'] = abs(feature_vector['up_avg_hour'] - feature_vector['order_dow'])
#feature_vector['reorder_total_ratio'] = feature_vector['up_reorders'] / feature_vector['total_reorders']
feature_vector['delta_dow'] = abs(feature_vector['up_days_since_prior_order'] - feature_vector['order_hour_of_day'])
feature_vector['numbers_since_last_order'] = feature_vector['order_number'] - feature_vector['last_ordered_number']
feature_vector['ratio_since_first_purchase'] = feature_vector['up_orders'] / (feature_vector['order_number'] -1 - feature_vector['first_ordered_number'])

print('\t Done')

features = ['order_dow', 'order_hour_of_day', 'days_since_prior_order',
        'reorder_rate', 'order_total','avg_add_to_cart_order', 'day_count', 'hour_count',
        'reorder_total', 'aisle_id', 'department_id', 'up_orders', 'up_reorders', 'up_reorder_rate',
        'up_add_to_cart_order', 'up_days_since_prior_order', 'delta_dow', 'delta_order_hour_of_day',
        'ordered_last_time', 'aisle_orders', 'depart_orders', 'numbers_since_last_order', 'first_last',
        'first_ordered_number', 'ratio_since_first_purchase', 'bought_times', 'like_organic',
        'mean_length_product_name', 'user_order_sum', 'user_avg_interval']

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

num_round = 120
print('Generating feature vectors')

train_set = feature_vector[feature_vector['eval_set'] == 'train']
test_set = feature_vector[feature_vector['eval_set'] == 'test']
print('Writing output to /data/processed')
train_set.to_csv('../data/processed/train_features.csv', index=False, mode='w+')
test_set.to_csv('../data/processed/test_features.csv', index=False, mode='w+')
end = time.time()
print(str((end - start) / 60), "minutes")