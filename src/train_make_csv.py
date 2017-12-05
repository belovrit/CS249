import os

import tensorflow as tf
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb
import csv
DIR = '../data/'

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
        'order_id': np.int32,
        'aisle_id': np.int16,
        'department_id': np.int16},
        usecols=['product_id', 'aisle_id', 'department_id'])
print('Joining data')
# prior set
prior_orders = pd.merge(priors, orders, on='order_id')
del priors
# retrieve traning set
train = train[train['reordered'] == 1].drop('add_to_cart_order', axis=1)
train_orders = orders[orders.eval_set == 'train']
# retrieve test set
test_orders = orders[orders.eval_set == 'test']
# generate candidate set based on products to user
user_products = pd.DataFrame()
print('Getting list of product ordered from each user')
user_products = prior_orders.groupby('user_id')['product_id'].apply(set)
user_products = user_products.reset_index()
user_products.columns = ['user_id', 'products']

# Linzuo: product reorder rate and product total ordered times
product_info = pd.DataFrame()
product_info['order_total'] = prior_orders.groupby('product_id')['reordered'].count()
product_info['reorder_total'] = prior_orders.groupby('product_id')['reordered'].sum()
product_info['reorder_rate'] = product_info['reorder_total'] / product_info['order_total']
product_info['avg_add_to_cart_order'] = prior_orders.groupby('product_id')['add_to_cart_order'].mean().astype(np.int8)
product_info = product_info.reset_index()
products = products.merge(product_info, on='product_id', how='left')
# Steven: Add features including relationship between products and days/hours
#product_day = train_orders[['product_id', 'order_dow']]
product_day_freq = pd.DataFrame(prior_orders[['product_id', 'order_dow']].groupby(['product_id', 'order_dow'])['order_dow'].count())
product_day_freq = product_day_freq.rename(columns = {'order_dow': 'day_count'}).reset_index()
#product_hour = train_orders[['product_id', 'order_hour_of_day']]
product_hour_freq = pd.DataFrame(prior_orders[['product_id', 'order_hour_of_day']].groupby(['product_id', 'order_hour_of_day'])['order_hour_of_day'].count())
product_hour_freq = product_hour_freq.rename(columns = {'order_hour_of_day': 'hour_count'}).reset_index()

print('Generating user X product features')
#user X product feature
#calculating U_P featuers
u_p = pd.DataFrame()
u_p['up_orders'] = prior_orders.groupby(['user_id', 'product_id'])['reordered'].count()
u_p['up_reorders'] = prior_orders.groupby(['user_id', 'product_id'])['reordered'].sum()
u_p['up_reorder_rate'] = u_p['up_reorders'] / u_p['up_orders']
u_p['up_add_to_cart_order'] = prior_orders.groupby(['user_id', 'product_id'])['add_to_cart_order'].mean()
u_p['up_days_since_prior_order'] = prior_orders.groupby(['user_id', 'product_id'])['days_since_prior_order'].mean()
u_p['up_avg_dow'] = prior_orders.groupby(['user_id', 'product_id'])['order_dow'].mean()
u_p['up_avg_hour'] = prior_orders.groupby(['user_id', 'product_id'])['order_hour_of_day'].mean()
u_p['last_ordered_number'] = prior_orders.groupby(['user_id', 'product_id'])['order_number'].max()
u_p['first_ordered_number'] = prior_orders.groupby(['user_id', 'product_id'])['order_number'].min()

u_p = u_p.reset_index()

user_features = pd.DataFrame()
user_features['total_reorders'] = prior_orders.groupby(['user_id'])['reordered'].count()
user_features = user_features.reset_index()
u_p = u_p.merge(user_features, on=['user_id'], how='left')
del user_features
#Bobby:
u_features = ['user_id','orders_sum', 'days_since_prior_std',
              'avg_basket', 'avg_reorder', 'num_unique_items',
              'comp_size', 'avg_diff', 'std_diff']
user_features = pd.read_csv(DIR + 'user_info.csv', dtype={
       'user_id': np.uint32,
       'orders_sum': np.uint16,
       'days_since_prior_std': np.float32,
       'avg_basket': np.float32,
       'avg_reorder': np.float32,
       'num_unique_items': np.uint16,
       'comp_size': np.uint32,
       'avg_diff': np.float32,
       'std_diff': np.float32},
       usecols=u_features)

print('Getting sets by orders')
products_by_orders = prior_orders.groupby(['user_id','order_number'])['product_id'].apply(set)
products_by_orders = products_by_orders.reset_index()
products_by_orders.columns = ['user_id', 'order_number', 'last_ordered_products']

lastOrder = pd.DataFrame()
lastOrder['order_number'] = prior_orders.groupby(['user_id'])['order_number'].max()
lastOrder = lastOrder.reset_index()
lastOrder = lastOrder.merge(products_by_orders, on=['user_id', 'order_number'], how='left')
lastOrder.drop('order_number', axis=1)
del products_by_orders
del prior_orders

def is_in_order(row):
    return row['product_id'] in  row['last_ordered_products']
        
def get_features(features, isTrain=True):
    # declared used features
    labels = []
    feature_vector = pd.DataFrame()
    feature_vector['user_id'] = features['user_id']
    feature_vector['order_id'] = features['order_id']
    feature_vector['order_dow'] = features['order_dow']
    feature_vector['order_hour_of_day'] = features['order_hour_of_day']
    feature_vector['days_since_prior_order'] = features['days_since_prior_order']

    print('Generating candidate Set for data')
    feature_vector = feature_vector.merge(user_products, on='user_id', how='left')
    # convert set to list
    feature_vector['products'] = feature_vector['products'].apply(lambda x: list(x))
    # expand product list, replicate each row
    print('Replicating each row based on list items')
    """ !!! This lamda function was copied from https://stackoverflow.com/questions/27263805/pandas-when-cell-contents-are-lists-create-a-row-for-each-element-in-the-list"""
    s = feature_vector.apply(lambda x: pd.Series(x['products']),axis=1).stack().reset_index(level=1, drop=True)
    s.name = 'product_id'
    feature_vector = feature_vector.drop('products', axis=1).join(s.astype(np.uint16))

    if isTrain:
        # merge ordered product based on train set
        feature_vector = feature_vector.merge(train, on=['order_id', 'product_id'], how='left')
        feature_vector['reordered'] = feature_vector['reordered'].fillna(0)
        feature_vector['reordered'] = feature_vector['reordered'].astype(np.uint8)
        labels = feature_vector['reordered']
    feature_vector = feature_vector.merge(products, on='product_id', how='left')
    feature_vector = feature_vector.merge(product_day_freq, on=['product_id', 'order_dow'], how='left')
    feature_vector = feature_vector.merge(product_hour_freq, on=['product_id', 'order_hour_of_day'], how='left')
    feature_vector = feature_vector.merge(user_features, on='user_id', how='left')
    feature_vector = feature_vector.merge(u_p, on=['user_id', 'product_id'], how='left')
    feature_vector = feature_vector.merge(lastOrder, on='user_id', how='left')     
    feature_vector['order_ratio'] = feature_vector['up_orders'] / feature_vector['orders_sum']
    feature_vector['delta_days_since_prior_order'] = abs(feature_vector['up_days_since_prior_order'] - feature_vector['days_since_prior_order'])
    feature_vector['delta_order_hour_of_day'] = abs(feature_vector['up_avg_hour'] - feature_vector['order_dow'])
    feature_vector['reorder_total_ratio'] = feature_vector['up_reorders'] / feature_vector['total_reorders']
    feature_vector['delta_dow'] = abs(feature_vector['up_days_since_prior_order'] - feature_vector['order_hour_of_day'])
    feature_vector['ordered_last_time'] = feature_vector.apply(is_in_order,axis=1)
    feature_vector['ordered_last_time'] = feature_vector['ordered_last_time'].astype(np.uint8)
    feature_vector['numbers_since_last_order'] = feature_vector['order_number'] - feature_vector['last_ordered_number']
    return feature_vector, labels

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
print('Generating training feature vectors')
train_x, label_train = get_features(train_orders, isTrain=True)
test_x, label_test = get_features(test_orders, isTrain=False)
del train_orders
del test_orders

print("train_x Features to csv")
train_x.to_csv(DIR + "train_x_features.csv", encoding = 'utf-8')
print("train_x Features WRITTEN")

print("label_train Features to csv")
label_train.to_csv(DIR + "label_train_features.csv", encoding = 'utf-8')
print("label_train Features WRITTEN")

print("test_x Features to csv")
test_x.to_csv(DIR + "test_x_features.csv", encoding = 'utf-8')
print("test_x Features WRITTEN")

