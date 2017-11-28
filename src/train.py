import numpy as np
import pandas as pd
import lightgbm as lgb

DIR = './data/'

priors = pd.read_csv(DIR + 'order_products__prior.csv', dtype={
            'order_id': np.int32,
            'product_id': np.uint16,
            'add_to_cart_order': np.int16,
            'reordered': np.int8})

train = pd.read_csv(DIR + 'order_products__train.csv', dtype={
            'order_id': np.int32,
            'product_id': np.uint16,
            'add_to_cart_order': np.int16,
            'reordered': np.int8})

orders = pd.read_csv(DIR + 'orders.csv', dtype={
        'order_id': np.int32,
        'user_id': np.int32,
        'eval_set': 'category',
        'order_number': np.int16,
        'order_dow': np.int8,
        'order_hour_of_day': np.int8,
        'days_since_prior_order': np.float32})

products = pd.read_csv(DIR + 'products.csv', dtype={
        'product_id': np.uint16,
        'order_id': np.int32,
        'aisle_id': np.uint8,
        'department_id': np.uint8},
        usecols=['product_id', 'aisle_id', 'department_id'])
# retrieve traning set
train_orders = pd.merge(train, orders, on='order_id')
del train
# prior set
prior_orders = pd.merge(priors, orders, on='order_id')
del priors

# retrieve test set
test_orders = orders[orders.eval_set == 'test'].sort_values('order_id')
# generate candidate set based on products to user
user_products = pd.DataFrame()
user_products = prior_orders.groupby('user_id')['product_id'].apply(set)
user_products = user_products.reset_index()
user_products.columns = ['user_id', 'products']
# Now generate the candidate set for test_orders
test_orders['products'] = test_orders['user_id'].map(user_products['products'])
# convert set to list
test_orders['products'] = test_orders['products'].apply(lambda x: list(x))
# expand product list, replicate each row
""" !!! This lamda function was copied from https://stackoverflow.com/questions/27263805/pandas-when-cell-contents-are-lists-create-a-row-for-each-element-in-the-list"""
s = test_orders.apply(lambda x: pd.Series(x['products']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'product_id'
test_orders = test_orders.drop('products', axis=1).join(s.astype(np.uint16))

def get_features(features, train=True):
    label = []
    feature_vector = pd.DataFrame()
    if train:
        # merge ordered product based on train set
        label = features['reordered']

    feature_vector['order_dow'] = features['order_dow']
    feature_vector['order_hour_of_day'] = feature_vector['order_hour_of_day']
    feature_vector['days_since_prior_order'] = features['days_since_prior_order']

    return feature_vector, label





""" TODO: Implement Cross-validation"""

def cross_validate(feature_vector,labels):
    return None
