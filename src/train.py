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
print('Joining data')
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
print('Getting list of product ordered from each user')
user_products = prior_orders.groupby('user_id')['product_id'].apply(set)
user_products = user_products.reset_index()
user_products.columns = ['user_id', 'products']
# Now generate the candidate set for test_orders
print('Generating candidate Set for test data')
test_orders = pd.merge(test_orders, user_products, on='user_id', how='left')
# convert set to list
test_orders['products'] = test_orders['products'].apply(lambda x: list(x))
# expand product list, replicate each row
print('Replicating each row based on list items')
""" !!! This lamda function was copied from https://stackoverflow.com/questions/27263805/pandas-when-cell-contents-are-lists-create-a-row-for-each-element-in-the-list"""
s = test_orders.apply(lambda x: pd.Series(x['products']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'product_id'
test_orders = test_orders.drop('products', axis=1).join(s.astype(np.uint16))


#user_num_orders = orders.groupby('user_id')['order_id'].count().astype(np.uint32).to_frame().reset_index()
#user_num_orders.columns = ['user_id', 'number_of_orders']
# Linzuo: product reorder rate and product total ordered times
product_info = pd.DataFrame()
product_info['order_total'] = prior_orders.groupby('product_id')['reordered'].count()
product_info['reorder_total'] = prior_orders.groupby('product_id')['reordered'].sum()
product_info['reorder_rate'] = product_info['reorder_total'] / product_info['order_total']
product_info['avg_add_to_cart_order'] = prior_orders.groupby('product_id')['add_to_cart_order'].mean().astype(np.int8)
product_info = product_info.reset_index()

# Steven: Add features including relationship between products and days/hours
#product_day = train_orders[['product_id', 'order_dow']]
product_day_freq = pd.DataFrame(prior_orders[['product_id', 'order_dow']].groupby(['product_id', 'order_dow'])['order_dow'].count())
product_day_freq = product_day_freq.rename(columns = {'order_dow': 'day_count'}).reset_index()
#product_hour = train_orders[['product_id', 'order_hour_of_day']]
product_hour_freq = pd.DataFrame(prior_orders[['product_id', 'order_hour_of_day']].groupby(['product_id', 'order_hour_of_day'])['order_hour_of_day'].count())
product_hour_freq = product_hour_freq.rename(columns = {'order_hour_of_day': 'hour_count'}).reset_index()

del prior_orders, user_products


def get_features(features, train=True):
    feature_vector = pd.DataFrame()
    labels = []
    feature_vector['user_id'] = features['user_id']
    feature_vector['order_id'] = features['order_id']
    feature_vector['product_id'] = features['product_id']
    feature_vector['order_dow'] = features['order_dow']
    feature_vector['order_hour_of_day'] = features['order_hour_of_day']
    feature_vector['days_since_prior_order'] = features['days_since_prior_order']
    # get aisle and departid
    feature_vector = pd.merge(features, products, on='product_id', how='left')
    feature_vector = pd.merge(feature_vector, product_info, on='product_id', how='left')
    feature_vector = pd.merge(feature_vector, product_day_freq, on=['product_id', 'order_dow'], how='left')
    feature_vector = pd.merge(feature_vector, product_hour_freq, on=['product_id', 'order_hour_of_day'], how='left')
    #feature_vector = pd.merge(feature_vector, user_num_orders, on='user_id', how='left')
    # get aisle and departid
    if train:
        # merge ordered product based on train set
        labels = feature_vector['reordered']
        feature_vector.drop('reordered', axis=1)
    return feature_vector, labels

# declared used features


features = ['order_dow', 'order_hour_of_day', 'days_since_prior_order',
            'reorder_rate', 'order_total','avg_add_to_cart_order', 'day_count', 'hour_count',
            'reorder_total',
            'aisle_id', 'department_id']

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
train_x, label = get_features(train_orders, train=True)
print('Building dataset...')
# keep features
train_data = lgb.Dataset(train_x[features], label=label, categorical_feature=['aisle_id', 'department_id'])
# starting to train
print('Training......')
bst = lgb.train(params, train_data, num_round)
del train_x
print('Predicting......')
test_x, label = get_features(test_orders, train=False)
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
threshold = 0.6
result = result[result['confidence'] >= threshold]
result = result.groupby('order_id')['product_id'].apply(list).reset_index()
result.columns = ['order_id', 'products']
result['products'] = result['products'].apply(lambda x: " ".join(str(num) for num in x))
submission = pd.read_csv(DIR + 'sample_submission.csv')
submission = submission.drop('products', axis=1)
submission = pd.merge(submission, result, on='order_id', how='left').fillna("None")
print("Writing output")
submission.to_csv('out.csv', index=False, mode='w+', quoting=csv.QUOTE_NONE)

""" TODO: Implement Cross-validation"""
def cross_validate(feature_vector,labels):

    return None
