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
print('priors done')

train = pd.read_csv(DIR + 'order_products__train.csv', dtype={
            'order_id': np.int32,
            'product_id': np.uint16,
            'add_to_cart_order': np.int16,
            'reordered': np.int8})
print('train done')

orders = pd.read_csv(DIR + 'orders.csv', dtype={
        'order_id': np.int32,
        'user_id': np.int32,
        'eval_set': 'category',
        'order_number': np.int16,
        'order_dow': np.int8,
        'order_hour_of_day': np.int8,
        'days_since_prior_order': np.float32})
print('orders done')

products = pd.read_csv(DIR + 'products.csv', dtype={
        'product_id': np.uint16,
        'order_id': np.int32,
        'aisle_id': np.uint8,
        'department_id': np.uint8},
        usecols=['product_id', 'aisle_id', 'order_id', 'department_id'])
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
#u_p = pd.DataFrame()
#u_p['up_orders'] = prior_orders.groupby(['user_id', 'product_id'])['reordered'].count()
#u_p['up_reorders'] = prior_orders.groupby(['user_id', 'product_id'])['reordered'].sum()
#u_p['up_reorder_rate'] = u_p['up_reorders'] / u_p['up_orders']
#u_p['up_add_to_cart_order'] = prior_orders.groupby(['user_id', 'product_id'])['add_to_cart_order'].mean()
#u_p['up_days_since_prior_order'] = prior_orders.groupby(['user_id', 'product_id'])['days_since_prior_order'].mean()
#u_p = u_p.reset_index()

#Bobby:
u_features = ['user_id','orders_sum', 'days_since_prior_std','avg_basket', 'avg_reorder', 'num_unique_items']
user_features = pd.read_csv(DIR + 'user_info.csv', dtype={
       'user_id': np.uint32,
       'orders_sum': np.uint16,
       'days_since_prior_std': np.float32,
       'avg_basket': np.float32,
       'avg_reorder': np.float32,
       'num_unique_items': np.uint16},
       usecols=u_features)

#Calculating first and last order diffs
u_f = pd.DataFrame()

order_sum_df = pd.DataFrame(prior_orders[['user_id','order_number']].groupby(['user_id','order_number'])['order_number'].count())
order_sum_df = order_sum_df.rename(columns = {'order_number': 'ord_sum'}).reset_index()

print("DOING COMP ANALYSIS")
wfile = open(DIR + "num5_features.csv", "w")
wfile.write("user,comp_size,avg_diff,std_diff\n")

cur = 0
end = len(user_features)
for u in user_features['user_id']:
    cur = cur+1
    print((cur/end) * 100)
    
    df = (order_sum_df.loc[order_sum_df['user_id'] == u])
    num_orders = max(df['order_number'])


    if(num_orders >= 6):
        comp_size = 3
    elif(num_orders >= 4):
        comp_size = 2
    else:
        comp_size = 1
    
    begin_size = ((df.loc[df['order_number'] == 1])).iloc[0]['ord_sum']
    end_size = ((df.loc[df['order_number'] == num_orders])).iloc[0]['ord_sum']

    comp_1 = ((df.loc[df['order_number'] <= comp_size]))['ord_sum'].tolist()
    comp_2 = ((df.loc[df['order_number'] > (num_orders - comp_size) ]))['ord_sum'].tolist()

    avg_diff = abs(np.average(comp_2) - np.average(comp_1))
    std_diff = abs(np.std(comp_2) - np.std(comp_1))
    comp_size = abs(end_size - begin_size)
    
    wfile.write(str(u) + "," + str(comp_size) + "," +
                str(avg_diff) + "," + str(std_diff) + "\n")

wfile.close()
print("COMP DONE")

op1 = pd.DataFrame(prior_orders[['user_id','order_number']]).groupby(['user_id','order_number'])
options = [prior_orders[['user_id','order_number']], order_sum_df]

df = (prior_orders.loc[prior_orders['user_id'] == 1])
begin_size = len(df.loc[df['order_number'] == 1])
num_orders = max(df['order_number'])
end_size = len(df.loc[df['order_number'] == num_orders])
comp_size = abs(end_size - begin_size)



(prior_orders['user_id'])['order_number']


#del prior_orders
