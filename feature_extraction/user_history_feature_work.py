################################################################
#   user_history_feature_work.py
#   Author: Team ZLABS
#   Created for UCLA CS249
#
#   Work script to calculate the average difference between a users
#   first orders versus their last orders
###############################################################

import numpy as np
import pandas as pd
import lightgbm as lgb
import csv
import os
DIR = "..\\data\\"
out_dir = DIR + "predict\\"


#Load past csv information
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

#Bobby:
u_features = ['user_id','orders_sum', 'days_since_prior_std','avg_basket', 'avg_reorder', 'num_unique_items']
user_features = pd.read_csv(out_dir + 'user_info_2.csv', dtype={
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


#Calculate the comparative differences between a users first orders and last orders
print("DOING HISTORY COMP ANALYSIS")
wfile = open(out_dir + "num5_features.csv", "w")
wfile.write("comp_size,avg_diff,std_diff\n")

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
    
    wfile.write(str(comp_size) + "," +
                str(avg_diff) + "," + str(std_diff) + "\n")

wfile.close()
print("COMP DONE")

print("COMBINING ALL USER FEATURE FILES")

wfile = open(out_dir + "user_info.csv", "w")
rfile1 = open(out_dir + "user_info_2.csv", "r")
rfile2 = open(out_dir + "num5_features.csv", "r")

lines_1 = rfile1.readlines()
lines_2 = rfile2.readlines()

for line in range(len(lines_1)):
    line1 = ((lines_1[line]).split("\n"))[0]
    line2 = ((lines_2[line]).split("\n"))[0]

    wfile.write( line1 + "," + line2 + "\n")

wfile.close()
rfile1.close()
rfile2.close()

print("REMOVING TEMP FILES:")
try:
    os.remove(out_dir + "user_info_1.csv")
except OSError:
    pass

try:
    os.remove(out_dir + "user_info_2.csv")
except OSError:
    pass

try:
    os.remove(out_dir + "num5_features.csv")
except OSError:
    pass
