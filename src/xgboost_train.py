import pandas as pd
import xgboost
import time
import numpy as np
#from sklearn.model_selection import train_test_split

DIR = '../data/'

start = time.time()

print("GETTING TRAINING FEATURES...")
train_x = pd.read_csv(DIR + 'processed/train_features.csv')
train_x = train_x.replace([np.inf, -np.inf], np.nan).fillna(0)
print("\tDone.")

train_labels = train_x['reordered']

print("GETTING TESTING FEATURES...")
test_x = pd.read_csv(DIR + 'processed/test_features.csv')
test_x = test_x.replace([np.inf, -np.inf], np.nan).fillna(0)

features = ['order_dow', 'order_hour_of_day', 'days_since_prior_order',
        'reorder_rate', 'order_total','avg_add_to_cart_order', 'day_count', 'hour_count',
        'reorder_total', 'aisle_id', 'department_id', 'up_orders', 'up_reorders', 'up_reorder_rate',
        'up_add_to_cart_order', 'up_days_since_prior_order', 'delta_dow', 'delta_order_hour_of_day',
        'ordered_last_time', 'aisle_orders', 'depart_orders', 'numbers_since_last_order', 'first_last',
        'first_ordered_number', 'ratio_since_first_purchase', 'bought_times', 'like_organic',
        'mean_length_product_name', 'user_order_sum', 'user_avg_interval'
        ]



#X_train, X_val, y_train, y_val = train_test_split(train_x[features], train_labels, test_size=0.9, random_state=42)

d_train = xgboost.DMatrix(train_x[features], train_labels)
d_test = xgboost.DMatrix(test_x[features])

params = {
    "objective"         : "binary:logistic"
    ,"eval_metric"      : "logloss"
    ,"eta"              : 0.1
    ,"max_depth"        : 6
    ,"min_child_weight" :10
    ,"gamma"            :0.70
    ,"subsample"        :0.76
    ,"colsample_bytree" :0.95
    ,"alpha"            :2e-05
    ,"lambda"           :10
}

watchlist= [(d_train, "train")]
print("Training")
bst = xgboost.train(params=params, dtrain=d_train, num_boost_round=100, evals=watchlist, verbose_eval=5)
print("\tDone")
print("Predicting")
test_x = test_x[['order_id', 'product_id']]
test_x['confidence'] = bst.predict(d_test)
print("\tDone")

print("Writing to xgboost_pred.csv")
test_x.to_csv(DIR+ 'predict/xgboost_pred.csv', mode='w+', index=False)

end = time.time()
print(str((end - start) / 60), "minutes")
