import pandas as pd
import xgboost
import time
from sklearn.model_selection import train_test_split

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

features = ['order_dow', 'order_hour_of_day', 'days_since_prior_order',
        'reorder_rate', 'order_total','avg_add_to_cart_order', 'day_count', 'hour_count',
        'reorder_total', 'orders_sum', 'days_since_prior_std','avg_basket', 'avg_reorder', 'num_unique_items',
        'aisle_id', 'department_id', 'up_orders', 'up_reorders', 'up_reorder_rate', 'up_add_to_cart_order',
        'up_days_since_prior_order', 'order_ratio', 'delta_dow', 'delta_order_hour_of_day', 'ordered_last_time',
        'reorder_total_ratio', 'numbers_since_last_order', 'first_ordered_number',
        'comp_size', 'avg_diff', 'std_diff']

#X_train, X_val, y_train, y_val = train_test_split(train_x[features], train_labels, test_size=0.9, random_state=42)

d_train = xgboost.DMatrix(train_x[features], train_labels)
d_test = xgboost.DMatrix(test_x[features])

params = {
    "objective"         : "reg:logistic"
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
bst = xgboost.train(params=params, dtrain=d_train, num_boost_round=80, evals=watchlist, verbose_eval=5)
print("\tDone")
print("Predicting")
test_x = test_x[['order_id', 'product_id']]
test_x['confidence'] = bst.predict(d_test)
print("\tDone")

print("Writing to xgboost_pred.csv")
test_x.to_csv('xgboost_pred.csv', mode='w+', index=False)

end = time.time()
print(str((end - start) / 60), "minutes")
