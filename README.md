# Instacart Market Basket Analysis

### Prerequisites

- Python 3.0+
- numpy
- lightGBM
- XGBoost
- pandas

> To meet the prereqs, please see [INSTALL.md](https://github.com/belovrit/CS249/blob/submission/INSTALL.md)

## The Task
The task is to predict which products a user will reorder in their next order. The evaluation metric is the F1-score between the set of predicted products and the set of true products.

Below is the full data schema ([source](https://www.kaggle.com/c/instacart-market-basket-analysis/data))

 > `orders` (3.4m rows, 206k users):
 > * `order_id`: order identifier
 > * `user_id`: customer identifier
 > * `eval_set`: which evaluation set this order belongs in (see `SET` described below)
 > * `order_number`: the order sequence number for this user (1 = first, n = nth)
 > * `order_dow`: the day of the week the order was placed on
 > * `order_hour_of_day`: the hour of the day the order was placed on
 > * `days_since_prior`: days since the last order, capped at 30 (with NAs for `order_number` = 1)
 >
 > `products` (50k rows):
 > * `product_id`: product identifier
 > * `product_name`: name of the product
 > * `aisle_id`: foreign key
 > * `department_id`: foreign key
 >
 > `aisles` (134 rows):
 > * `aisle_id`: aisle identifier
 > * `aisle`: the name of the aisle
 >
 > `deptartments` (21 rows):
 > * `department_id`: department identifier
 > * `department`: the name of the department
 >
 > `order_products__SET` (30m+ rows):
 > * `order_id`: foreign key
 > * `product_id`: foreign key
 > * `add_to_cart_order`: order in which each product was added to cart
 > * `reordered`: 1 if this product has been ordered by this user in the past, 0 otherwise
 >
 > where `SET` is one of the four following evaluation sets (`eval_set` in `orders`):
 > * `"prior"`: orders prior to that users most recent order (~3.2m orders)
 > * `"train"`: training data supplied to participants (~131k orders)
 > * `"test"`: test data reserved for machine learning competitions (~75k orders)

## The Approach
We manually extracted 35 features based on the data given. And used LightGBM and XGBoost as our top level model. Then, a weighted average from these two models are combined as our final result.

## Running
We have provided a data folder with all .csv files in a google drive. Please go ahead and download them and replace them with the project's ./data/ folder

You can download the data from [google drive](https://drive.google.com/open?id=1LSmh5qglsCY4lGjRYmhIpkYzn5GP6nDN)

### Submit
```
./run.sh
```
This will first run all the programs necessary to compute 30 features. The results will be saved at **data/processed/*.csv**. Then it will run both LightGBM and XGBoost independently to train and predict the data set. The prediction results will be saved at **data/predict/*.csv**. Final predictions from both model will be combined (0.6 XGBoost, 0.4 LightGBM)to produce a submission file for Kaggle at **./out.csv**

## Authors
* **Robert Cowen**
* **Zhiwen Hu**
* **Linzuo Li**
* **Alex Wang**
* **Xiao Zeng**
