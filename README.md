# Instacart Market Basket Analysis

### Prerequisites

- Python 3.0+
- numpy
- lightGBM
- XGBoost
- pandas

> To meet the prereqs, please see [INSTALL.md](https://gist.github.com/jeremystan/c3b39d947d9b88b3ccff3147dbcf6c6b)

## The Task
The task is to predict which products a user will reorder in their next order. The evaluation metric is the F1-score between the set of predicted products and the set of true products.

Below is the full data schema ([source](https://gist.github.com/jeremystan/c3b39d947d9b88b3ccff3147dbcf6c6b))

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
We have provided all the features .csv files in a google drive where you can download them and put them in corresponding folders.

You can download them at

#### Generate feature csv files (Optional, not recommended)

>Since We have included all the necessary feature files (.csv files in data/ folders). So it is not recommended to run them again and compute yourself. You can skip this to the next step, but if you want to:
>
```
./compute_features.sh
```
This will run all the programs necessary to compute 35 features we have for this project. The results will be saved at **data/processed/*.csv**

### Train/Predict and Submit
```
./run.sh
```

This will use both LightGBM and XGBoost independently to train and predict the data set. The prediction results will be saved at **data/predict/*.csv**. Then, final predictions from both model will be combined to produce a final submission file for Kaggle at

## Authors

* **Robert Cowen**
* **Zhiwen Hu**
* **Linzuo Li**
* **Alex Wang**
* **Xiao Zeng**
