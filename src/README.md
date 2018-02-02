### Submit Results to Kaggle

The running of submit.py requires two csv files located at data/predicted. They can be computed be ./run.sh

- lgbm_pred.csv
- xgboost_pred.csv

```
cd /src
python submit.py
```

**or** run it with optional arguments

```
cd /src
python submit.py -l LWEIGHT -x XWEIGHT --threshold THRESHOLDS
```

- LWEIGHT: weight constant on LightGBM predictions (Default: 0.4)

- XWEIGHT: weight constant on XGBoost predictions (Default: 0.6)

- THRESHOLDS: output threshold for binary classification (Default: 0.20)
