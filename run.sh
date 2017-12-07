# runs python files sequentially to final submission
cd src
python lgbm_train.py
python xgboost.py
python submit.py
