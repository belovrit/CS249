# runs python files sequentially to final submission
cd feature_extraction
python features.py
cd ../
cd src
python lgbm_train.py
python xgboost_train.py
python submit.py
