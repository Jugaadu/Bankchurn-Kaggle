from sklearn import ensemble
import xgboost as xgb

MODELS ={

        "randomforest": ensemble.RandomForestClassifier(n_estimators = 200, n_jobs = -1, verbose = 2),
        "extratrees" : ensemble.ExtraTreesClassifier(n_estimators = 200, n_jobs = -1, verbose = 2),
        "xgboost": xgb.XGBClassifier(n_estimators = 200, learning_rate = 0.05,  early_stopping_rounds = 5, n_jobs = -1)
        }

