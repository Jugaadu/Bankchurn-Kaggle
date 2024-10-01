import os
import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
import joblib

from . import dispacher

MODEL = os.environ.get("MODEL")
TEST_DATA = os.environ.get("TEST_DATA")

def predict(test_data, model_type, model_path):

    df = pd.read_csv(test_data)
    predictions = None
    test_idx = df.id.values
    for FOLD in range(5):
        df = pd.read_csv(test_data)

        encoders = joblib.load(f"{model_path}/{model_type}_{FOLD}_label_encoders.pkl")
        cols = joblib.load(f"{model_path}/{model_type}_{FOLD}_columns.pkl")

        for c in encoders:
            lbl = encoders[c]
            df.loc[:,c] = df.loc[:,c].astype(str).fillna("NONE")
            df.loc[:,c] = lbl.transform(df[c].values.tolist())

        df = df[cols]
        clf = joblib.load(f"{model_path}/{model_type}_{FOLD}.pkl")
        
        if MODEL =="xgboost":
            preds = clf.predict(df)[:,1]
        else:
            preds = clf.predict_proba(df)[:,1]
        

        if FOLD ==0:
            predictions = preds
        else:
            predictions += preds
    predictions /= 5

    sub = pd.DataFrame(np.column_stack((test_idx, predictions)), columns=["id", "Exited"])
    return sub



if __name__ =="__main__":

    submission = predict(test_data = TEST_DATA, model_type= MODEL, model_path = "models")
    submission.loc[:,"id"] = submission.loc[:,"id"].astype(int)
    submission.to_csv(f"models/{MODEL}_sumbission.csv",index= False)
