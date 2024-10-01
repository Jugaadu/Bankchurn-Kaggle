import os
import pandas as pd
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
import joblib

from . import dispacher

TRAINING_DATA =os.environ.get("TRAINING_DATA")
TEST_DATA =os.environ.get("TEST_DATA")
FOLD =int(os.environ.get("FOLD"))
MODEL = os.environ.get("MODEL")


FOLD_MAPPING = {
    0: [1, 2, 3, 4],
    1: [0, 2, 3, 4],
    2: [0, 1, 3, 4],
    3: [0, 1, 2, 4],
    4: [0, 1, 2, 3],
}


if __name__ == "__main__":


    df = pd.read_csv(TRAINING_DATA)
    test_df = pd.read_csv(TEST_DATA)
    train_df = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))].reset_index(drop = True)
    valid_df = df[df.kfold == FOLD].reset_index(drop=True)

    y_train = train_df.Exited.values
    y_valid = valid_df.Exited.values

    train_df = train_df.drop(["id", "Exited", "kfold"], axis = 1)
    valid_df = valid_df.drop(['id', 'Exited', 'kfold'], axis = 1)

    categorical_vars = ['CustomerId','Surname','Geography', 'Gender']

    label_encoders = {}
    for c in categorical_vars:
        lbl = preprocessing.LabelEncoder()
        train_df.loc[:,c] = train_df.loc[:,c].astype(str).fillna("NONE") 
        valid_df.loc[:,c] = valid_df.loc[:,c].astype(str).fillna("NONE") 
        test_df.loc[:,c] = test_df.loc[:,c].astype(str).fillna("NONE") 
        lbl.fit(train_df[c].values.tolist() + valid_df[c].values.tolist()+ test_df[c].values.tolist())

        train_df.loc[:,c] = lbl.transform(train_df[c].values.tolist())
        valid_df.loc[:,c] = lbl.transform(valid_df[c].values.tolist())
        test_df.loc[:,c] = lbl.transform(test_df[c].values.tolist())
        label_encoders[c] = lbl

    clf = dispacher.MODELS[MODEL]
    clf.fit(train_df, y_train)
    preds = clf.predict_proba(valid_df)[:,1]
    print(metrics.roc_auc_score(y_valid,preds))


    joblib.dump(label_encoders, f'models/{MODEL}_{FOLD}_label_encoders.pkl')
    joblib.dump(clf, f'models/{MODEL}_{FOLD}.pkl')
    joblib.dump(train_df.columns, f'models/{MODEL}_{FOLD}_columns.pkl')
