"""Repeatable training and offline evaluation pipeline."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from src.features import split_xy

ROOT = Path(__file__).resolve().parents[1]
MODEL_VERSION = "telco-churn-logreg-v1.0.0"

def metrics(y, pred, proba):
    return {"accuracy": accuracy_score(y,pred), "roc_auc": roc_auc_score(y,proba),
            "precision": precision_score(y,pred,zero_division=0), "recall": recall_score(y,pred,zero_division=0),
            "f1": f1_score(y,pred,zero_division=0), "confusion_matrix": confusion_matrix(y,pred).tolist()}

def main():
    df=pd.read_csv('C:/BITS Pilani/MLME/mini_production_ml_system/data/raw/telco_customer_churn.csv')
    X,y=split_xy(df)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.20,stratify=y,random_state=42)
    cat=X_train.select_dtypes(include=['object','category']).columns.tolist()
    num=[c for c in X_train.columns if c not in cat]
    pre=ColumnTransformer([('num',Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),
                           ('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),cat)])
    baseline=Pipeline([('pre',pre),('model',DummyClassifier(strategy='prior'))])
    candidate=Pipeline([('pre',pre),('model',LogisticRegression(max_iter=1500,class_weight='balanced',C=0.7,random_state=42))])
    results={}
    for name,model in [('baseline',baseline),('candidate',candidate)]:
        model.fit(X_train,y_train)
        p=model.predict(X_test); pr=model.predict_proba(X_test)[:,1]
        results[name]=metrics(y_test,p,pr)
    promote=results['candidate']['roc_auc']>=0.80 and results['candidate']['roc_auc']>=results['baseline']['roc_auc']+0.01
    results['promotion']={'rule':'candidate_auc >= 0.80 and >= baseline_auc + 0.01','promote':bool(promote),'model_version':MODEL_VERSION}
    (ROOT/'models').mkdir(exist_ok=True); ('C:/BITS Pilani/MLME/mini_production_ml_system/artifacts/eval').mkdir(parents=True,exist_ok=True)
    joblib.dump(candidate,'C:/BITS Pilani/MLME/mini_production_ml_system/models/churn_model.joblib')
    joblib.dump({'model_version':MODEL_VERSION,'feature_columns':X.columns.tolist()},'C:/BITS Pilani/MLME/mini_production_ml_system/models/model_metadata.joblib')
    with open(ROOT/'artifacts/eval/evaluation_report.json','w') as f: json.dump(results,f,indent=2)
    pd.DataFrame({'y_true':y_test,'y_pred':candidate.predict(X_test),'churn_probability':candidate.predict_proba(X_test)[:,1]}).to_csv('C:/BITS Pilani/MLME/mini_production_ml_system/artifacts/eval/test_predictions.csv',index=False)
    print(json.dumps(results,indent=2))
if __name__=='__main__': main()
