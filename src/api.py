"""FastAPI online inference service for customer churn probability."""
from pathlib import Path
from typing import Literal
import time, joblib, pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from src.features import build_features
ROOT=Path(__file__).resolve().parents[1]
model=joblib.load('C:/BITS Pilani/MLME/mini_production_ml_system/models/churn_model.joblib')
meta=joblib.load('C:/BITS Pilani/MLME/mini_production_ml_system/models/model_metadata.joblib')
app=FastAPI(title='Telco Churn Prediction API',version=meta['model_version'])
class CustomerRequest(BaseModel):
    gender: Literal['Female','Male']; SeniorCitizen:int=Field(ge=0,le=1); Partner:Literal['Yes','No']; Dependents:Literal['Yes','No']; tenure:int=Field(ge=0,le=72)
    PhoneService:str; MultipleLines:str; InternetService:str; OnlineSecurity:str; OnlineBackup:str; DeviceProtection:str; TechSupport:str; StreamingTV:str; StreamingMovies:str
    Contract:str; PaperlessBilling:Literal['Yes','No']; PaymentMethod:str; MonthlyCharges:float=Field(ge=0); TotalCharges:float=Field(ge=0)
@app.get('/health')
def health(): return {'status':'ok','model_version':meta['model_version']}
@app.post('/predict')
def predict(req:CustomerRequest):
    start=time.perf_counter(); df=pd.DataFrame([req.model_dump()]); X=build_features(df).drop(columns=['Churn','customerID'],errors='ignore')
    prob=float(model.predict_proba(X)[:,1][0]); pred=int(prob>=0.5)
    return {'prediction':pred,'label':'Yes' if pred else 'No','churn_probability':round(prob,6),'model_version':meta['model_version'],'latency_ms':round((time.perf_counter()-start)*1000,3)}
