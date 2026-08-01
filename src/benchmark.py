"""Local inference benchmark using the saved pipeline (no network overhead)."""
from pathlib import Path
import time, json, joblib, numpy as np, pandas as pd
from src.features import build_features
ROOT=Path(__file__).resolve().parents[1]
def main(n=500):
    model=joblib.load('C:/BITS Pilani/MLME/mini_production_ml_system/models/churn_model.joblib'); raw=pd.read_csv('C:/BITS Pilani/MLME/mini_production_ml_system/data/raw/telco_customer_churn.csv').drop(columns=['Churn','customerID']).head(n)
    times=[]
    for _,r in raw.iterrows():
        x=build_features(pd.DataFrame([r])).drop(columns=['Churn','customerID'],errors='ignore'); t=time.perf_counter(); model.predict_proba(x); times.append((time.perf_counter()-t)*1000)
    result={'requests':len(times),'avg_latency_ms':float(np.mean(times)),'p95_latency_ms':float(np.percentile(times,95)),'throughput_rows_per_sec':float(1000/np.mean(times))}
    ('C:/BITS Pilani/MLME/mini_production_ml_system/artifacts/eval/latency_report.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
