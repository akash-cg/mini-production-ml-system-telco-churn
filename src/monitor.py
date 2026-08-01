"""Lightweight data-quality and drift checks for a recent batch."""
from pathlib import Path
import json, argparse
import pandas as pd
from src.features import clean_raw_data
ROOT=Path(__file__).resolve().parents[1]
def check(reference_csv, recent_csv):
    ref=clean_raw_data(pd.read_csv(reference_csv)); rec=clean_raw_data(pd.read_csv(recent_csv))
    warnings=[]
    required=['tenure','MonthlyCharges','TotalCharges','Contract','InternetService']
    missing_cols=[c for c in required if c not in rec.columns]
    if missing_cols: warnings.append(f'Missing columns: {missing_cols}')
    null_rate=float(rec[required].isna().mean().max()) if not missing_cols else 1.0
    if null_rate>0.02: warnings.append(f'Null rate exceeds 2%: {null_rate:.2%}')
    if ((rec['tenure']<0)|(rec['tenure']>72)).any(): warnings.append('tenure outside [0,72]')
    drift={}
    for c in ['MonthlyCharges','tenure']:
        score=abs(rec[c].mean()-ref[c].mean())/(ref[c].std()+1e-9)
        drift[c]=float(score)
        if score>0.25: warnings.append(f'{c} mean drift z-score {score:.3f} > 0.25')
    report={'rows':len(rec),'max_null_rate':null_rate,'drift_scores':drift,'status':'WARNING' if warnings else 'PASS','warnings':warnings}
    out=ROOT/'artifacts/monitoring/data_quality_report.json'; out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2)); return report
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('reference_csv'); p.add_argument('recent_csv'); a=p.parse_args(); check(a.reference_csv,a.recent_csv)
