"""Idempotent batch ingestion: merges new CSV files by customerID and logs each run."""
from pathlib import Path
import argparse, json
from datetime import datetime, timezone
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
def ingest(input_csv: str, output_csv: str | None=None):
    inp=Path(input_csv); out=Path(output_csv) if output_csv else 'C:/BITS Pilani/MLME/mini_production_ml_system/data/processed/training_data.csv'
    new=pd.read_csv(inp)
    before=0
    if out.exists():
        old=pd.read_csv(out); before=len(old)
        merged=pd.concat([old,new],ignore_index=True).drop_duplicates('customerID',keep='last')
    else: merged=new.drop_duplicates('customerID',keep='last')
    out.parent.mkdir(parents=True,exist_ok=True); merged.to_csv(out,index=False)
    log={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'source':str(inp),'rows_read':len(new),'rows_before':before,'rows_after':len(merged)}
    (ROOT/'logs').mkdir(exist_ok=True)
    with open(ROOT/'logs/ingestion.jsonl','a') as f: f.write(json.dumps(log)+'\n')
    print(json.dumps(log,indent=2)); return log
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('input_csv'); p.add_argument('--output_csv'); a=p.parse_args(); ingest(a.input_csv,a.output_csv)
