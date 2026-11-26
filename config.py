from pathlib import Path
ROOT=Path(__file__).resolve().parent
DATA=ROOT/'dataset'; MODELS=ROOT/'models'; RESULTS=ROOT/'results'
SIZE=(224,224); SEED=42; BATCH=16
for p in (DATA,MODELS,RESULTS): p.mkdir(exist_ok=True)
