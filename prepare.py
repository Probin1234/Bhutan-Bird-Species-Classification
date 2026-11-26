"""Prepare a manually checklist-verified subset from the original folder dataset."""
import argparse, hashlib, json
from pathlib import Path
from PIL import Image
import pandas as pd
from config import DATA
p=argparse.ArgumentParser(); p.add_argument('--source',type=Path,default=DATA/'original'); p.add_argument('--limit',type=int,default=100); a=p.parse_args()
# CSV must contain exact dataset folder names and checklist evidence URLs.
classes=pd.read_csv(DATA/'species.csv').dropna(subset=['folder','evidence'])
if len(classes)<2: raise ValueError('Verify at least two species in species.csv')
rows=[]; seen=set(); names=sorted(classes.folder.unique()); rng=__import__('random').Random(42)
for split in ['test','valid','train']:
 for name in names:
  folder=a.source/split/name
  if not folder.is_dir(): raise FileNotFoundError(folder)
  files=sorted(folder.iterdir()); rng.shuffle(files); count=0
  for f in files:
   if not f.is_file(): continue
   try:
    with Image.open(f) as im: im.verify()
    with Image.open(f) as im: digest=hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()
   except Exception: continue
   if digest in seen: continue
   seen.add(digest); rows.append((str(f.resolve()),names.index(name),split)); count+=1
   if split=='train' and count>=a.limit: break
  if not count: raise ValueError(f'No usable images: {folder}')
pd.DataFrame(rows,columns=['path','label','split']).to_csv(DATA/'manifest.csv',index=False)
(DATA/'classes.json').write_text(json.dumps(names)); print(pd.DataFrame(rows,columns=['path','label','split']).groupby(['split','label']).size())
