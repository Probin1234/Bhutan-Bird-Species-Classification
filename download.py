"""Download selected split archives from the proposed dataset mirror; no silent substitution."""
import argparse, tarfile, requests
from config import DATA
p=argparse.ArgumentParser(); p.add_argument('--split',choices=['train','valid','test','all'],default='all'); a=p.parse_args()
root=DATA/'original'; root.mkdir(exist_ok=True)
for split in (['train','valid','test'] if a.split=='all' else [a.split]):
 url=f'https://huggingface.co/datasets/chriamue/bird-species-dataset/resolve/main/data/{split}.tar.gz'
 dest=DATA/f'{split}.tar.gz'
 if not dest.exists():
  with requests.get(url,stream=True,timeout=(20,120)) as r:
   r.raise_for_status()
   with dest.with_suffix('.part').open('wb') as f:
    for chunk in r.iter_content(1024*1024): f.write(chunk)
  dest.with_suffix('.part').replace(dest)
 with tarfile.open(dest) as t:
  for m in t.getmembers():
   target=(root/m.name).resolve()
   if not target.is_relative_to(root.resolve()) or not (m.isfile() or m.isdir()): raise ValueError('Unsafe archive member')
  t.extractall(root)
 print('Extracted',split)
