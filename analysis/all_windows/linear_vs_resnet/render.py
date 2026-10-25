"""Execute the cache-only comparison notebook after training, or on demand."""
from pathlib import Path
import os
import nbformat
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[3]
os.environ['MPLBACKEND']='Agg'
p=ROOT/'analysis/all_windows/linear_vs_resnet.ipynb'
b=nbformat.read(p,as_version=4)
NotebookClient(b,timeout=300,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
nbformat.write(b,p)
print(f'Rendered {p}',flush=True)
