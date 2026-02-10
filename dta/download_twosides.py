#!/usr/bin/env python3
"""
Download TwoSIDES dataset from official source (nsides.io)
→ NO redistribution: fetches directly from Tatonetti Lab servers
→ Preserves academic integrity and copyright compliance
"""

import requests
import gzip
import shutil
import os
from tqdm import tqdm

URL = "https://nsides.io/data/TWOSIDES.csv.gz"
RAW_DIR = "data/raw"
RAW_PATH = f"{RAW_DIR}/TWOSIDES.csv.gz"
CSV_PATH = f"{RAW_DIR}/TWOSIDES.csv"

def download():
    os.makedirs(RAW_DIR, exist_ok=True)
    
    if os.path.exists(CSV_PATH):
        print(f"✓ TwoSIDES already downloaded: {CSV_PATH}")
        return
    
    print("📥 Downloading TwoSIDES from nsides.io (official source)...")
    r = requests.get(URL, stream=True)
    total = int(r.headers.get('content-length', 0))
    
    with open(RAW_PATH, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True, desc="TWOSIDES.csv.gz") as pbar:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))
    
    print("✓ Download complete. Decompressing...")
    with gzip.open(RAW_PATH, 'rb') as f_in, open(CSV_PATH, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    
    size_gb = os.path.getsize(CSV_PATH) / 1e9
    print(f"✅ TwoSIDES ready: {CSV_PATH} ({size_gb:.2f} GB)")

if __name__ == "__main__":
    download()
