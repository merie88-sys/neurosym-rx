#!/usr/bin/env python3
"""
Download TwoSIDES dataset from official Tatonetti Lab source
Source: https://nsides.io/data/TWOSIDES.csv.gz
No redistribution — fetches directly from nsides.io

Author: Anonymous (for double-blind review)
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

def download_twosides():
    """Download TwoSIDES dataset from official source"""
    os.makedirs(RAW_DIR, exist_ok=True)
    
    if os.path.exists(CSV_PATH):
        print(f"✓ TwoSIDES already downloaded: {CSV_PATH}")
        return CSV_PATH
    
    print("📥 Downloading TwoSIDES from official source (nsides.io)...")
    print(f"   URL: {URL}")
    r = requests.get(URL, stream=True)
    total = int(r.headers.get('content-length', 0))
    
    with open(RAW_PATH, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True, desc="Downloading") as pbar:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))
    
    print("✓ Decompressing...")
    with gzip.open(RAW_PATH, 'rb') as f_in, open(CSV_PATH, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    
    size_gb = os.path.getsize(CSV_PATH) / 1e9
    print(f"✅ TwoSIDES ready: {CSV_PATH} ({size_gb:.2f} GB)")
    print(f"   Total interactions: {sum(1 for _ in open(CSV_PATH)) - 1:,}")
    return CSV_PATH

if __name__ == "__main__":
    download_twosides()
