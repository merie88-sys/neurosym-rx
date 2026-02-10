# Dataset Management

## TwoSIDES Dataset

- **Official source**: https://nsides.io/data/TWOSIDES.csv.gz
- **Description**: Largest publicly available database of drug-drug-effect relationships (645,942 interactions)
- **License**: Academic use only (Tatonetti Lab)

## Usage Policy

⚠️ **We do NOT redistribute the dataset**. Per academic best practices and copyright compliance:

1. Run `download_twosides.py` to fetch directly from official source
2. Run `preprocess.py` to filter high-confidence interactions
3. Processed data saved to `data/processed/twosides_filtered.csv`

## Directory Structure
