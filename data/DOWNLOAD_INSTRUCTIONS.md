# Dataset Download Instructions

## How to Download the Building Data Genome Project 2 Dataset

### Option 1: Kaggle (Recommended - Easiest)

1. Visit: https://www.kaggle.com/datasets/claytonmiller/buildingdatagenomeproject2
2. Click the **"Download"** button (you may need to create a free Kaggle account)
3. Extract the downloaded ZIP file
4. Copy the CSV files to: `data/raw/`

### Option 2: Zenodo (Official Repository)

1. Visit: https://zenodo.org/records/3887306
2. Scroll down to **"Files"** section
3. Click **"building-data-genome-project-2-v1.0.zip"** (595 MB)
4. Extract the ZIP file
5. Navigate to `data/meters/raw/` folder inside
6. Copy all CSV files to your project's `data/raw/` folder

### Option 3: Direct Download (if above fail)

You can also download via GitHub repository:
- https://github.com/buds-lab/building-data-genome-project-2

### What You Should Have

After downloading and extracting, your `data/raw/` folder should contain:
- Multiple CSV files (one per building or meter)
- Or a single large combined CSV file
- Total size: ~1-2 GB uncompressed

### Verification

Run this Python code to verify:

```python
import os

raw_data_path = 'data/raw/'
files = [f for f in os.listdir(raw_data_path) if f.endswith('.csv')]

print(f"Found {len(files)} CSV files in data/raw/")
print("First 5 files:", files[:5])
```

### Need Help?

If you encounter issues downloading:
1. Check your internet connection
2. Try a different browser
3. Use a VPN if the site is blocked
4. Ask for help in the project issues section

---

**Next Step**: Once downloaded, open `notebooks/01_data_exploration.ipynb` to start the project!
