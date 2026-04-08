<![CDATA[# Quick Start Guide — IntelliGrid EnergyGuard AI

## Welcome! 🎉

Get up and running in **15 minutes**.

---

## Step 1: Download the Dataset (5 min)

You need the `chilledwater.csv` file. Options:

1. **Kaggle**: https://www.kaggle.com/datasets/claytonmiller/buildingdatagenomeproject2
2. **Zenodo**: https://zenodo.org/records/3887306

Place the CSV in `data/raw/`.

---

## Step 2: Set Up Environment (5 min)

```bash
cd IntelliGrid-EnergyGuard-AI

# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

---

## Step 3: Run the Project

### Option A: Jupyter Notebook (Recommended)

```bash
jupyter notebook notebooks/IntelliGrid_EnergyGuard_AI.ipynb
```

Use **Kernel → Restart & Run All** for a clean execution.

### Option B: Streamlit Dashboard

```bash
streamlit run src/app.py
```

Upload `chilledwater.csv` when prompted. Select a building, adjust sensitivity, and click **Run Anomaly Detection**.

### Option C: Python Script

```bash
python src/complete_pipeline.py
```

Runs the entire pipeline end-to-end.

---

## What You'll Get

- ✅ 5-tier anomaly detection results
- ✅ Classification + Regression + Agreement metrics
- ✅ Automated comparison visualizations
- ✅ Interactive Streamlit dashboard
- ✅ Downloadable anomaly reports (CSV)

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `NameError` in notebook | Run cells **in order** (Kernel → Restart & Run All) |
| Dataset not found | Ensure `chilledwater.csv` is in `data/raw/` |
| Streamlit won't start | Check `streamlit` is installed: `pip install streamlit` |

---

## Learning Path

If you're new to data science, start with the `learning/` folder:
1. `01_Python_Basics.ipynb`
2. `02_Data_Science_Fundamentals.ipynb`
3. `03_Machine_Learning_Basics.ipynb`
4. `05_Anomaly_Detection.ipynb`

Then dive into the main project notebook! 🚀
]]>
