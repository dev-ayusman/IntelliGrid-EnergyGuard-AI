<![CDATA[# ⚡ IntelliGrid EnergyGuard AI

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E.svg)](https://scikit-learn.org)

> **AI-Powered Building Energy Anomaly Detection System**
>
> A multi-tier machine learning pipeline that detects, classifies, and evaluates energy consumption anomalies in commercial buildings using real-world chilled water data.

---

## 🎯 Project Overview

Commercial buildings account for approximately **30% of global energy consumption**, with anomalies — unexpected spikes or drops caused by equipment failures, operational inefficiencies, or occupancy mismatches — driving billions of dollars in annual waste.

**IntelliGrid EnergyGuard AI** tackles this challenge through a sophisticated **5-tier anomaly detection pipeline** that combines statistical methods with machine learning to robustly identify energy anomalies.

### Key Capabilities

- 🔍 **Multi-method anomaly detection** — Rolling Z-Score, Temporal, Isolation Forest, LOF, and One-Class SVM
- 🗳️ **Consensus ground truth** via majority voting across all 5 detection signals
- 📊 **Comprehensive evaluation** — Classification, Regression, and Agreement metrics
- 📈 **Interactive Streamlit dashboard** for real-time anomaly monitoring
- 📉 **Automated visualizations** — Heatmaps, time-series plots, and model comparison charts

---

## 🗂️ Project Structure

```
IntelliGrid-EnergyGuard-AI/
├── notebooks/
│   └── IntelliGrid_EnergyGuard_AI.ipynb    # Main project notebook (full pipeline)
├── src/
│   ├── app.py                               # Streamlit interactive dashboard
│   ├── complete_pipeline.py                 # End-to-end pipeline script
│   ├── complete_pipeline_with_saving.py     # Pipeline with model persistence
│   ├── model_manager.py                     # Model save/load utilities
│   ├── predict_with_saved_models.py         # Inference with saved models
│   └── __init__.py
├── data/
│   ├── raw/                                 # Place chilledwater.csv here
│   ├── processed/                           # Cleaned output data
│   └── DOWNLOAD_INSTRUCTIONS.md
├── models/                                  # Trained model artifacts (.pkl)
├── results/                                 # Generated visualizations & reports
├── learning/                                # Educational notebooks (beginner-friendly)
│   ├── 01_Python_Basics.ipynb
│   ├── 02_Data_Science_Fundamentals.ipynb
│   ├── 03_Machine_Learning_Basics.ipynb
│   └── 05_Anomaly_Detection.ipynb
├── requirements.txt
├── LICENSE
├── CONTRIBUTING.md
├── QUICK_START.md
└── README.md
```

---

## 🔬 Methodology: 5-Tier Anomaly Detection Pipeline

### Tier 1 — Exploratory Data Analysis (EDA)
- Load and inspect the chilled water dataset
- Analyze distributions, missing values, and temporal patterns
- Generate correlation heatmaps and summary statistics

### Tier 2 — Statistical Anomaly Detection (Rolling Z-Score)
- Compute rolling mean and standard deviation (7-day window)
- Flag points where Z-score exceeds threshold **AND** value is significantly above rolling mean
- Per-building statistical anomaly maps

### Tier 3 — Temporal Anomaly Detection
- Compare each reading against the seasonal baseline (median by month × hour)
- Flag readings deviating more than 3× the seasonal median
- Captures time-of-day and seasonal irregularities

### Tier 4 — Machine Learning Models
Three unsupervised ML algorithms applied independently:

| Model | Approach | Strengths |
|-------|----------|-----------|
| **Isolation Forest** | Tree-based isolation | High-dimensional data, scalable |
| **Local Outlier Factor (LOF)** | Density-based comparison | Local anomaly detection |
| **One-Class SVM** | Boundary learning | Robust to noise |

### Tier 5 — Consensus & Evaluation
- **Majority Voting**: A data point is flagged as a "true anomaly" if ≥ 3 out of 5 methods agree
- **Comprehensive Metrics**:

| Category | Metrics |
|----------|---------|
| **Classification** | Precision, Recall, F1-Score, Specificity, Accuracy, Balanced Accuracy |
| **Agreement** | Matthews Correlation Coefficient (MCC), Cohen's Kappa, Jaccard Score |
| **Regression-style** | Mean Squared Error (MSE), Mean Absolute Error (MAE), R² Score |

- **Automated Visualizations**: 2×2 classification chart + 1×3 regression chart

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/IntelliGrid-EnergyGuard-AI.git
cd IntelliGrid-EnergyGuard-AI
```

### 2. Set Up Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Add the Dataset

Download the **chilledwater.csv** file and place it in the `data/raw/` directory.
See [data/DOWNLOAD_INSTRUCTIONS.md](data/DOWNLOAD_INSTRUCTIONS.md) for details.

### 4. Run the Notebook

```bash
jupyter notebook notebooks/IntelliGrid_EnergyGuard_AI.ipynb
```

Run all cells in order (Kernel → Restart & Run All).

### 5. Launch the Dashboard

```bash
streamlit run src/app.py
```

Upload your `chilledwater.csv` when prompted.

---

## 📊 Dashboard Preview

The Streamlit dashboard provides:
- **Building selector** — Choose from top-5 most complete buildings
- **Sensitivity slider** — Adjust detection threshold in real-time
- **Cost calculator** — Estimate annual energy waste at custom $/unit rates
- **Interactive timeline** — Plotly chart with anomaly markers
- **Downloadable reports** — Export anomaly logs as CSV

---

## 🛠️ Technologies

| Tool | Purpose |
|------|---------|
| Python 3.9+ | Core language |
| Pandas & NumPy | Data manipulation |
| Scikit-learn | ML algorithms & metrics |
| Matplotlib & Seaborn | Static visualizations |
| Plotly | Interactive charts |
| Streamlit | Web dashboard |
| SciPy | Statistical computations |
| Jupyter | Interactive development |

---

## 📈 Future Roadmap

- [ ] LSTM Autoencoder for sequential anomaly patterns
- [ ] Real-time streaming detection pipeline
- [ ] SHAP-based model explainability
- [ ] Predictive maintenance alerts
- [ ] REST API deployment
- [ ] Multi-building comparative analysis

---

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 📚 References

1. Miller, C., Kathirgamanathan, A., Picchetti, B. et al. (2020). *The Building Data Genome Project 2*. Scientific Data 7, 368. [DOI](https://doi.org/10.1038/s41597-020-00712-x)
2. Liu, F.T., Ting, K.M., & Zhou, Z.H. (2008). *Isolation Forest*. IEEE ICDM.
3. Breunig, M.M. et al. (2000). *LOF: Identifying Density-Based Local Outliers*. ACM SIGMOD.

---

**⭐ Star this repo if you find it useful!**

Built with ❤️ by Ayusman Choudhury
]]>
