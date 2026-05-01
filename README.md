# IntelliGrid EnergyGuard AI

A machine learning pipeline that detects energy consumption anomalies in commercial buildings using chilled water system data.

## Overview

Commercial buildings waste significant energy due to equipment failures, inefficient operations, and occupancy mismatches. This project uses a multi-tier approach to identify these anomalies automatically, combining statistical methods with unsupervised machine learning.

The pipeline processes real building data through five detection layers:
- Rolling Z-Score analysis for statistical outliers
- Temporal pattern detection comparing against seasonal baselines
- Isolation Forest for high-dimensional anomaly detection
- Local Outlier Factor for density-based detection
- One-Class SVM for boundary-based identification

A consensus mechanism combines all five methods - if at least three agree on an anomaly, it's flagged for investigation.

## Installation

```bash
git clone https://github.com/dev-ayusman/IntelliGrid-EnergyGuard-AI.git
cd IntelliGrid-EnergyGuard-AI
pip install -r requirements.txt
```

## Usage

### Jupyter Notebook
The main analysis is in `notebooks/IntelliGrid_EnergyGuard_AI.ipynb`. Open it with:
```bash
jupyter notebook notebooks/IntelliGrid_EnergyGuard_AI.ipynb
```

### Streamlit Dashboard
Launch the interactive dashboard:
```bash
streamlit run src/app.py
```
Upload your `chilledwater.csv` file to visualize anomalies and estimate cost impacts.

### Command Line
Run the full pipeline directly:
```bash
python src/complete_pipeline.py
```

## Dataset

Place your `chilledwater.csv` file in `data/raw/`. The dataset should contain building energy readings with timestamps. See `data/DOWNLOAD_INSTRUCTIONS.md` for sourcing the data.

## Project Structure

```
IntelliGrid-EnergyGuard-AI/
├── notebooks/          # Main analysis notebook
├── src/               # Python modules and scripts
│   ├── app.py                    # Streamlit dashboard
│   ├── complete_pipeline.py      # End-to-end pipeline
│   ├── model_manager.py          # Model persistence utilities
│   └── predict_with_saved_models.py
├── data/
│   ├── raw/           # Place dataset here
│   └── processed/     # Cleaned data output
├── models/            # Saved model artifacts
├── results/           # Generated visualizations
├── learning/          # Educational notebooks
└── tests/             # Unit tests
```

## Technologies

- **Python 3.9+**
- **pandas, numpy** - Data processing
- **scikit-learn** - Machine learning models
- **matplotlib, seaborn, plotly** - Visualizations
- **streamlit** - Interactive dashboard
- **scipy** - Statistical computations

## Results

The pipeline generates:
- Anomaly detection reports with severity classification
- Cost impact estimates based on configurable energy rates
- Interactive timelines showing normal vs anomalous consumption
- Model comparison visualizations
- CSV exports for further analysis

## Future Work

- LSTM autoencoder for sequential pattern detection
- Real-time streaming detection
- Model explainability with SHAP
- REST API for integration with building management systems

## License

MIT License - see LICENSE file for details.

## Author

Ayusman Choudhury
