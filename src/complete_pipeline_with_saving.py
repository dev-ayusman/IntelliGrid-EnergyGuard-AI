# Complete Pipeline with Model Saving
# Enhanced version that saves trained models to disk

import pandas as pd
import numpy as np
from pathlib import Path

# Import ML libraries
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import MinMaxScaler
import joblib

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

from model_manager import ModelManager

print("=" * 80)
print("ENERGY ANOMALY DETECTION - COMPLETE IMPLEMENTATION (WITH MODEL SAVING)")
print("=" * 80)

# Initialize model manager
model_manager = ModelManager('models')

# STEP 1: Load Data
print("\n[1/8] Loading dataset...")
data_files = list(Path('data/raw').glob('*.csv'))
if data_files:
    df = pd.read_csv(data_files[0])
    print(f"Loaded {len(df):,} records with {df.shape[1]} features")
else:
    print("No data files found. Please download the dataset first.")
    exit()

# STEP 2: Data Cleaning
print("\n[2/8] Cleaning data...")
df = df.fillna(method='ffill').fillna(method='bfill')
df = df.drop_duplicates()

timestamp_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
if timestamp_cols:
    df[timestamp_cols[0]] = pd.to_datetime(df[timestamp_cols[0]], errors='coerce')
    df = df.sort_values(timestamp_cols[0])

print(f"Cleaned dataset: {len(df):,} records")

# STEP 3: Feature Engineering
print("\n[3/8] Engineering features...")

energy_cols = [col for col in df.columns if 'energy' in col.lower() or 'electric' in col.lower()]
if energy_cols:
    energy_col = energy_cols[0]
else:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    energy_col = numeric_cols[0] if len(numeric_cols) > 0 else df.columns[1]

df['energy_rolling_mean_7d'] = df[energy_col].rolling(window=168, min_periods=1).mean()
df['energy_rolling_std_7d'] = df[energy_col].rolling(window=168, min_periods=1).std()
df['energy_deviation'] = (df[energy_col] - df['energy_rolling_mean_7d']) / (df['energy_rolling_std_7d'] + 1e-5)
df = df.fillna(0)

print(f"Created features. Total features: {df.shape[1]}")

# STEP 4: Prepare for Modeling
print("\n[4/8] Preparing for modeling...")

feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [col for col in feature_cols if 'rolling' not in col and 'deviation' not in col][:4]

X = df[feature_cols].fillna(0)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler
model_manager.save_model(scaler, 'feature_scaler', {
    'features': feature_cols,
    'n_samples': len(X)
})

print(f"Selected {len(feature_cols)} features for modeling")
print(f"   Features: {feature_cols}")

# STEP 5: Train Anomaly Detection Models
print("\n[5/8] Training and saving anomaly detection models...")

# Model 1: Isolation Forest  
print("   Training Isolation Forest...")
iso_forest = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
anomaly_iso = iso_forest.fit_predict(X_scaled)
model_manager.save_model(iso_forest, 'isolation_forest', {
    'algorithm': 'Isolation Forest',
    'contamination': 0.05,
    'n_samples': len(X)
})

# Model 2: Local Outlier Factor
print("   Training Local Outlier Factor...")
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
anomaly_lof = lof.fit_predict(X_scaled)
model_manager.save_model(lof, 'local_outlier_factor', {
    'algorithm': 'Local Outlier Factor',
    'n_neighbors': 20,
    'contamination': 0.05
})

# Model 3: Robust Covariance
print("   Training Elliptic Envelope...")
robust_cov = EllipticEnvelope(contamination=0.05, random_state=42)
anomaly_maha = robust_cov.fit_predict(X_scaled)
model_manager.save_model(robust_cov, 'elliptic_envelope', {
    'algorithm': 'Elliptic Envelope',
    'contamination': 0.05
})

# Ensemble voting
df['anomaly_votes'] = (anomaly_iso == -1).astype(int) + \
                       (anomaly_lof == -1).astype(int) + \
                       (anomaly_maha == -1).astype(int)
df['is_anomaly'] = (df['anomaly_votes'] >= 2).astype(int)

anomaly_count = df['is_anomaly'].sum()
anomaly_percent = (anomaly_count / len(df)) * 100

print(f"All 3 models trained and saved")
print(f"   Detected {anomaly_count:,} anomalies ({anomaly_percent:.2f}%)")

# STEP 6: List saved models
print("\n[6/8] Listing saved models...")
model_manager.list_models()

# STEP 7: Visualize Results
print("\n[7/8] Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Energy distribution
axes[0, 0].hist(df[df['is_anomaly']==0][energy_col], bins=50, alpha=0.7, label='Normal', color='green')
axes[0, 0].hist(df[df['is_anomaly']==1][energy_col], bins=50, alpha=0.7, label='Anomaly', color='red')
axes[0, 0].set_title('Energy Distribution: Normal vs Anomaly')
axes[0, 0].set_xlabel('Energy (kWh)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Time series
sample_size = min(500, len(df))
axes[0, 1].plot(range(sample_size), df[energy_col].iloc[:sample_size], linewidth=1, alpha=0.7)
anomaly_indices = df[df['is_anomaly']==1].index[:sample_size]
axes[0, 1].scatter(anomaly_indices, df.loc[anomaly_indices, energy_col], 
                   color='red', s=50, label='Anomaly', zorder=5)
axes[0, 1].set_title(f'Energy with Anomalies (First {sample_size} records)')
axes[0, 1].set_xlabel('Record Index')
axes[0, 1].set_ylabel('Energy (kWh)')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Model agreement
vote_counts = df['anomaly_votes'].value_counts().sort_index()
axes[1, 0].bar(vote_counts.index, vote_counts.values, color=['green', 'yellow', 'orange', 'red'])
axes[1, 0].set_title('Ensemble Model Agreement')
axes[1, 0].set_xlabel('Number of Models Flagging as Anomaly')
axes[1, 0].set_ylabel('Count')
axes[1, 0].grid(axis='y', alpha=0.3)

# Plot 4: Feature correlation
if len(feature_cols) > 0:
    feature_importance = abs(df[feature_cols].corrwith(df['is_anomaly'])).sort_values(ascending=False)
    feature_importance.plot(kind='barh', ax=axes[1, 1], color='steelblue')
    axes[1, 1].set_title('Feature Correlation with Anomalies')
    axes[1, 1].set_xlabel('Absolute Correlation')
    axes[1, 1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('results/anomaly_detection_results.png', dpi=300, bbox_inches='tight')
print("Visualizations saved to: results/anomaly_detection_results.png")
plt.show()

# STEP 8: Business Insights
print("\n[8/8] Generating business insights...")

avg_kwh_cost = 0.12
total_anomaly_energy = df[df['is_anomaly']==1][energy_col].sum()
anomaly_cost = total_anomaly_energy * avg_kwh_cost

print(f"\nKEY FINDINGS:")
print(f"   • Total anomalies detected: {anomaly_count:,} ({anomaly_percent:.2f}%)")
print(f"   • Anomalous energy consumption: {total_anomaly_energy:,.0f} kWh")
print(f"   • Estimated cost impact: ${anomaly_cost:,.2f}")
print(f"   • Average anomaly energy: {df[df['is_anomaly']==1][energy_col].mean():.2f} kWh")
print(f"   • Average normal energy: {df[df['is_anomaly']==0][energy_col].mean():.2f} kWh")

# Save results
df[['is_anomaly', 'anomaly_votes'] + feature_cols + [energy_col]].to_csv(
    'results/anomaly_results.csv', index=False
)
print(f"\nResults saved to: results/anomaly_results.csv")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
print(f"\nTrained models saved in: models/")
print(f"   • isolation_forest.pkl")
print(f"   • local_outlier_factor.pkl")
print(f"   • elliptic_envelope.pkl")
print(f"   • feature_scaler.pkl")
print(f"\nTo use saved models:")
print(f"   from src.model_manager import ModelManager")
print(f"   manager = ModelManager()")
print(f"   model = manager.load_model('isolation_forest')")
print(f"   predictions = model.predict(new_data)")
