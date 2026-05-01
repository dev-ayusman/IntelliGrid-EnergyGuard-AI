# Energy Anomaly Detection - Quick Start Script
# This script provides a complete end-to-end implementation

import pandas as pd
import numpy as np
from pathlib import Path

# Import ML libraries
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 80)
print("ENERGY ANOMALY DETECTION - COMPLETE IMPLEMENTATION")
print("=" * 80)

# STEP 1: Load Data
print("\n[1/7] Loading dataset...")
# Update this path based on your data location
data_files = list(Path('data/raw').glob('*.csv'))
if data_files:
    df = pd.read_csv(data_files[0])
    print(f"Loaded {len(df):,} records with {df.shape[1]} features")
else:
    print("No data files found. Please download the dataset first.")
    exit()

# STEP 2: Data Cleaning
print("\n[2/7] Cleaning data...")
# Handle missing values
df = df.fillna(method='ffill').fillna(method='bfill')

# Remove duplicates
df =df.drop_duplicates()

# Convert timestamp if exists
timestamp_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
if timestamp_cols:
    df[timestamp_cols[0]] = pd.to_datetime(df[timestamp_cols[0]], errors='coerce')
    df = df.sort_values(timestamp_cols[0])

print(f"Cleaned dataset: {len(df):,} records")

# STEP 3: Feature Engineering
print("\n[3/7] Engineering features...")

# Select energy column
energy_cols = [col for col in df.columns if 'energy' in col.lower() or 'electric' in col.lower()]
if energy_cols:
    energy_col = energy_cols[0]
else:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    energy_col = numeric_cols[0] if len(numeric_cols) > 0 else df.columns[1]

# Create rolling statistics
df['energy_rolling_mean_7d'] = df[energy_col].rolling(window=168, min_periods=1).mean()
df['energy_rolling_std_7d'] = df[energy_col].rolling(window=168, min_periods=1).std()

# Deviation from baseline
df['energy_deviation'] = (df[energy_col] - df['energy_rolling_mean_7d']) / (df['energy_rolling_std_7d'] + 1e-5)

# Handle any remaining NaN
df = df.fillna(0)

print(f"Created features. Total features: {df.shape[1]}")

# STEP 4: Prepare for Modeling
print("\n[4/7] Preparing for modeling...")

# Select features for modeling
feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# Remove derived columns from features
feature_cols = [col for col in feature_cols if 'rolling' not in col and 'deviation' not in col][:4]

X = df[feature_cols].fillna(0)

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

print(f"Selected {len(feature_cols)} features for modeling")
print(f"   Features: {feature_cols}")

# STEP 5: Train Anomaly Detection Models
print("\n[5/7] Training anomaly detection models...")

# Model 1: Isolation Forest  
iso_forest = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
anomaly_iso = iso_forest.fit_predict(X_scaled)

# Model 2: Local Outlier Factor
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
anomaly_lof = lof.fit_predict(X_scaled)

# Model 3: Robust Covariance
robust_cov = EllipticEnvelope(contamination=0.05, random_state=42)
anomaly_maha = robust_cov.fit_predict(X_scaled)

# Ensemble voting (2/3 vote required)
df['anomaly_votes'] = (anomaly_iso == -1).astype(int) + \
                       (anomaly_lof == -1).astype(int) + \
                       (anomaly_maha == -1).astype(int)
df['is_anomaly'] = (df['anomaly_votes'] >= 2).astype(int)

anomaly_count = df['is_anomaly'].sum()
anomaly_percent = (anomaly_count / len(df)) * 100

print(f"Models trained successfully")
print(f"   Detected {anomaly_count:,} anomalies ({anomaly_percent:.2f}%)")

# STEP 6: Visualize Results
print("\n[6/7] Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Energy distribution with anomalies
axes[0, 0].hist(df[df['is_anomaly']==0][energy_col], bins=50, alpha=0.7, label='Normal', color='green')
axes[0, 0].hist(df[df['is_anomaly']==1][energy_col], bins=50, alpha=0.7, label='Anomaly', color='red')
axes[0, 0].set_title('Energy Distribution: Normal vs Anomaly')
axes[0, 0].set_xlabel('Energy (kWh)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Time series with anomalies (first 500 points)
sample_size = min(500, len(df))
axes[0, 1].plot(range(sample_size), df[energy_col].iloc[:sample_size], linewidth=1, alpha=0.7, label='Energy')
anomaly_indices = df[df['is_anomaly']==1].index[:sample_size]
axes[0, 1].scatter(anomaly_indices, df.loc[anomaly_indices, energy_col], 
                   color='red', s=50, label='Anomaly', zorder=5)
axes[0, 1].set_title(f'Energy Consumption with Anomalies (First {sample_size} records)')
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

# Plot 4: Anomaly rate by feature
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

# STEP 7: Business Insights
print("\n[7/7] Generating business insights...")

# Calculate cost impact (assuming $0.12 per kWh)
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
print(f"\nNext steps:")
print(f"  1. Review the visualizations in results/anomaly_detection_results.png")
print(f"  2. Examine detailed results in results/anomaly_results.csv")
print(f"  3. Investigate the anomalous records to understand patterns")
print(f"  4. Consider implementing real-time monitoring based on these models")
