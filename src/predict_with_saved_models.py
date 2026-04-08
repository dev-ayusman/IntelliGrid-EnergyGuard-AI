# Model Loading and Prediction Example

"""
This script demonstrates how to load saved models and make predictions
on new data without retraining.
"""

import pandas as pd
import numpy as np
from model_manager import ModelManager

print("=" * 80)
print("LOADING SAVED MODELS AND MAKING PREDICTIONS")
print("=" * 80)

# Initialize model manager
manager = ModelManager('models')

# List available models
print("\n[1/4] Checking available models...")
available_models = manager.list_models()

if not available_models:
    print("\n❌ No models found!")
    print("   Please run 'complete_pipeline_with_saving.py' first to train and save models.")
    exit()

# Load models
print("\n[2/4] Loading trained models...")
try:
    iso_forest = manager.load_model('isolation_forest')
    lof = manager.load_model('local_outlier_factor')
    elliptic = manager.load_model('elliptic_envelope')
    scaler = manager.load_model('feature_scaler')
    print("✅ All models loaded successfully!")
except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    print("   Run 'complete_pipeline_with_saving.py' first.")
    exit()

# Create sample new data for prediction
print("\n[3/4] Creating sample data for prediction...")
sample_data = pd.DataFrame({
    'energy_kwh': [1500, 2800, 1400, 900, 1600],  # 2nd and 4th might be anomalies
    'temperature': [22, 28, 21, 15, 23],
    'humidity': [60, 55, 65, 70, 58],
    'building_area': [5000, 5000, 5000, 5000, 5000]
})

print(f"✅ Created {len(sample_data)} sample records")
print(sample_data)

# Scale features
print("\n[4/4] Making predictions...")
X_scaled = scaler.transform(sample_data)

# Get predictions from each model
iso_pred = iso_forest.predict(X_scaled)
lof_pred = lof.predict(X_scaled)
elliptic_pred = elliptic.predict(X_scaled)

# Ensemble voting
votes = (iso_pred == -1).astype(int) + (lof_pred == -1).astype(int) + (elliptic_pred == -1).astype(int)
is_anomaly = (votes >= 2).astype(int)

# Create results dataframe
results = sample_data.copy()
results['iso_forest'] = ['Anomaly' if x == -1 else 'Normal' for x in iso_pred]
results['lof'] = ['Anomaly' if x == -1 else 'Normal' for x in lof_pred]
results['elliptic'] = ['Anomaly' if x == -1 else 'Normal' for x in elliptic_pred]
results['votes'] = votes
results['final_prediction'] = ['🚨 ANOMALY' if x == 1 else '✅ Normal' for x in is_anomaly]

print("\n📊 PREDICTION RESULTS:")
print("=" * 80)
print(results.to_string(index=False))

# Summary
anomaly_count = is_anomaly.sum()
print(f"\n📈 SUMMARY:")
print(f"   • Total predictions: {len(sample_data)}")
print(f"   • Anomalies detected: {anomaly_count}")
print(f"   • Normal readings: {len(sample_data) - anomaly_count}")

print("\n" + "=" * 80)
print("✅ PREDICTIONS COMPLETE!")
print("=" * 80)
print("\n💡 You can now use these models to predict on any new energy data!")
print("   Just load the models and call predict() on scaled features.")
