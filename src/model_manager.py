"""
Model Saver/Loader Module

This module provides utilities to save and load trained models.
"""

import pickle
import joblib
from pathlib import Path
import json
from datetime import datetime


class ModelManager:
    """Manage saving and loading of trained models"""
    
    def __init__(self, models_dir='models'):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
    
    def save_model(self, model, model_name, metadata=None):
        """
        Save a trained model to disk
        
        Args:
            model: The trained model object
            model_name (str): Name for the model file
            metadata (dict): Optional metadata about the model
        """
        # Save model
        model_path = self.models_dir / f"{model_name}.pkl"
        joblib.dump(model, model_path)
        print(f"Model saved: {model_path}")
        
        # Save metadata if provided
        if metadata:
            metadata['saved_at'] = datetime.now().isoformat()
            metadata_path = self.models_dir / f"{model_name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"Metadata saved: {metadata_path}")
    
    def load_model(self, model_name):
        """
        Load a trained model from disk
        
        Args:
            model_name (str): Name of the model file (without .pkl)
            
        Returns:
            The loaded model object
        """
        model_path = self.models_dir / f"{model_name}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model = joblib.load(model_path)
        print(f"Model loaded: {model_path}")
        
        # Load metadata if available
        metadata_path = self.models_dir / f"{model_name}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"   Trained: {metadata.get('saved_at', 'Unknown')}")
        
        return model
    
    def list_models(self):
        """List all saved models"""
        models = list(self.models_dir.glob('*.pkl'))
        if models:
            print(f"\n📦 Found {len(models)} saved model(s):")
            for model_path in models:
                size_mb = model_path.stat().st_size / (1024 * 1024)
                print(f"   • {model_path.stem} ({size_mb:.2f} MB)")
        else:
            print("\n📦 No saved models found")
        return [m.stem for m in models]


# Example usage
if __name__ == "__main__":
    # Create manager
    manager = ModelManager()
    
    # List existing models
    manager.list_models()
    
    print("\nTo use this in your pipeline:")
    print("  from src.model_manager import ModelManager")
    print("  manager = ModelManager()")
    print("  manager.save_model(iso_forest, 'isolation_forest')")
    print("  loaded_model = manager.load_model('isolation_forest')")
