"""
Tests for the model_manager module
"""

import pytest
from pathlib import Path
import tempfile
import joblib
from src.model_manager import ModelManager


def test_model_manager_init():
    """Test ModelManager initialization"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ModelManager(models_dir=tmpdir)
        assert manager.models_dir == Path(tmpdir)
        assert manager.models_dir.exists()


def test_save_and_load_model():
    """Test saving and loading a model"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ModelManager(models_dir=tmpdir)
        
        # Create a simple model-like object
        model = {"type": "test_model", "data": [1, 2, 3]}
        
        # Save model
        manager.save_model(model, "test_model")
        
        # Load model
        loaded = manager.load_model("test_model")
        
        assert loaded == model


def test_list_models():
    """Test listing saved models"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ModelManager(models_dir=tmpdir)
        
        # Initially no models
        models = manager.list_models()
        assert len(models) == 0
        
        # Save a model
        model = {"test": True}
        manager.save_model(model, "model1")
        
        models = manager.list_models()
        assert len(models) == 1
        assert "model1" in models
