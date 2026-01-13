"""
Model Store - Save and load trained ML models
Uses joblib for efficient serialization
"""
import os
import joblib
from datetime import datetime
from typing import Dict, Optional, Any, List
from dataclasses import dataclass
from pathlib import Path
import json

from app.config import get_settings


@dataclass
class ModelMetadata:
    """Metadata for stored model"""
    model_name: str
    symbol: str
    trained_at: datetime
    metrics: Dict[str, float]
    feature_count: int
    train_samples: int
    version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "symbol": self.symbol,
            "trained_at": self.trained_at.isoformat(),
            "metrics": self.metrics,
            "feature_count": self.feature_count,
            "train_samples": self.train_samples,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ModelMetadata":
        return cls(
            model_name=data["model_name"],
            symbol=data["symbol"],
            trained_at=datetime.fromisoformat(data["trained_at"]),
            metrics=data["metrics"],
            feature_count=data["feature_count"],
            train_samples=data["train_samples"],
            version=data.get("version", "1.0"),
        )


class ModelStore:
    """
    Persistent storage for trained ML models

    Structure:
    models/
    ├── RELIANCE.NS/
    │   ├── ensemble/
    │   │   ├── model.joblib
    │   │   └── metadata.json
    │   ├── xgboost/
    │   │   ├── model.joblib
    │   │   └── metadata.json
    │   └── ...
    └── TCS.NS/
        └── ...
    """

    def __init__(self, base_path: Optional[str] = None):
        settings = get_settings()
        self.base_path = Path(base_path or settings.ML_MODEL_PATH)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_model_path(self, symbol: str, model_name: str) -> Path:
        """Get path for model files"""
        # Clean symbol for filesystem
        clean_symbol = symbol.replace(".", "_")
        return self.base_path / clean_symbol / model_name

    def save_model(
        self,
        model: Any,
        symbol: str,
        model_name: str,
        metrics: Dict[str, float],
        feature_count: int,
        train_samples: int
    ) -> str:
        """
        Save a trained model

        Args:
            model: Trained model object
            symbol: Stock symbol
            model_name: Name of the model
            metrics: Training metrics
            feature_count: Number of features
            train_samples: Number of training samples

        Returns:
            Path to saved model
        """
        model_path = self._get_model_path(symbol, model_name)
        model_path.mkdir(parents=True, exist_ok=True)

        # Save model
        model_file = model_path / "model.joblib"
        joblib.dump(model, model_file)

        # Save metadata
        metadata = ModelMetadata(
            model_name=model_name,
            symbol=symbol,
            trained_at=datetime.now(),
            metrics=metrics,
            feature_count=feature_count,
            train_samples=train_samples,
        )

        metadata_file = model_path / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)

        return str(model_file)

    def load_model(
        self,
        symbol: str,
        model_name: str
    ) -> Optional[Any]:
        """
        Load a saved model

        Args:
            symbol: Stock symbol
            model_name: Name of the model

        Returns:
            Loaded model or None if not found
        """
        model_path = self._get_model_path(symbol, model_name)
        model_file = model_path / "model.joblib"

        if not model_file.exists():
            return None

        try:
            return joblib.load(model_file)
        except Exception as e:
            print(f"Error loading model {model_name} for {symbol}: {e}")
            return None

    def get_metadata(
        self,
        symbol: str,
        model_name: str
    ) -> Optional[ModelMetadata]:
        """
        Get model metadata

        Args:
            symbol: Stock symbol
            model_name: Name of the model

        Returns:
            ModelMetadata or None
        """
        model_path = self._get_model_path(symbol, model_name)
        metadata_file = model_path / "metadata.json"

        if not metadata_file.exists():
            return None

        try:
            with open(metadata_file, 'r') as f:
                data = json.load(f)
            return ModelMetadata.from_dict(data)
        except Exception as e:
            print(f"Error loading metadata for {model_name}: {e}")
            return None

    def model_exists(self, symbol: str, model_name: str) -> bool:
        """Check if a model exists"""
        model_path = self._get_model_path(symbol, model_name)
        return (model_path / "model.joblib").exists()

    def is_model_stale(
        self,
        symbol: str,
        model_name: str,
        max_age_hours: int = 168  # 1 week
    ) -> bool:
        """
        Check if model needs retraining

        Args:
            symbol: Stock symbol
            model_name: Name of the model
            max_age_hours: Maximum age in hours

        Returns:
            True if model is stale or doesn't exist
        """
        metadata = self.get_metadata(symbol, model_name)
        if metadata is None:
            return True

        age = (datetime.now() - metadata.trained_at).total_seconds() / 3600
        return age > max_age_hours

    def list_models(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all saved models

        Args:
            symbol: Filter by symbol (optional)

        Returns:
            List of model info dictionaries
        """
        models = []

        if symbol:
            clean_symbol = symbol.replace(".", "_")
            symbol_path = self.base_path / clean_symbol
            if symbol_path.exists():
                for model_dir in symbol_path.iterdir():
                    if model_dir.is_dir():
                        metadata = self.get_metadata(symbol, model_dir.name)
                        if metadata:
                            models.append(metadata.to_dict())
        else:
            # List all symbols
            for symbol_dir in self.base_path.iterdir():
                if symbol_dir.is_dir():
                    symbol_name = symbol_dir.name.replace("_", ".")
                    for model_dir in symbol_dir.iterdir():
                        if model_dir.is_dir():
                            metadata = self.get_metadata(symbol_name, model_dir.name)
                            if metadata:
                                models.append(metadata.to_dict())

        return models

    def delete_model(self, symbol: str, model_name: str) -> bool:
        """
        Delete a saved model

        Args:
            symbol: Stock symbol
            model_name: Name of the model

        Returns:
            True if deleted, False if not found
        """
        model_path = self._get_model_path(symbol, model_name)

        if not model_path.exists():
            return False

        import shutil
        shutil.rmtree(model_path)
        return True

    def cleanup_old_models(self, max_age_days: int = 30) -> int:
        """
        Remove models older than specified age

        Args:
            max_age_days: Maximum age in days

        Returns:
            Number of models removed
        """
        removed = 0

        for symbol_dir in self.base_path.iterdir():
            if symbol_dir.is_dir():
                symbol_name = symbol_dir.name.replace("_", ".")
                for model_dir in symbol_dir.iterdir():
                    if model_dir.is_dir():
                        metadata = self.get_metadata(symbol_name, model_dir.name)
                        if metadata:
                            age_days = (datetime.now() - metadata.trained_at).days
                            if age_days > max_age_days:
                                if self.delete_model(symbol_name, model_dir.name):
                                    removed += 1

        return removed

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        total_models = 0
        total_size = 0
        symbols = set()

        for symbol_dir in self.base_path.iterdir():
            if symbol_dir.is_dir():
                symbols.add(symbol_dir.name)
                for model_dir in symbol_dir.iterdir():
                    if model_dir.is_dir():
                        total_models += 1
                        for file in model_dir.iterdir():
                            total_size += file.stat().st_size

        return {
            "total_models": total_models,
            "total_symbols": len(symbols),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "base_path": str(self.base_path),
        }


# Global instance
model_store = ModelStore()
