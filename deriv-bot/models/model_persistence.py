"""
Model Persistence
==================
Save and load model snapshots to disk.
Uses pickle for River models (they're designed for this).
"""

import pickle
import time
from pathlib import Path
from typing import Optional

from config import MODEL_DIR
from utils.logger import setup_logger

logger = setup_logger("models.persistence")


class ModelPersistence:
    """
    Save/load model snapshots.
    
    Snapshots include:
    - The River model weights
    - The feature scaler
    - Model stats
    - Calibration data
    - Replay buffer (last N samples)
    """
    
    def __init__(self, model_dir: str = str(MODEL_DIR)):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    def save_snapshot(self, model, snapshot_name: Optional[str] = None):
        """
        Save current model state to disk.
        
        Args:
            model: OverUnderModel instance
            snapshot_name: Optional custom name. Default: timestamp-based.
        """
        if snapshot_name is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            snapshot_name = f"model_v{model.stats.model_version}_{timestamp}"
        
        filepath = self.model_dir / f"{snapshot_name}.pkl"
        
        snapshot_data = {
            "model": model.model,
            "scaler": model.scaler,
            "stats": model.stats,
            "replay_buffer": list(model.replay_buffer),
            "confidence_bins": dict(model._confidence_bins),
            "saved_at": time.time(),
        }
        
        try:
            with open(filepath, "wb") as f:
                pickle.dump(snapshot_data, f)
            logger.info(f"Model saved: {filepath.name}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return None
    
    def load_snapshot(self, model, filepath: str) -> bool:
        """
        Load a model snapshot from disk into the given model instance.
        
        Args:
            model: OverUnderModel instance to restore into
            filepath: Path to the .pkl snapshot file
        
        Returns:
            True if loaded successfully
        """
        path = Path(filepath)
        if not path.exists():
            logger.error(f"Snapshot not found: {filepath}")
            return False
        
        try:
            with open(path, "rb") as f:
                snapshot_data = pickle.load(f)
            
            model.model = snapshot_data["model"]
            model.scaler = snapshot_data["scaler"]
            model.stats = snapshot_data["stats"]
            model.replay_buffer = collections.deque(
                snapshot_data["replay_buffer"],
                maxlen=model.replay_buffer.maxlen,
            )
            model._confidence_bins = collections.defaultdict(
                lambda: {"correct": 0, "total": 0},
                snapshot_data.get("confidence_bins", {}),
            )
            
            logger.info(f"Model loaded: {path.name} "
                         f"(v{model.stats.model_version}, "
                         f"{model.stats.total_updates} updates)")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def list_snapshots(self) -> list:
        """List all saved model snapshots."""
        snapshots = []
        for f in sorted(self.model_dir.glob("*.pkl"), reverse=True):
            snapshots.append({
                "name": f.name,
                "path": str(f),
                "size_kb": f.stat().st_size / 1024,
                "modified": time.strftime("%Y-%m-%d %H:%M:%S",
                                           time.localtime(f.stat().st_mtime)),
            })
        return snapshots
    
    def cleanup_old_snapshots(self, keep_last: int = 5):
        """Remove old snapshots, keeping only the most recent N."""
        snapshots = sorted(self.model_dir.glob("*.pkl"))
        if len(snapshots) > keep_last:
            for old_file in snapshots[:-keep_last]:
                old_file.unlink()
                logger.info(f"Deleted old snapshot: {old_file.name}")


import collections
