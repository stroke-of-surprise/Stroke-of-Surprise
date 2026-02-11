"""Logging and configuration utilities for training."""

import json
from pathlib import Path
from typing import Any, Dict, Union
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum


def _serialize_config_value(value: Any) -> Any:
    """Recursively serialize a config value to JSON-serializable format.
    
    Handles:
    - Path objects -> str
    - Enum objects -> value
    - dataclass objects -> dict
    - lists/tuples -> list (recursively)
    - dicts -> dict (recursively)
    - other types -> as-is
    """
    if value is None:
        return None
    
    # Handle Path objects
    if isinstance(value, Path):
        return str(value)
    
    # Handle Enum objects
    if isinstance(value, Enum):
        return value.value
    
    # Handle dataclass objects
    if is_dataclass(value):
        result = {}
        for field in fields(value):
            field_value = getattr(value, field.name)
            result[field.name] = _serialize_config_value(field_value)
        return result
    
    # Handle lists and tuples
    if isinstance(value, (list, tuple)):
        return [_serialize_config_value(item) for item in value]
    
    # Handle dicts
    if isinstance(value, dict):
        return {k: _serialize_config_value(v) for k, v in value.items()}
    
    # Handle basic types (int, float, str, bool) - return as-is
    return value


def save_config(
    cfg: Any,
    output_dir: Union[str, Path],
) -> None:
    """Save complete training configuration to JSON file.
    
    This function saves all configuration fields from the TrainConfig object,
    including all sub-configurations (data, optim, log, bezier, bspline, neural).
    
    Args:
        cfg: Training configuration object (TrainConfig dataclass instance)
        output_dir: Output directory where config.json will be saved
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert entire config to serializable dict
    config_dict = _serialize_config_value(cfg)
    
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=4)
    
    print(f"Config saved to {config_path}")

