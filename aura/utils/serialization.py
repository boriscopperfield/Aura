"""
Serialization utilities for AURA system.
"""
import json
import base64
import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Type

import numpy as np
from pydantic import BaseModel


class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for AURA types."""
    
    def default(self, obj: Any) -> Any:
        """Encode custom types.
        
        Args:
            obj: Object to encode
            
        Returns:
            JSON-serializable object
        """
        # Handle datetime objects
        if isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()
        
        # Handle Enum objects
        if isinstance(obj, Enum):
            return obj.value
        
        # Handle Path objects
        if isinstance(obj, Path):
            return str(obj)
        
        # Handle numpy arrays
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        
        # Handle bytes
        if isinstance(obj, bytes):
            return {
                "__type__": "bytes",
                "data": base64.b64encode(obj).decode("utf-8")
            }
        
        # Handle Pydantic models
        if isinstance(obj, BaseModel):
            return obj.dict()
        
        # Handle sets
        if isinstance(obj, set):
            return list(obj)
        
        # Let the base class handle it
        return super().default(obj)


def json_dumps(obj: Any, **kwargs: Any) -> str:
    """Serialize object to JSON string.
    
    Args:
        obj: Object to serialize
        **kwargs: Additional arguments for json.dumps
        
    Returns:
        JSON string
    """
    return json.dumps(obj, cls=JSONEncoder, **kwargs)


def json_loads(s: str, **kwargs: Any) -> Any:
    """Deserialize JSON string to object.
    
    Args:
        s: JSON string
        **kwargs: Additional arguments for json.loads
        
    Returns:
        Deserialized object
    """
    def object_hook(d: Dict[str, Any]) -> Any:
        """Custom object hook for deserialization.
        
        Args:
            d: Dictionary to deserialize
            
        Returns:
            Deserialized object
        """
        # Handle bytes
        if "__type__" in d and d["__type__"] == "bytes" and "data" in d:
            return base64.b64decode(d["data"])
        
        return d
    
    return json.loads(s, object_hook=object_hook, **kwargs)


def to_serializable(obj: Any) -> Any:
    """Convert object to JSON-serializable form.
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON-serializable object
    """
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return obj.to_dict()
    
    if isinstance(obj, (list, tuple)):
        return [to_serializable(item) for item in obj]
    
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    
    if isinstance(obj, Enum):
        return obj.value
    
    if isinstance(obj, Path):
        return str(obj)
    
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    
    if isinstance(obj, bytes):
        return base64.b64encode(obj).decode("utf-8")
    
    if isinstance(obj, BaseModel):
        return obj.dict()
    
    if isinstance(obj, set):
        return list(obj)
    
    return obj


def from_serializable(data: Any, target_type: Optional[Type] = None) -> Any:
    """Convert serialized data back to original type.
    
    Args:
        data: Serialized data
        target_type: Optional target type
        
    Returns:
        Deserialized object
    """
    if target_type is not None:
        # Handle Pydantic models
        if issubclass(target_type, BaseModel):
            return target_type.parse_obj(data)
        
        # Handle Enum types
        if issubclass(target_type, Enum):
            return target_type(data)
        
        # Handle Path objects
        if target_type == Path:
            return Path(data)
        
        # Handle datetime objects
        if target_type == datetime.datetime:
            return datetime.datetime.fromisoformat(data)
        
        if target_type == datetime.date:
            return datetime.date.fromisoformat(data)
        
        # Handle basic types
        if target_type in (str, int, float, bool):
            return target_type(data)
        
        # Handle lists
        if target_type == list and isinstance(data, list):
            return data
        
        # Handle dictionaries
        if target_type == dict and isinstance(data, dict):
            return data
    
    # No target type specified, try to infer
    if isinstance(data, dict):
        return {k: from_serializable(v) for k, v in data.items()}
    
    if isinstance(data, list):
        return [from_serializable(item) for item in data]
    
    return data