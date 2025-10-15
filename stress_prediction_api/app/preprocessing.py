from typing import Dict, Any
import numpy as np

def preprocess_input(data: Dict[str, Any]):
    """Convert incoming dict of features into a 2D numpy array.

    If conversion fails, returns the original dict to allow fallback behavior in the service.
    """
    if not isinstance(data, dict):
        raise ValueError("Input must be a dict of features")

    values = []
    for k, v in data.items():
        try:
            values.append(float(v))
        except Exception:
            if isinstance(v, bool):
                values.append(1.0 if v else 0.0)
            else:
                # unknown categorical -> 0.0 placeholder
                values.append(0.0)

    if not values:
        return data

    return np.array([values], dtype=float)
