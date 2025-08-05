import json
import pickle
import numpy as np
from pathlib import Path

def load_from_json(filepath):
    """Load a JSON file and return a Python object."""
    filepath = Path(filepath)
    with filepath.open("r", encoding="utf-8") as f:
        return json.load(f)

def write_to_json(data, filepath):
    """Write a Python object to JSON file."""
    filepath = Path(filepath)
    with filepath.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def save_as_json(data, filepath):
    """Alias for write_to_json (for code compatibility)."""
    write_to_json(data, filepath)

def write_to_pickle(data, filepath):
    """Save Python object using pickle (for intermediate results)."""
    with open(filepath, "wb") as f:
        pickle.dump(data, f)

def normalize_dict(d):
    """Normalize values in dictionary so they sum to 1."""
    total = sum(d.values())
    if total <= 0:
        return {k: 0.0 for k in d}
    return {k: v / total for k, v in d.items()}

def top_k_accuracy(predictions, true_id, k):
    """
    Check whether true_id is in the top-k predictions.
    predictions: dict of {id: score}
    """
    if not predictions or true_id not in predictions:
        return 0
    topk = sorted(predictions, key=predictions.get, reverse=True)[:k]
    return int(true_id in topk)

def average_top_k_scores(results_list):
    """
    Compute mean success at each top-k position.
    Input: list of arrays [N x k]
    Output: array of length k with average score at each position
    """
    return np.mean(results_list, axis=0) if results_list else np.zeros(20)
