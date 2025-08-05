import json
from constants import VIGNETTES_FILE, DATA_PATH
from pathlib import Path

# Mapping from categorical severity to numeric
SEVERITY_MAPPING = {
    "NOT_PRESENT": 0.0,
    "MILD": 0.3,
    "MODERATE": 0.6,
    "PRESENT": 1.0,
    "SEVERE": 1.2,  # if used
}

def load_vignettes(path: Path = DATA_PATH / VIGNETTES_FILE):
    """Load raw vignettes from JSON file."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def convert_symptom_severity(vignettes: dict) -> dict:
    """Convert symbolic severity into numeric for all vignettes."""
    for vignette_id, vignette_data in vignettes.items():
        for symptom in vignette_data.get("card", {}).get("symptoms", []):
            severity_str = symptom.get("severity", "PRESENT")  # fallback to PRESENT
            numeric_value = SEVERITY_MAPPING.get(severity_str.strip().upper(), 1.0)
            symptom["severity_numeric"] = numeric_value  # add new key
    return vignettes

def extract_risk_factor_ids(vignettes: dict) -> set:
    """Extract all unique risk factor IDs present in the vignettes."""
    rf_ids = set()
    for vignette in vignettes.values():
        for rf in vignette["card"].get("risk_factors", []):
            rf_ids.add(rf["concept"]["id"])
    return rf_ids

def preprocess_vignettes(path: Path = DATA_PATH / VIGNETTES_FILE):
    """Full preprocessing pipeline: load, convert, return transformed vignettes."""
    vignettes = load_vignettes(path)
    vignettes = convert_symptom_severity(vignettes)
    return vignettes

if __name__ == "__main__":
    v = preprocess_vignettes()
    import json
    print(
        "Sample vignette (after preprocessing):\n"
        + json.dumps(next(iter(v.items())), indent=2)
    )