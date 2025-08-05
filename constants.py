#from pathlib import Path
# Define base directories relative to this file's location
#DATA_DIR = Path(__file__).resolve().parent.parent / "data"
#RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Input data files
#NETWORKS_FILE = DATA_DIR / "example_networks.json"
#VIGNETTES_FILE = DATA_DIR / "vignettes.json"

# Output file for final posterior + CF scores
#RESULTS_FILE = RESULTS_DIR / "experimental_results.json"

# Optional pickled intermediate files (can be used for debugging or saving intermediate state)
#RESULTS_OBS_FILE = RESULTS_DIR / "results_obs.p"
#RESULTS_CF_DISABLEMENT_FILE = RESULTS_DIR / "results_counter_diss.p"
#RESULTS_CF_SUFFICIENCY_FILE = RESULTS_DIR / "results_counter_suff.p"


from pathlib import Path

DATA_PATH = Path(__file__).parent / "data"

NETWORKS_FILE = "example_networks.json"
VIGNETTES_FILE = "vignettes.json"
RESULTS_FILE = "experimental_results.json"

RESULTS_OBS_FILE = "results_obs.p"
RESULTS_CF_DISABLEMENT_FILE = "results_counter_diss.p"
RESULTS_CF_SUFFICIENCY_FILE = "results_counter_suff.p"
