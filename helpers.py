import copy
import numpy as np
import networkx as nx
from functools import lru_cache
from scipy.stats import binomtest

from constants import NETWORKS_FILE
from utils import load_from_json

# ------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------
THRESH = 0.3  # threshold for deciding symptom “presence” in counterfactuals


# ------------------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_networks(datapath, filename=NETWORKS_FILE):
    """Load and cache the disease network JSON."""
    from pathlib import Path
    datapath = Path(datapath)
    return load_from_json(datapath / filename)


# ------------------------------------------------------------------------
# NETWORK STRUCTURE HELPERS
# ------------------------------------------------------------------------
def build_disease_graph(network: dict):
    """Create a directed graph (DAG) from the network dict."""
    G = nx.DiGraph()
    for node_id, node in network.items():
        G.add_node(node_id, **node)
        for parent in node.get("parents", []):
            G.add_edge(parent, node_id)
    return G

def get_symptom_nodes(network: dict):
    """Return all node IDs labeled as 'Symptom'."""
    return [nid for nid, data in network.items() if data.get("label") == "Symptom"]

def get_risk_nodes(network: dict):
    """Return all node IDs labeled as 'Risk'."""
    return [nid for nid, data in network.items() if data.get("label") == "Risk"]


# ------------------------------------------------------------------------
# BAYESIAN INFERENCE UTILITIES
# ------------------------------------------------------------------------
def noisy_or(parents, network, input_values, epsilon=1e-9):
    """
    Compute Noisy-OR P(node=1 | parents, evidence).
    - parents: list of parent node IDs
    - network: dict of node definitions (with 'cpt')
    - input_values: {node_id: value ∈ [0,1.2]} from evidence
    """
    probs = []
    for pid in parents:
        p_val = input_values.get(pid, 0.0)
        # link_strength = 1 − leak_prob = 1 − CPT[parent=0]
        leak = network[pid].get("cpt", [1.0, 0.0])[0]
        link_strength = 1.0 - leak
        # generalize to real‐valued p_val by exponentiation
        adjusted = link_strength * p_val  # Scale link by severity
        adjusted = min(adjusted, 1.0)     # clamp to [0, 1] to avoid overdrive
        probs.append(1.0 - adjusted)

    if not probs:
        return 0.0

    # Combine with standard Noisy-OR product rule
    return 1.0 - np.prod([1.0 - p + epsilon for p in probs])


# ------------------------------------------------------------------------
# COUNTERFACTUAL INFERENCE UTILITIES
# ------------------------------------------------------------------------
def make_twin_network(original_network: dict, disable=None, force=None):
    """
    Return a *copy* of original_network where one disease node’s CPT
    is overridden to simulate an intervention:

      - disable=disease_id → CPT = [1.0, 0.0] (always False)
      - force=disease_id   → CPT = [0.0, 1.0] (always True)

    This replaces the OG “append _cf” approach so that posterior_inference
    on the twin_net actually differs from the original.
    """
    twin = copy.deepcopy(original_network)

    # Apply the intervention on the specified node
    if disable and disable in twin:
        twin[disable]["cpt"] = [1.0, 0.0]
    if force and force in twin:
        twin[force]["cpt"] = [0.0, 1.0]

    return twin


def count_disabled_symptoms(network: dict,
                            symptom_nodes: list,
                            original_values: dict,
                            counterfactual_values: dict,
                            recovery=False):
    """
    Compute disablement or sufficiency using actual difference in symptom probability.
    - For disablement: sum how much symptom probability *drops*
    - For sufficiency: sum how much symptom probability *rises*
    """
    total = 0.0
    for sid in symptom_nodes:
        orig = original_values.get(sid, 0.0)
        cf   = counterfactual_values.get(sid, 0.0)

        delta = orig - cf if not recovery else cf - orig
        if delta > 0:
            total += delta

    return total



# ------------------------------------------------------------------------
# DOCTOR DIFFERENTIAL EVALUATION (unchanged)
# ------------------------------------------------------------------------
def get_doctor_differential(li):
    return [val["concept"]["id"] for val in li if val["concept"]["id"] is not None]

def produce_differentials(card):
    return {
        val["user"]["id"]: get_doctor_differential(val["doctor_diseases"])
        for val in card.get("outcomes", [])
        if "doctor_diseases" in val
    }

def doctor_top_ns(card, true_disease):
    return [
        [uid, len(diff), 1 if true_disease in diff else 0]
        for uid, diff in produce_differentials(card).items()
    ]


# ------------------------------------------------------------------------
# STATISTICS / DEBUGGING (unchanged)
# ------------------------------------------------------------------------
def mean_list(li):
    return "none" if not li else np.mean(li)

def bintest(x, y, conf_thresh):
    if conf_thresh == 0:
        return sum(x) >= sum(y)
    p_val = binomtest(sum(x), n=len(x), p=(sum(y) / len(y)), alternative="greater").pvalue
    return p_val < conf_thresh
