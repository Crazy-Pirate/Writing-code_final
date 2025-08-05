import copy
import numpy as np
import networkx as nx
from functools import lru_cache
from scipy.stats import binomtest

from constants import NETWORKS_FILE
from utils import load_from_json


# ------------------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_networks(datapath, filename=NETWORKS_FILE):
    """Load and cache the disease network."""
    # allow datapath as str or Path
    from pathlib import Path
    datapath = Path(datapath)
    return load_from_json(datapath / filename)


# ------------------------------------------------------------------------
# NETWORK STRUCTURE HELPERS
# ------------------------------------------------------------------------

def build_disease_graph(network: dict):
    """Create a directed graph (DAG) from the network."""
    G = nx.DiGraph()
    for node_id, node in network.items():
        G.add_node(node_id, **node)
        for parent in node.get("parents", []):
            G.add_edge(parent, node_id)
    return G


def get_symptom_nodes(network: dict):
    """Return nodes labeled as 'Symptom'."""
    return [node_id for node_id, data in network.items() if data.get("label") == "Symptom"]


def get_risk_nodes(network: dict):
    """Return nodes labeled as 'Risk'."""
    return [node_id for node_id, data in network.items() if data.get("label") == "Risk"]


# ------------------------------------------------------------------------
# BAYESIAN INFERENCE UTILITIES
# ------------------------------------------------------------------------

def noisy_or(parents, network, input_values, epsilon=1e-9):
    """
    Compute Noisy-OR probability for a node given its parents.
    input_values: dict of {node_id: value between 0 and 1}
    """
    probs = []
    for parent_id in parents:
        prob_true = input_values.get(parent_id, 0.0)
        link_strength = 1.0 - network[parent_id].get("cpt", [1.0, 0.0])[0]
        probs.append(1.0 - link_strength ** prob_true)
    if not probs:
        return 0.0
    # no parents → no activation
    return 1.0 - np.prod([1.0 - p + epsilon for p in probs])  # avoid underflow


# ------------------------------------------------------------------------
# COUNTERFACTUAL INFERENCE UTILITIES
# ------------------------------------------------------------------------

def make_twin_network(original_network: dict, disable=None, force=None):
    """
    Create a twin network for counterfactual reasoning.
    If `disable` is set, that disease is turned off in the twin (CPT = [1.0, 0.0]).
    If `force` is set, that disease is turned on in the twin (CPT = [0.0, 1.0]).
    """
    twin_net = copy.deepcopy(original_network)
    # deep copy to avoid mutating the original

    for node_id, node in original_network.items():
        twin_id = f"{node_id}_cf"
        twin_node = {
            "label": node.get("label", ""),
            "parents": [f"{p}_cf" for p in node.get("parents", [])],
            "cpt": copy.deepcopy(node.get("cpt", [1.0, 0.0])),
        }

        # Force intervention if needed
        if disable == node_id:
            twin_node["cpt"] = [1.0, 0.0]  # Disease always False
        elif force == node_id:
            twin_node["cpt"] = [0.0, 1.0]  # Disease always True

        twin_net[twin_id] = twin_node

    return twin_net



def count_disabled_symptoms(network: dict,
                            symptom_nodes: list,
                            original_values: dict,
                            counterfactual_values: dict,
                            recovery=False):
    """
    Now: weight each symptom change by its severity.
    - For disablement (recovery=False), if symptom S disappears (orig ≥ .5 → cf < .5),
      add orig‐severity to the total.
    - For sufficiency (recovery=True), if S appears (orig < .5 → cf ≥ .5),
      add cf‐severity to the total.
    """
    total_weight = 0.0
    for sid in symptom_nodes:
        orig = original_values.get(sid, 0.0)
        cf   = counterfactual_values.get(sid, 0.0)

        if not recovery:
            if orig >= 0.5 and cf < 0.5:
                total_weight += orig
        else:
            if orig < 0.5 and cf >= 0.5:
                total_weight += cf

    return total_weight # weighted by severity numeric





# ------------------------------------------------------------------------
# DOCTOR DIFFERENTIAL EVALUATION
# ------------------------------------------------------------------------

def get_doctor_differential(li):
    return [val["concept"]["id"] for val in li if val["concept"]["id"] is not None]


def produce_differentials(card):
    return dict(
        [
            [val["user"]["id"], get_doctor_differential(val["doctor_diseases"])]
            for val in card["outcomes"]
            if "doctor_diseases" in val
        ]
    )


def doctor_top_ns(card, true_disease):
    differentials = produce_differentials(card)
    return [
        [key, len(val), 1 if true_disease in val else 0]
        for key, val in differentials.items()
    ]


# ------------------------------------------------------------------------
# STATISTICS / DEBUGGING
# ------------------------------------------------------------------------

def mean_list(li):
    return "none" if not li else np.mean(li)


def bintest(x, y, comf_thresh):
    if comf_thresh == 0:
        return sum(x) >= sum(y)
    p_val = binomtest(sum(x), n=len(x), p=sum(y) / len(y), alternative="greater").pvalue
    return p_val < comf_thresh

