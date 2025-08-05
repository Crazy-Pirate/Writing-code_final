import numpy as np
from helpers import make_twin_network, count_disabled_symptoms, get_symptom_nodes, noisy_or

from utils import normalize_dict, load_from_json
from constants import VIGNETTES_FILE, NETWORKS_FILE
from helpers import build_disease_graph

RISK_BOOST = 1.5 

def get_evidence_from_casecard(card):
    """
    Extracts evidence (symptoms + risk) from a vignette casecard,
    returning real-valued severity + boosted risk.
    """
    evidence = {}
    # severity map
    sev_map = {
        "NOT_PRESENT": 0.0,
        "MILD":       0.3,
        "MODERATE":   0.6,
        "PRESENT":    0.9,
        "SEVERE":     1.0,
    }
    # symptoms
    for sym in card.get("symptoms", []):
        sid = sym["concept"]["id"]
        evidence[sid] = sym.get("severity_numeric", sev_map.get(
            sym.get("severity", "PRESENT").upper(), 0.9))
    # risk factors
    for rf in card.get("risk_factors", []):
        rid = rf["concept"]["id"]
        evidence[rid] = max(evidence.get(rid, 0), 1.0) * RISK_BOOST
    return evidence


def posterior_inference(network, evidence):
    """Compute P(node=1 | evidence) for Disease, Symptom, and Risk"""
    results = {}
    for node_id, node in network.items():
        if node.get("label") in ("Disease", "Symptom"):
            parents = node.get("parents", [])
            results[node_id] = noisy_or(parents, network, evidence)
        elif node.get("label") == "Risk":
            # Just return 1.0 if present in evidence, else 0.0
            results[node_id] = evidence.get(node_id, 0.0)
    # ensure we always return a complete dict
    from utils import normalize_dict
    if not results:
        print("[ERROR] posterior_inference: no scores computed!")
    return normalize_dict(results)


def expected_disablement(network, evidence, disease_ids, symptom_nodes):
    """
    For each disease: disable it, count how many symptoms disappear
    Returns: {disease_id: score}
    """
    results = {}
    for disease_id in disease_ids:
        twin_net = make_twin_network(network, disable=disease_id)
        cf_values = posterior_inference(twin_net, evidence)
        original_values = posterior_inference(network, evidence)
        count = count_disabled_symptoms(network, symptom_nodes, original_values, cf_values, recovery=False)
        print(f"[Disablement] {disease_id} → score: {count:.3f}")
        results[disease_id] = count
    from utils import normalize_dict
    if not results:
        print("[ERROR] expected_disablement: no scores computed!")
    return normalize_dict(results)


def expected_sufficiency(network, evidence, disease_ids, symptom_nodes):
    """
    For each disease: force it on, count how many symptoms reappear (weighted by severity)
    Returns: {disease_id: score}
    """
    results = {}
    for disease_id in disease_ids:
        twin_net = make_twin_network(network, force=disease_id)
        cf_values = posterior_inference(twin_net, evidence)
        original_values = posterior_inference(network, evidence)
        count = count_disabled_symptoms(network, symptom_nodes, original_values, cf_values, recovery=True)
        print(f"[Sufficiency] {disease_id} → score: {count:.3f}")
        results[disease_id] = count
    from utils import normalize_dict
    if not results:
        print("[ERROR] expected_sufficiency: no scores computed!")
    return normalize_dict(results)



__all__ = ["get_evidence_from_casecard", "posterior_inference",
           "expected_disablement", "expected_sufficiency",
           "create_marginals_files"]
