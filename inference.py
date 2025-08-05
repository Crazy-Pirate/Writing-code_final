from helpers import make_twin_network, count_disabled_symptoms, get_symptom_nodes, noisy_or
from preprocessing import SEVERITY_MAPPING  # ✅ Use centralized mapping
#from utils import load_from_json  # ✅ only if used during testing

RISK_BOOST = 5.0  # ✅ Tunable — can adjust to control impact of risk factors


def get_evidence_from_casecard(card):
    """
    Extracts evidence (symptoms + risk) from a vignette casecard,
    returning real-valued severity + boosted risk.
    """
    evidence = {}
    inner = card  # ✅ Matches OG code structure

    # ✅ SYMPTOMS
    for sym in inner.get("symptoms", []):
        # Skip if label is 'Super' or concept ID is missing
        if sym.get("label") == "Super" or sym.get("concept", {}).get("id") is None:
            continue
        sid = sym["concept"]["id"]

        # Use precomputed severity_numeric if present; fallback to severity string
        if "severity_numeric" in sym:
            sev_num = sym["severity_numeric"]
        else:
            sev = sym.get("severity", "NOT_PRESENT").upper()
            sev_num = SEVERITY_MAPPING.get(sev, 1.0 if sev != "NOT_PRESENT" else 0.0)
        evidence[sid] = sev_num

    # ✅ RISK FACTORS
    for rf in inner.get("risk_factors", []):
        if rf.get("label") != "Risk" or rf.get("concept", {}).get("id") is None:
            continue
        if rf.get("presence", "").upper() == "PRESENT":
            rid = rf["concept"]["id"]
            evidence[rid] = RISK_BOOST  # ✅ Fixed boost, only if present

    return evidence


def posterior_inference(network, evidence):
    """Compute P(node=1 | evidence) for Disease, Symptom, and Risk"""
    results = {}
    for node_id, node in network.items():
        if node.get("label") in ("Disease", "Symptom"):
            parents = node.get("parents", [])
            results[node_id] = noisy_or(parents, network, evidence)
        elif node.get("label") == "Risk":
            results[node_id] = evidence.get(node_id, 0.0)
    if not results:
        print("[ERROR] posterior_inference: no scores computed!")
    return results  # ✅ no normalization



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
    if not results:
        print("[ERROR] expected_disablement: no scores computed!")
    return results 


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
    if not results:
        print("[ERROR] expected_sufficiency: no scores computed!")
    return results 


