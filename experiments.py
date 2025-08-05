import numpy as np
import json
import pickle
from tqdm import tqdm
from preprocessing import convert_symptom_severity, load_vignettes, preprocess_vignettes

from constants import (
    VIGNETTES_FILE,
    NETWORKS_FILE,
    RESULTS_OBS_FILE,
    RESULTS_CF_DISABLEMENT_FILE,
    RESULTS_CF_SUFFICIENCY_FILE,
)

from utils import load_from_json, save_as_json, write_to_pickle
from helpers import (
    load_networks,
    noisy_or,
    make_twin_network,
    count_disabled_symptoms,
    build_disease_graph,
    get_symptom_nodes,
    get_risk_nodes,
)


from inference import expected_disablement, expected_sufficiency, get_evidence_from_casecard



def compute_disease_posteriors(network, facts):
    """
    Compute the noisy-OR based posterior probability for each disease.
    """
    graph = build_disease_graph(network)
    posterior = {}

    for node_id in graph.nodes:
        node = network[node_id]
        if node.get("label") == "Disease":
            # Only consider diseases
            parents = node.get("parents", [])
            posterior[node_id] = noisy_or(parents, network, facts)

    return posterior


def run_vignettes_experiment_raw(vignettes_data, network_data, first_n=None):
    
    if first_n is not None:
        vignettes_data = dict(list(vignettes_data.items())[:first_n])

    """
    For all vignettes: compute
    - Posterior ranking scores
    - Counterfactual expected disablement
    - Counterfactual expected sufficiency
    """
    posterior_results = {}
    disablement_results = {}
    sufficiency_results = {}
    
    symptom_nodes = get_symptom_nodes(network_data)
    all_diseases = [nid for nid, node in network_data.items() if node.get("label") == "Disease"]

    # ensure we attach severity_numeric
    vignettes_data = preprocess_vignettes(vignettes_data)  # uses load+convert


    for v_id, vignette in tqdm(vignettes_data.items(), desc="Casecards"):
        card = vignette["card"] 

        # 1. Extract symptom and risk factors from the card
        facts = get_evidence_from_casecard(card)


        # 2. Run normal inference to get posterior scores
        posterior = compute_disease_posteriors(network_data, facts)
        # normalize posterior
        from utils import normalize_dict
        posterior = normalize_dict(posterior)
        posterior_results[v_id] = posterior

        # 3. Compute expected disablement over twin network
        disablement = expected_disablement(
            network_data, facts, all_diseases, symptom_nodes
        )
        disablement_results[v_id] = disablement

        # 4. Compute expected sufficiency scores
        sufficiency = expected_sufficiency(
            network_data, facts, all_diseases, symptom_nodes
        )
        sufficiency_results[v_id] = sufficiency

        # sanity check: warn if true disease missing
        true_id = card["diseases"][0]["id"]
        for method, scores in [("posterior", posterior),
                               ("disablement", disablement),
                               ("sufficiency", sufficiency)]:
            if true_id not in scores:
                print(f"[WARN] case {v_id}: true disease {true_id} missing from {method}")
        

    return posterior_results, disablement_results, sufficiency_results


def save_results(output_dir, posteriors, disablements, sufficiencies):
    """Save outputs to disk."""
    write_to_pickle(posteriors, output_dir / RESULTS_OBS_FILE)
    write_to_pickle(disablements, output_dir / RESULTS_CF_DISABLEMENT_FILE)
    write_to_pickle(sufficiencies, output_dir / RESULTS_CF_SUFFICIENCY_FILE)

    # For compatibility with original repo — merge for JSON export
    merged = {
        v_id: {
            "posterior": posteriors[v_id],
            "disablement": disablements[v_id],
            "sufficiency": sufficiencies[v_id],
        }
        for v_id in posteriors
    }
    save_as_json(merged, output_dir / "experimental_results.json")

from preprocessing import preprocess_vignettes

def main(data_path, output_path):
    """Main entrypoint from run.py."""
    print("> Loading data...")
    vignette_data = preprocess_vignettes(data_path / VIGNETTES_FILE)
    network_data = load_networks(data_path)

    print("> Running all experiments...")
    posteriors, disablements, sufficiencies = run_vignettes_experiment_raw(
        vignette_data, network_data
    )

    print("> Saving outputs...")
    save_results(output_path, posteriors, disablements, sufficiencies)

    print("> Done.")


def run_vignettes_experiment(*, args):
    """Wrapper to match CLI-style invocation from run.py"""
    vignette_data = load_from_json(args.datapath / VIGNETTES_FILE)
    network_data = load_networks(args.datapath)

    posteriors, disablements, sufficiencies = run_vignettes_experiment_raw(
        vignette_data, network_data, first_n=args.first
    )

    save_results(args.results, posteriors, disablements, sufficiencies)
