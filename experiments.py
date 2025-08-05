from tqdm import tqdm

from constants import (
    VIGNETTES_FILE,
    RESULTS_OBS_FILE,
    RESULTS_CF_DISABLEMENT_FILE,
    RESULTS_CF_SUFFICIENCY_FILE,
)

from utils import load_from_json, save_as_json, write_to_pickle
from preprocessing import preprocess_vignettes, convert_symptom_severity
from helpers import load_networks, get_symptom_nodes
from inference import (
    expected_disablement,
    expected_sufficiency,
    get_evidence_from_casecard,
    posterior_inference,
)


def compute_disease_posteriors(network, facts):
    return posterior_inference(network, facts)


def run_vignettes_experiment_raw(vignettes_data, network_data, first_n=None):
    """
    For all vignettes:
    - Compute posterior disease scores
    - Compute counterfactual expected disablement
    - Compute counterfactual expected sufficiency
    """
    if first_n is not None:
        vignettes_data = dict(list(vignettes_data.items())[:first_n])

    # Attach severity_numeric to each symptom
    vignettes_data = convert_symptom_severity(vignettes_data)

    symptom_nodes = get_symptom_nodes(network_data)
    all_diseases = [
        nid for nid, node in network_data.items()
        if node.get("label") == "Disease"
    ]

    posterior_results = {}
    disablement_results = {}
    sufficiency_results = {}

    for v_id, vignette in tqdm(vignettes_data.items(), desc="Casecards"):
        card = vignette["card"]
        net_name = card["network_name"]
        network = network_data.get(net_name)

        if network is None:
            print(f"[ERROR] Missing network '{net_name}' for case {v_id}")
            continue

        symptom_nodes = get_symptom_nodes(network)
        all_diseases = [nid for nid, node in network.items() if node.get("label") == "Disease"]

        facts = get_evidence_from_casecard(card)

        posterior = compute_disease_posteriors(network, facts)
        disablement = expected_disablement(
            network, facts, all_diseases, symptom_nodes
        )
        sufficiency = expected_sufficiency(
            network, facts, all_diseases, symptom_nodes
        )

        posterior_results[v_id] = posterior
        disablement_results[v_id] = disablement
        sufficiency_results[v_id] = sufficiency

        # Warn if true disease is missing from any metric
        true_id = card["diseases"][0]["id"]
        for method, scores in [
            ("posterior", posterior),
            ("disablement", disablement),
            ("sufficiency", sufficiency),
        ]:
            if true_id not in scores:
                print(f"[WARN] case {v_id}: true disease {true_id} missing from {method}")

    return posterior_results, disablement_results, sufficiency_results


def save_results(output_dir, posteriors, disablements, sufficiencies):
    """
    Save result dictionaries to pickle and JSON files.
    """
    write_to_pickle(posteriors, output_dir / RESULTS_OBS_FILE)
    write_to_pickle(disablements, output_dir / RESULTS_CF_DISABLEMENT_FILE)
    write_to_pickle(sufficiencies, output_dir / RESULTS_CF_SUFFICIENCY_FILE)

    # Merge into a single JSON file for inspection
    merged = {
        v_id: {
            "posterior": posteriors[v_id],
            "disablement": disablements[v_id],
            "sufficiency": sufficiencies[v_id],
        }
        for v_id in posteriors
    }
    save_as_json(merged, output_dir / "experimental_results.json")


def main(data_path, output_path):
    """
    Entrypoint used by run.py: loads data, runs experiments, saves output.
    """
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
    """
    CLI-compatible wrapper used by run.py
    """
    vignette_data = load_from_json(args.datapath / VIGNETTES_FILE)
    network_data = load_networks(args.datapath)

    posteriors, disablements, sufficiencies = run_vignettes_experiment_raw(
        vignette_data, network_data, first_n=args.first
    )

    save_results(args.results, posteriors, disablements, sufficiencies)
