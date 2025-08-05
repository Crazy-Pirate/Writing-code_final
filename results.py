import json
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from constants import VIGNETTES_FILE, RESULTS_FILE, DATA_PATH
from pathlib import Path

from utils import load_from_json


def top_n_accuracy(results_dict, vignettes, method_key, N=20):
    topn_hits = np.zeros(N)
    total = 0

    for vid, result in results_dict.items():
        card = vignettes[vid]
        true_disease = card["card"]["diseases"][0]["id"]

        scores = result.get(method_key, {})
        if true_disease not in scores:
            continue

        sorted_diseases = sorted(scores, key=scores.get, reverse=True)
        for k in range(N):
            if true_disease in sorted_diseases[:k + 1]:
                topn_hits[k] += 1
        total += 1

    return topn_hits / total if total else np.zeros(N)


def plot_topn_accuracy(all_results, vignettes):
    methods = ["posterior", "disablement", "sufficiency"]
    labels = {"posterior": "Posterior", "disablement": "Disablement", "sufficiency": "Sufficiency"}

    for method in methods:
        acc = top_n_accuracy(all_results, vignettes, method)
        plt.plot(range(1, 21), acc, label=labels[method])

    plt.xlabel("Top-N")
    plt.ylabel("Accuracy")
    plt.title("Top-N Accuracy Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def doctor_score_matrix(vignettes, results_dict):
    """Compares top-1 disease by model vs doctor annotations."""
    correct = {"posterior": 0, "disablement": 0, "sufficiency": 0}
    total = 0

    for vid, result in results_dict.items():
        card = vignettes[vid]
        true_id = card["card"]["diseases"][0]["id"]
        doctor_lists = [
            [d["concept"]["id"] for d in outcome["doctor_diseases"]]
            for outcome in card["outcomes"]
            if "doctor_diseases" in outcome
        ]
        doctor_rank = set(sum(doctor_lists, []))

        for method, count in correct.items():
            scores = result.get(method, {})
            if not scores:
                continue
            top_disease = max(scores, key=scores.get)
            if top_disease in doctor_rank:
                correct[method] += 1

        total += 1

    return {k: v / total for k, v in correct.items()}


def stratify_by_rareness(vignettes, results_dict, key):
    rarity_scores = defaultdict(list)

    for vid, result in results_dict.items():
        card = vignettes[vid]
        true_id = card["card"]["diseases"][0]["id"]
        rareness = card["card"]["diseases"][0].get("rareness", "unknown")

        score_dict = result[key]
        if true_id in score_dict:
            rarity_scores[rareness].append(score_dict[true_id])

    return {
        k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
        for k, v in rarity_scores.items()
    }


def main(results_folder: Path = Path("my_results")):
    vignettes = load_from_json(DATA_PATH / VIGNETTES_FILE)
    # load results from the folder passed to run.py (match default or arg)
    results_path = DATA_PATH.parent / "my_results" / RESULTS_FILE
    results_dict = load_from_json(Path(results_path))


    # Top-N accuracy
    print("\n>> Top-N Accuracy Plot")
    plot_topn_accuracy(results_dict, vignettes)

    # Doctor comparison score
    print("\n>> Doctor Agreement Score")
    doctor_scores = doctor_score_matrix(vignettes, results_dict)
    for k, v in doctor_scores.items():
        print(f"{k.title()} Score: {v:.4f}")

    # Stratified metrics by rareness
    for metric in ["posterior", "disablement", "sufficiency"]:
        strat = stratify_by_rareness(vignettes, results_dict, key=metric)
        print(f"\n>> {metric.title()} Results Stratified by Disease Rareness")
        for r, stat in strat.items():
            print(f"  {r:15}: mean={stat['mean']:.3f} std={stat['std']:.3f}")


if __name__ == "__main__":
    main()
