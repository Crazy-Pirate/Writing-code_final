import json
import seaborn as sns
import pandas as pd
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from constants import VIGNETTES_FILE, RESULTS_FILE, DATA_PATH
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
        for method in correct.keys():
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


def plot_score_distributions(results_dict):
    methods = ["posterior", "disablement", "sufficiency"]
    plt.figure(figsize=(15, 4))
    for i, method in enumerate(methods):
        scores = []
        for v in results_dict.values():
            scores.extend(list(v.get(method, {}).values()))
        plt.subplot(1, 3, i + 1)
        plt.hist(scores, bins=30, color="skyblue", edgecolor="black")
        plt.title(f"{method.title()} Score Distribution")
        plt.xlabel("Score")
        plt.ylabel("Frequency")
        plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_rareness_vs_avg_severity_heatmap(vignettes, results_dict, metric="posterior"):
    heatmap_data = defaultdict(list)
    for vid, result in results_dict.items():
        card = vignettes[vid]["card"]
        rareness = card["diseases"][0].get("rareness", "unknown")
        true_id = card["diseases"][0]["id"]
        severities = [
            s.get("severity_numeric", 0.0)
            for s in card.get("symptoms", [])
            if s.get("severity_numeric", 0.0) > 0
        ]
        avg_sev = round(np.mean(severities), 1) if severities else 0.0
        score = result.get(metric, {}).get(true_id)
        if score is not None:
            heatmap_data[(rareness, avg_sev)].append(score)
    data = defaultdict(dict)
    for (rareness, avg_sev), scores in heatmap_data.items():
        data[rareness][avg_sev] = np.mean(scores)
    df = pd.DataFrame(data).T.sort_index()
    df = df.fillna(0)
    plt.figure(figsize=(10, 6))
    sns.heatmap(df, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title(f"Avg {metric.title()} Score by Rareness × Avg Severity")
    plt.xlabel("Avg Symptom Severity")
    plt.ylabel("Disease Rareness")
    plt.tight_layout()
    plt.show()


def main(results_folder: Path = Path("my_results")):
    vignettes = load_from_json(DATA_PATH / VIGNETTES_FILE)
    results_path = results_folder / RESULTS_FILE
    results_dict = load_from_json(results_path)

    print("\n>> Top-N Accuracy Plot")
    plot_topn_accuracy(results_dict, vignettes)

    print("\n>> Doctor Agreement Score")
    doctor_scores = doctor_score_matrix(vignettes, results_dict)
    for k, v in doctor_scores.items():
        print(f"{k.title()} Score: {v:.4f}")

    for metric in ["posterior", "disablement", "sufficiency"]:
        strat = stratify_by_rareness(vignettes, results_dict, key=metric)
        print(f"\n>> {metric.title()} Results Stratified by Disease Rareness")
        for r, stat in strat.items():
            print(f"  {r:15}: mean={stat['mean']:.3f} std={stat['std']:.3f}")

    print("\n>> Score Distribution Histograms")
    plot_score_distributions(results_dict)

    for metric in ["posterior", "disablement", "sufficiency"]:
        print(f"\n>> Heatmap: {metric.title()} by Rareness × Avg Severity")
        plot_rareness_vs_avg_severity_heatmap(vignettes, results_dict, metric=metric)


if __name__ == "__main__":
    main()
