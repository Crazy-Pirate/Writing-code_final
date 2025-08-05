Here is a complete `README.md` file tailored for your modified causal diagnosis project based on the Richens et al. (2020) paper:

---

# Counterfactual Medical Diagnosis with Causal Machine Learning (Modified)

This project is a **modification of the original causal inference framework** from the paper:

> **"Improving the accuracy of medical diagnosis with causal machine learning"**
> Richens, Murdock, and Kotoulas, *Nature Communications* (2020).
> [DOI: 10.1038/s41467-020-17419-7](https://doi.org/10.1038/s41467-020-17419-7)

---

## Modifications Introduced

This version preserves the **core mathematical inference engine** from the original codebase while implementing **three key enhancements**:

1. **Symptom Severity Handling**

   * Symptom presence is no longer binary.
   * Symbolic levels like `"MILD"`, `"MODERATE"`, and `"SEVERE"` are mapped to continuous numeric values using:

     ```python
     {
       "NOT_PRESENT": 0.0,
       "MILD": 0.3,
       "MODERATE": 0.6,
       "PRESENT": 1.0,
       "SEVERE": 1.2
     }
     ```

2. **Risk Factor Boosting**

   * Presence of a risk factor amplifies its influence on inference by a tunable multiplier (default = `5.0`).

3. **Score Interpretation Updates**

   * The posterior, sufficiency, and disablement scores remain float-valued.
   * Output can optionally be converted into Top-k binary vectors for performance evaluation.

---

## Project Structure

```bash
.
├── data/                         # Input JSON files: vignettes and Bayesian networks
│   ├── example_networks.json     # Bayesian network(s), keyed by 'A', 'B', 'G'
│   └── vignettes.json            # Casecards with symptoms, diseases, metadata
│
├── my_results/                   # Output folder (default)
│   ├── experimental_results.json # Final result combining all metrics
│   ├── results_obs.p             # Posterior probabilities
│   ├── results_counter_diss.p    # Counterfactual disablement scores
│   └── results_counter_suff.p    # Counterfactual sufficiency scores
│
├── run.py                        # Entrypoint to run all experiments
├── results.py                    # (Optional) Compute Top-k metrics or analysis
├── experiments.py                # Core experiment runner
├── inference.py                  # Inference wrapper using original logic
├── preprocessing.py              # Symptom severity and risk factor processing
├── helpers.py                    # Graph utils, twin network, disablement logic
├── utils.py                      # I/O helpers and metrics
└── README.md                     # ← You are here
```

---

## How to Run

1. **Install dependencies**

```bash
pip install numpy networkx tqdm scipy
```

2. **Place input files**

Make sure you have:

* `data/vignettes.json`
* `data/example_networks.json`

These are from the original paper's dataset.

3. **Run the modified diagnosis experiments**

```bash
python run.py --first 10
```

This runs inference on the first 10 vignettes (for debugging). Omit `--first` to run all.

4. **(Optional) Evaluate Results**

```bash
python results.py
```

This will compute evaluation metrics or inspect the results.

---

## Inference Pipeline Overview

* **Evidence Extraction**
  Symptoms are processed for both symbolic severity and binary presence.
  Risk factors marked `"PRESENT"` are upweighted.

* **Bayesian Network Inference**
  A separate Bayesian network is used for each vignette, chosen by `"network_name"` (`A`, `B`, `G`).

* **Counterfactual Evaluation**

  * **Disablement**: what happens if we disable a disease node?
  * **Sufficiency**: what happens if we force a disease node?

* **Results Output**
  Raw float scores for each disease are saved per vignette, across all metrics.

---

## Key Notes

   **Original Inference Retained**:
  You are using the exact same causal inference engine (`approximate_inference`) from the original paper, including:

  * Marginal priors
  * Time (DOS) correction via `"duration"` field
  * Twin network logic for counterfactuals

   **Data Format**:
  Dataset format must match that of the original Babylon Health dataset.
  The code assumes `"network_name"`, `"severity"`, and `"presence"` fields are present.

   **Top-k Evaluation**:
  Output scores can be transformed into ranked vectors (1 if true disease is in Top-k) to compare with doctor differentials.

---

# Reference

> Richens, J.G., Murdock, H.W., & Kotoulas, S. (2020). Improving the accuracy of medical diagnosis with causal machine learning. *Nature Communications*, 11, 3923.
> [https://doi.org/10.1038/s41467-020-17419-7](https://doi.org/10.1038/s41467-020-17419-7)

---

# Acknowledgments

* Original code and dataset courtesy of Babylon Health and the authors.
* Modified by me for research on improved causal diagnosis scoring with symptom severity and risk awareness.


