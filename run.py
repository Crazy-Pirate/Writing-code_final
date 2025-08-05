import argparse
from pathlib import Path

from experiments import run_vignettes_experiment

from inference import create_marginals_files


def parse_args():
    parser = argparse.ArgumentParser(description="Run Causal Diagnostic Experiments")

    parser.add_argument(
        "--datapath", type=Path, default=Path("data"),
        help="Path to folder containing input data (networks, vignettes, etc.)"
    )
    parser.add_argument(
        "--results", type=Path, default=Path("my_results"),
        help="Output folder for storing results"
    )
    parser.add_argument(
        "--first", type=int, default=None,
        help="Run only the first N vignettes (for debugging or quick test)"
    )
    parser.add_argument(
        "--reproduce", action="store_true",
        help="Use previously computed results.json instead of running inference"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print progress info while running"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Ensure results folder exists
    args.results.mkdir(parents=True, exist_ok=True)

    # Run the vignette experiments (generates posterior, disablement, sufficiency)
    run_vignettes_experiment(args=args)

    print("\n>> Inference complete. Run `python results.py` to evaluate results.")


if __name__ == "__main__":
    args = parse_args()
    # ensure results dir
    args.results.mkdir(parents=True, exist_ok=True)

    # Step 1: build marginals (only once)
    from inference import create_marginals_files
    create_marginals_files(args=args)

    # Step 2: run experiments
    run_vignettes_experiment(args=args)

    print("\n>> Inference complete. Run `python results.py --results <folder>` to evaluate.")