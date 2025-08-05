import argparse
from pathlib import Path

from experiments import run_vignettes_experiment

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
    return parser.parse_args()

def main():
    args = parse_args()
    args.results.mkdir(parents=True, exist_ok=True)

    # Run the vignette experiments (generates posterior, disablement, sufficiency)
    run_vignettes_experiment(args=args)

    print("\n>> Inference complete. Run `python results.py` to evaluate results.")

if __name__ == "__main__":
    main()
