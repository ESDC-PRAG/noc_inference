"""
run_inference.py
================
Entry-point script — uses the noc_inference package.

USAGE
-----
Single smoke-test (3 hardcoded examples):
    python run_inference.py

Batch mode — predict a full CSV:
    python run_inference.py --mode batch --input jobs.csv
    python run_inference.py --mode batch --input jobs.csv --output results.csv

FOLDER LAYOUT (required)
------------------------
    your_folder/
        con_noccai_v7/      <- the model package
        noc_inference/      <- the wrapper package  (config.py, batch.py, __init__.py)
        run_inference.py    <- this file

FIRST-TIME SETUP
----------------
1.  Open noc_inference/config.py and fill in the four model paths.
2.  conda activate jobpostings
3.  python run_inference.py
"""

import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Smoke-test examples ───────────────────────────────────────────────────────

EXAMPLES = [
    {
        "title": "Senior Software Developer",
        "description": (
            "Designs, develops, and maintains Python microservices and REST APIs. "
            "Leads code reviews, mentors junior developers, and contributes to "
            "system architecture decisions. Experience with Docker and CI/CD pipelines."
        ),
    },
    {
        "title": "Registered Nurse",
        "description": (
            "Provides direct patient care in a hospital medical-surgical unit. "
            "Administers medications, monitors vital signs, and coordinates with "
            "physicians and allied health professionals to develop care plans."
        ),
    },
    {
        "title": "Truck Driver",
        "description": (
            "Operates heavy trucks for long-haul transportation of goods across "
            "provincial routes. Responsible for vehicle inspections, logbooks, "
            "and safe cargo delivery within regulatory timelines."
        ),
    },
]


def run_single(top_k: int = 5) -> None:
    """Run hardcoded smoke-test examples and print results."""
    from noc_inference import load_predictor, predict_rows

    predictor = load_predictor()

    descriptions = [ex["description"] for ex in EXAMPLES]
    titles       = [ex["title"]       for ex in EXAMPLES]

    pred_df = predict_rows(
        descriptions=descriptions,
        titles=titles,
        predictor=predictor,
        top_k=top_k,
    )

    print()
    print("=" * 72)
    print("  con_noccai_v7  —  smoke-test predictions")
    print("=" * 72)

    for i, ex in enumerate(EXAMPLES):
        row = pred_df.iloc[i]
        print(f"\nJob title  : {ex['title']}")
        print(f"Description: {ex['description'][:88]}...")
        print()
        print(f"  {'Rank':<5} {'NOC':>6}  {'Confidence':>11}  Title")
        print(f"  {'----':<5} {'---':>6}  {'-----------':>11}  -----")

        for rank in range(1, top_k + 1):
            code_col = f"pred_{rank}_noc_code"
            if code_col not in pred_df.columns:
                break
            code  = row[f"pred_{rank}_noc_code"]
            title = row[f"pred_{rank}_noc_title"]
            conf  = float(row[f"pred_{rank}_confidence"])
            print(f"  {rank:<5} {code:>6}  {conf:>10.2%}  {title}")

        print()
        print(f"  Hierarchy  : {row.get('top1_hierarchy', 'n/a')}")
        print(f"  Lvl confs  : {row.get('level_confidences', 'n/a')}")
        print("-" * 72)

    print()


def run_batch(input_path: str, output_path: str | None) -> None:
    """Predict a full CSV and write results."""
    from noc_inference import predict_csv, print_summary

    final_df = predict_csv(
        input_path=input_path,
        output_path=output_path,   # None -> auto-named <stem>_predictions.csv
    )
    print_summary(final_df)


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="NOC inference -- single smoke-test or batch CSV mode"
    )
    p.add_argument(
        "--mode",
        choices=["single", "batch"],
        default="single",
        help="Run mode (default: single)",
    )
    p.add_argument(
        "--input",
        default=None,
        help="[batch] Path to input CSV",
    )
    p.add_argument(
        "--output",
        default=None,
        help="[batch] Path to output CSV (default: <input>_predictions.csv)",
    )
    p.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="[single] Number of ranked predictions to show (default: 5)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Validate config paths before loading anything heavy
    try:
        from noc_inference import config
    except ModuleNotFoundError:
        logger.error(
            "Cannot import noc_inference. "
            "Make sure noc_inference/ is in the same folder as this script "
            "and your jobpostings conda env is active."
        )
        sys.exit(1)

    errors = config.validate()
    if errors:
        for e in errors:
            logger.error("  x  %s", e)
        logger.error(
            "Fix the paths in noc_inference/config.py and re-run."
        )
        sys.exit(1)

    if args.mode == "single":
        run_single(top_k=args.top_k)

    elif args.mode == "batch":
        if not args.input:
            logger.error("--input is required for batch mode.")
            sys.exit(1)
        run_batch(args.input, args.output)


if __name__ == "__main__":
    main()
