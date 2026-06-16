"""
setup.py — noc_inference package (hierarchy_v7 / con_noccai_v7)
================================================================
Makes the package pip-installable directly from the GCCode GitLab repo:

    pip install git+https://gccode.ssc-spc.gc.ca/ryan.r.spencer/con_noccai.git

After install, users import with:

    from noc_inference import load_predictor, predict_csv, predict_rows, print_summary

Model weights (checkpoint_step_42952.pt and distilbert-base-uncased/pytorch_model.bin)
are stored under noc_inference/checkpoints/ and noc_inference/models/ via Git LFS.
They are downloaded automatically when pip clones the repo.

REQUIRES git-lfs on the installing machine:
    git lfs install      # run once per machine before pip install
"""

from setuptools import setup, find_packages
from pathlib import Path

# Long description from README if it exists
_HERE = Path(__file__).parent
long_description = ""
readme = _HERE / "README.md"
if readme.exists():
    long_description = readme.read_text(encoding="utf-8")

setup(
    name="noc_inference",
    version="0.7.0",

    description=(
        "NOC 2021 hierarchical job-description classifier — hierarchy_v7. "
        "DistilBERT-base-uncased backbone, 515 L5 unit-group classes."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",

    author="Ryan Spencer",
    author_email="ryan.r.spencer@hrsdc-rhdcc.gc.ca",

    # Packages to install: both the public API wrapper AND the model implementation
    packages=find_packages(
        exclude=["tests*", "*.tests*", "training*", "evaluate*"]
    ),

    # Non-.py files that must be bundled with the package
    package_data={
        # noc_inference: taxonomy CSV, model checkpoint, DistilBERT base model
        "noc_inference": [
            "data/noc_2021_merged_clean.csv",
            "checkpoints/checkpoint_step_42952.pt",         # via Git LFS
            "models/distilbert-base-uncased/config.json",
            "models/distilbert-base-uncased/tokenizer_config.json",
            "models/distilbert-base-uncased/tokenizer.json",
            "models/distilbert-base-uncased/vocab.txt",
            "models/distilbert-base-uncased/pytorch_model.bin",  # via Git LFS
        ],
        # con_noccai_v7: label map JSON (maps 0-514 indices ↔ 5-digit NOC codes)
        "con_noccai_v7": [
            "label_map.json",
        ],
    },
    include_package_data=True,

    # Runtime dependencies — versions pinned to what the jobpostings env uses
    install_requires=[
        "torch>=2.3.0",
        "transformers>=4.40.0",
        "pandas>=1.5.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.64.0",
    ],

    python_requires=">=3.9",

    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
