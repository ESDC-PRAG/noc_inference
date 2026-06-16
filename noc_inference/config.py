"""
noc_inference/config.py
=======================
Centralised configuration for the noc_inference inference package (hierarchy_v7).

After pip install, model files live INSIDE the installed package directory:
    noc_inference/checkpoints/checkpoint_step_42952.pt
    noc_inference/models/distilbert-base-uncased/

These paths are resolved automatically relative to this config file, so no
manual path-setting is needed after a clean `pip install git+https://...`.

────────────────────────────────────────────────────────────────────────────────
OVERRIDING PATHS (optional)
────────────────────────────────────────────────────────────────────────────────
If the installed LFS files are missing (e.g. git-lfs wasn't installed on the
machine that ran pip install), you can point the package at a local copy:

  Option A — environment variables (set before launching Python):
      set NOC_CHECKPOINT_PATH=C:\\NOC_Model\\checkpoints\\checkpoint_step_42952.pt
      set NOC_DISTILBERT_PATH=C:\\NOC_Model\\models\\distilbert-base-uncased

  Option B — runtime call before load_predictor():
      import noc_inference
      noc_inference.set_paths(
          checkpoint = r"\\\\server\\share\\NOC_Model\\checkpoint_step_42952.pt",
          distilbert = r"\\\\server\\share\\NOC_Model\\distilbert-base-uncased",
      )
      predictor = noc_inference.load_predictor()
────────────────────────────────────────────────────────────────────────────────

Architecture constants for hierarchy_v7 (do NOT change — must match checkpoint):
    Backbone   : distilbert-base-uncased  (English, 768-dim)
    Heads      : classification_heads.level_1 … level_5
    L1→L5 sizes: 10, 45, 89, 162, 515
    Head type  : L1/L2 shallow (Dropout→Linear)
                 L3/L4/L5 deep (Dropout→Linear(768→384)→ReLU→Dropout→Linear(384→N))
    Proj head  : Linear(768→512)→ReLU→Dropout→Linear(512→256)
    EMA weights: checkpoint['ema_state']['shadow']
    Best ckpt  : checkpoint_step_42952.pt  (epoch 14, L5 val acc 93.98%)
"""

from __future__ import annotations

import os
from pathlib import Path

# ─── Package-relative base paths ─────────────────────────────────────────────
# _PKG_DIR resolves to wherever pip installed the noc_inference package,
# e.g.  .../site-packages/noc_inference/
# _V7_DIR  resolves to the sibling con_noccai_v7 package installed alongside it.
_PKG_DIR = Path(__file__).resolve().parent          # .../noc_inference/
_V7_DIR  = _PKG_DIR.parent / "con_noccai_v7"       # .../con_noccai_v7/

# ─── Bundled file paths (set at install time via Git LFS) ────────────────────
_BUNDLED_CHECKPOINT  = _PKG_DIR / "checkpoints"  / "checkpoint_step_42952.pt"
_BUNDLED_DISTILBERT  = _PKG_DIR / "models"       / "distilbert-base-uncased"
_BUNDLED_TAXONOMY    = _PKG_DIR / "data"          / "noc_2021_merged_clean.csv"
_BUNDLED_LABEL_MAP   = _V7_DIR  / "label_map.json"

# ─── Configurable paths (env-var override → bundled default) ─────────────────
# These module-level variables are what the rest of the package reads.
# Use set_paths() or env vars to override without editing this file.

CHECKPOINT_PATH: str = os.environ.get(
    "NOC_CHECKPOINT_PATH",
    str(_BUNDLED_CHECKPOINT),
)

BERT_MODEL_PATH: str = os.environ.get(
    "NOC_DISTILBERT_PATH",
    str(_BUNDLED_DISTILBERT),
)

NOC_TITLES_PATH: str = os.environ.get(
    "NOC_TAXONOMY_PATH",
    str(_BUNDLED_TAXONOMY),
)

# Label map is always bundled alongside con_noccai_v7 — no env-var override
# needed because it's a tiny JSON file (not a large binary).
LABEL_MAP_PATH: str = str(_BUNDLED_LABEL_MAP)

# ─── Inference constants ──────────────────────────────────────────────────────
TOP_K: int          = 5     # number of ranked NOC predictions per row
BATCH_SIZE: int     = 64    # rows per GPU batch (RTX A2000 12GB fits 64 comfortably)
MAX_LENGTH: int     = 512   # DistilBERT token limit; sliding window applied if exceeded
SHORT_TEXT_MIN_WORDS: int = 25   # rows below this are flagged as potentially unreliable

HIGH_CONF_THRESHOLD: float = 0.80   # ≥ this  →  high confidence bucket
LOW_CONF_THRESHOLD:  float = 0.40   # <  this  →  flag for review bucket

# ─── set_paths() runtime override ────────────────────────────────────────────

def set_paths(
    checkpoint: str | None = None,
    distilbert: str | None = None,
    taxonomy:   str | None = None,
    label_map:  str | None = None,
) -> None:
    """
    Override model file paths at runtime.  Call this BEFORE load_predictor().

    All arguments are optional — only supply the ones you want to change.

    Parameters
    ----------
    checkpoint : str, optional
        Absolute path to the .pt checkpoint file.
        e.g. r"C:\\NOC_Model\\checkpoints\\checkpoint_step_42952.pt"
    distilbert : str, optional
        Absolute path to the distilbert-base-uncased model folder.
        e.g. r"C:\\NOC_Model\\models\\distilbert-base-uncased"
    taxonomy : str, optional
        Absolute path to noc_2021_merged_clean.csv.
        Defaults to the bundled copy inside the package — rarely needs changing.
    label_map : str, optional
        Absolute path to label_map.json.
        Defaults to the bundled copy inside con_noccai_v7 — rarely needs changing.

    Example
    -------
    Shared GoC network drive:

        import noc_inference
        noc_inference.set_paths(
            checkpoint = r"\\\\server\\share\\NOC_Model\\checkpoint_step_42952.pt",
            distilbert = r"\\\\server\\share\\NOC_Model\\distilbert-base-uncased",
        )
        predictor = noc_inference.load_predictor()

    Local override after install without LFS:

        import noc_inference
        noc_inference.set_paths(
            checkpoint = r"C:\\Users\\you\\NOC_Model\\checkpoint_step_42952.pt",
            distilbert = r"C:\\Users\\you\\NOC_Model\\distilbert-base-uncased",
        )
    """
    global CHECKPOINT_PATH, BERT_MODEL_PATH, NOC_TITLES_PATH, LABEL_MAP_PATH
    if checkpoint is not None:
        CHECKPOINT_PATH = str(checkpoint)
    if distilbert  is not None:
        BERT_MODEL_PATH = str(distilbert)
    if taxonomy    is not None:
        NOC_TITLES_PATH = str(taxonomy)
    if label_map   is not None:
        LABEL_MAP_PATH  = str(label_map)


# ─── validate() sanity check ──────────────────────────────────────────────────

def validate() -> list[str]:
    """
    Check that all required paths exist on disk.

    Returns
    -------
    list[str]
        List of error messages.  Empty list means everything is OK.

    Example
    -------
        import noc_inference.config as cfg
        errors = cfg.validate()
        if errors:
            for e in errors:
                print("MISSING:", e)
        else:
            print("All paths OK — ready to load predictor.")
    """
    errors: list[str] = []
    checks: dict[str, str] = {
        "CHECKPOINT_PATH (checkpoint_step_42952.pt)": CHECKPOINT_PATH,
        "BERT_MODEL_PATH (distilbert-base-uncased/)": BERT_MODEL_PATH,
        "NOC_TITLES_PATH (noc_2021_merged_clean.csv)": NOC_TITLES_PATH,
        "LABEL_MAP_PATH  (label_map.json)":             LABEL_MAP_PATH,
    }
    for label, path in checks.items():
        if not Path(path).exists():
            errors.append(f"{label}  →  NOT FOUND: {path}")
    return errors


def print_paths() -> None:
    """
    Print the active path configuration and whether each file exists.
    Useful for quick diagnosis when setting up a new machine.

    Example
    -------
        python -c "from noc_inference.config import print_paths; print_paths()"
    """
    checks = {
        "Checkpoint (.pt)  ": CHECKPOINT_PATH,
        "DistilBERT folder ": BERT_MODEL_PATH,
        "NOC taxonomy CSV  ": NOC_TITLES_PATH,
        "Label map JSON    ": LABEL_MAP_PATH,
    }
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║            noc_inference  path configuration                ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    for label, path in checks.items():
        status = "✓" if Path(path).exists() else "✗ MISSING"
        print(f"║  {status}  {label}")
        print(f"║         {path}")
        print("║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
