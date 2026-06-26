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
CHANGING INFERENCE DEFAULTS (optional)
────────────────────────────────────────────────────────────────────────────────
All inference constants below are soft defaults.  Override them globally with
set_defaults(), or pass explicit values per-call to any predict_* function.

  Option A — change defaults once for the whole session:
      import noc_inference
      noc_inference.set_defaults(top_k=1, show_confidence=False, batch_size=32)

  Option B — pass per-call (overrides the session default for that call only):
      df = noc_inference.predict_csv(input_path, top_k=1, show_confidence=False)

────────────────────────────────────────────────────────────────────────────────
Architecture notes (hierarchy_v7 — for developers, not end users):
    Backbone   : distilbert-base-uncased  (English, 768-dim)
    Heads      : classification_heads.level_1 … level_5  (NOT heads.lN)
    L1→L5 sizes: 10, 45, 89, 162, 515
    Head type  : L1/L2 shallow  (Dropout → Linear)
                 L3/L4/L5 deep  (Dropout → Linear(768→384) → ReLU → Dropout → Linear(384→N))
    Proj head  : Linear(768→512) → ReLU → Dropout → Linear(512→256)
    EMA weights: checkpoint['ema_state']['shadow']
    Best ckpt  : checkpoint_step_42952.pt  (epoch 14, L5 val acc 93.98%, in-distribution)
"""

from __future__ import annotations

import os
from pathlib import Path

# ─── Package-relative base paths ─────────────────────────────────────────────
# _PKG_DIR resolves to wherever pip installed noc_inference,
# e.g.  .../site-packages/noc_inference/
# _V7_DIR  resolves to the sibling con_noccai_v7 package installed alongside it.
_PKG_DIR = Path(__file__).resolve().parent          # .../noc_inference/
_V7_DIR  = _PKG_DIR.parent / "con_noccai_v7"       # .../con_noccai_v7/

# ─── Bundled file paths (populated via Git LFS at install time) ───────────────
_BUNDLED_CHECKPOINT = _PKG_DIR / "checkpoints" / "checkpoint_step_42952.pt"
_BUNDLED_DISTILBERT = _PKG_DIR / "models"      / "distilbert-base-uncased"
_BUNDLED_TAXONOMY   = _PKG_DIR / "data"        / "noc_2021_merged_clean.csv"


def _find_label_map() -> Path:
    """
    Try multiple candidate locations for label_map.py, in priority order.

    Priority:
      1. con_noccai_v7.__file__ location — imports the package and asks
         Python directly where it lives.  Works for pip install, editable
         installs, and local-folder setups equally.
      2. Sibling con_noccai_v7/ folder next to noc_inference/ (pip install
         case where both packages land in the same site-packages directory).
      3. Known GoC machine fallback paths for the local dev layout.

    Returns the first path that actually exists on disk, or falls back to
    the sibling candidate so that validate() surfaces a clear error.
    """
    candidates: list[Path] = []

    # 1. Locate via the importable con_noccai_v7 package (most reliable)
    try:
        import con_noccai_v7  # noqa: PLC0415
        candidates.append(Path(con_noccai_v7.__file__).resolve().parent / "label_map.py")
    except ImportError:
        pass

    # 2. Sibling package directory (clean pip install from GitLab)
    candidates.append(_V7_DIR / "label_map.py")

    # 3. Known fallback paths for the GoC local-dev layout
    candidates += [
        Path(r"C:\Users\ryan.r.spencer\OneDrive - ESDC EDSC\Desktop\NOC Model\code\con_noccai_v7\label_map.py"),
        Path(r"C:\Users\ryan.r.spencer\NOC_Model\hierarchy_v7\label_map.py"),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Nothing found — return primary so validate() surfaces a clear error
    return _V7_DIR / "label_map.py"


_BUNDLED_LABEL_MAP = _find_label_map()

# ─── Model/file paths  (env-var → bundled default) ───────────────────────────
# Use set_paths() or env vars to change without editing this file.

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

# Label map is always bundled — tiny JSON, no env-var override needed.
LABEL_MAP_PATH: str = str(_BUNDLED_LABEL_MAP)

# ─── Inference defaults  (all are soft — change freely via set_defaults()) ────
# Every predict_* function accepts these as explicit call-time arguments too,
# which always win over the session default set here.

TOP_K: int              = 5      # ranked predictions per row  (1–515)
BATCH_SIZE: int         = 64     # rows per GPU forward pass
SHOW_CONFIDENCE: bool   = True   # include *_confidence columns in output
SHORT_TEXT_FLAG: bool   = True   # append short_text_flag column (< SHORT_TEXT_MIN_WORDS)
SHORT_TEXT_MIN_WORDS: int = 25   # word count below which flag is set

# Thresholds used only by print_summary() — change freely
HIGH_CONF_THRESHOLD: float = 0.80
LOW_CONF_THRESHOLD:  float = 0.40


# ─── set_defaults() — change session-wide inference defaults ─────────────────

def set_defaults(
    top_k:                int   | None = None,
    batch_size:           int   | None = None,
    show_confidence:      bool  | None = None,
    short_text_flag:      bool  | None = None,
    short_text_min_words: int   | None = None,
    high_conf_threshold:  float | None = None,
    low_conf_threshold:   float | None = None,
) -> None:
    """
    Override session-wide inference defaults.  Call once at the top of a
    notebook or script; all subsequent predict_* calls will use these values
    unless overridden per-call.

    Parameters
    ----------
    top_k : int, optional
        How many ranked NOC predictions to return per row.  1 = top prediction
        only; 5 = top-5.  Any integer 1–515 is valid.
    batch_size : int, optional
        Rows sent to the GPU in a single forward pass.  Reduce if you hit
        CUDA out-of-memory errors (try 32 or 16).
    show_confidence : bool, optional
        If False, all ``*_confidence`` and ``level_confidences`` columns are
        dropped from the output DataFrame.
    short_text_flag : bool, optional
        If False, the ``short_text_flag`` column is not added to the output.
    short_text_min_words : int, optional
        Word count below which short_text_flag is set True.  Default 25.
    high_conf_threshold : float, optional
        Confidence ≥ this value → "high confidence" bucket in print_summary().
    low_conf_threshold : float, optional
        Confidence < this value → "flag for review" bucket in print_summary().

    Examples
    --------
    Top-1 only, no confidence scores, smaller GPU batches::

        import noc_inference
        noc_inference.set_defaults(top_k=1, show_confidence=False, batch_size=32)

    Top-5, with confidence, tighter thresholds for summary::

        noc_inference.set_defaults(
            top_k=5,
            show_confidence=True,
            high_conf_threshold=0.90,
            low_conf_threshold=0.50,
        )
    """
    global TOP_K, BATCH_SIZE, SHOW_CONFIDENCE, SHORT_TEXT_FLAG
    global SHORT_TEXT_MIN_WORDS, HIGH_CONF_THRESHOLD, LOW_CONF_THRESHOLD

    if top_k                is not None:
        if not (1 <= top_k <= 515):
            raise ValueError(f"top_k must be between 1 and 515, got {top_k}")
        TOP_K = top_k

    if batch_size           is not None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be ≥ 1, got {batch_size}")
        BATCH_SIZE = batch_size

    if show_confidence      is not None:
        SHOW_CONFIDENCE = bool(show_confidence)

    if short_text_flag      is not None:
        SHORT_TEXT_FLAG = bool(short_text_flag)

    if short_text_min_words is not None:
        SHORT_TEXT_MIN_WORDS = int(short_text_min_words)

    if high_conf_threshold  is not None:
        HIGH_CONF_THRESHOLD = float(high_conf_threshold)

    if low_conf_threshold   is not None:
        LOW_CONF_THRESHOLD  = float(low_conf_threshold)


# ─── set_paths() — change model file paths ────────────────────────────────────

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
    distilbert : str, optional
        Absolute path to the distilbert-base-uncased model folder.
    taxonomy : str, optional
        Absolute path to noc_2021_merged_clean.csv.
        Defaults to the bundled copy — rarely needs changing.
    label_map : str, optional
        Absolute path to label_map.py.
        Defaults to the bundled copy — rarely needs changing.

    Examples
    --------
    Shared GoC network drive::

        import noc_inference
        noc_inference.set_paths(
            checkpoint = r"\\\\server\\share\\NOC_Model\\checkpoint_step_42952.pt",
            distilbert = r"\\\\server\\share\\NOC_Model\\distilbert-base-uncased",
        )
        predictor = noc_inference.load_predictor()

    Local install without LFS::

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


# ─── validate() and print_paths() ────────────────────────────────────────────

def validate() -> list[str]:
    """
    Check all required model files exist on disk.

    Returns
    -------
    list[str]
        Error messages, one per missing file.  Empty list = all OK.

    Example
    -------
    ::

        import noc_inference.config as cfg
        errors = cfg.validate()
        if errors:
            for e in errors:
                print("MISSING:", e)
    """
    errors: list[str] = []
    checks: dict[str, str] = {
        "CHECKPOINT_PATH (checkpoint_step_42952.pt)": CHECKPOINT_PATH,
        "BERT_MODEL_PATH (distilbert-base-uncased/)": BERT_MODEL_PATH,
        "NOC_TITLES_PATH (noc_2021_merged_clean.csv)": NOC_TITLES_PATH,
        "LABEL_MAP_PATH  (label_map.py) ":             LABEL_MAP_PATH,
    }
    for label, path in checks.items():
        if not Path(path).exists():
            errors.append(f"{label}  →  NOT FOUND: {path}")
    return errors


def print_paths() -> None:
    """
    Print the active path and default configuration.
    Run this to diagnose path issues on a new machine.

    Example
    -------
    ::

        python -c "from noc_inference.config import print_paths; print_paths()"
    """
    path_checks = {
        "Checkpoint (.pt)  ": CHECKPOINT_PATH,
        "DistilBERT folder ": BERT_MODEL_PATH,
        "NOC taxonomy CSV  ": NOC_TITLES_PATH,
        "Label map JSON    ": LABEL_MAP_PATH,
    }
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║         noc_inference — active configuration                    ║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    print("║  PATHS                                                          ║")
    for label, path in path_checks.items():
        status = "✓" if Path(path).exists() else "✗ MISSING"
        print(f"║  {status}  {label}")
        print(f"║           {path}")
        print("║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    print("║  INFERENCE DEFAULTS  (change with set_defaults())               ║")
    print(f"║    top_k                : {TOP_K:<40}║")
    print(f"║    batch_size           : {BATCH_SIZE:<40}║")
    print(f"║    show_confidence      : {str(SHOW_CONFIDENCE):<40}║")
    print(f"║    short_text_flag      : {str(SHORT_TEXT_FLAG):<40}║")
    print(f"║    short_text_min_words : {SHORT_TEXT_MIN_WORDS:<40}║")
    print(f"║    high_conf_threshold  : {HIGH_CONF_THRESHOLD:<40}║")
    print(f"║    low_conf_threshold   : {LOW_CONF_THRESHOLD:<40}║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()
