"""
noc_inference/batch.py
======================
Core batch prediction logic for hierarchy_v7 / con_noccai_v7.

This module is the engine behind the public API in __init__.py.
After `pip install git+https://gccode.ssc-spc.gc.ca/ryan.r.spencer/con_noccai.git`
both `noc_inference` and `con_noccai_v7` are installed packages — no sys.path
manipulation is needed.

Public functions
----------------
    load_predictor(force_reload=False) -> NOCPredictor
    predict_rows(descriptions, titles, predictor, top_k) -> pd.DataFrame
    predict_csv(input_path, output_path, desc_col, title_col, predictor, top_k) -> pd.DataFrame
    print_summary(df) -> None

Architecture reminder (hierarchy_v7 / con_noccai_v7):
    Backbone  : distilbert-base-uncased  (English, GoC network: local_files_only=True)
    Heads     : classification_heads.level_1 … level_5  (NOT heads.lN)
    L5 classes: 515
    EMA       : ema_state['shadow']  (preferred over model_state_dict)
    Best ckpt : checkpoint_step_42952.pt  (epoch 14, 93.98% L5 val acc, in-distribution)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from . import config as cfg

logger = logging.getLogger(__name__)


# ─── Lazy predictor singleton ─────────────────────────────────────────────────
# Keeps the loaded model in memory between calls within the same Python session.
# Avoids the ~10–20 s reload cost on every predict_csv() call.
_predictor_cache: Optional[object] = None


def load_predictor(force_reload: bool = False) -> object:
    """
    Load and return the NOCPredictor, using a module-level cache so the
    ~270 MB checkpoint is only loaded once per Python session.

    Parameters
    ----------
    force_reload : bool
        If True, discard the cached predictor and reload from disk.
        Use this if you called set_paths() after a previous load.

    Returns
    -------
    NOCPredictor
        Ready-to-use predictor instance with model loaded onto GPU (or CPU).

    Raises
    ------
    FileNotFoundError
        If CHECKPOINT_PATH or BERT_MODEL_PATH do not exist.
        Run `noc_inference.config.print_paths()` to diagnose.
    """
    global _predictor_cache

    if _predictor_cache is not None and not force_reload:
        return _predictor_cache

    # ── Sanity-check paths before importing the heavy model ──────────────────
    path_errors = cfg.validate()
    if path_errors:
        msg = (
            "Cannot load predictor — one or more required files are missing:\n"
            + "\n".join(f"  {e}" for e in path_errors)
            + "\n\nFix options:"
            + "\n  1. Run git lfs pull inside the repo to download LFS binaries."
            + "\n  2. Call noc_inference.set_paths(checkpoint=..., distilbert=...) "
            +   "before load_predictor()."
            + "\n  3. Set env vars NOC_CHECKPOINT_PATH and NOC_DISTILBERT_PATH."
            + "\n  4. Run: python -c \"from noc_inference.config import print_paths; "
            +   "print_paths()\"  for a full diagnosis."
        )
        raise FileNotFoundError(msg)

    # ── Import con_noccai_v7 (installed as a sibling package by pip) ─────────
    # No sys.path hacks needed after pip install — both packages are on sys.path.
    from con_noccai_v7 import NOCPredictor  # noqa: PLC0415

    logger.info(
        "Loading hierarchy_v7 predictor …\n"
        "  checkpoint : %s\n"
        "  backbone   : %s",
        cfg.CHECKPOINT_PATH,
        cfg.BERT_MODEL_PATH,
    )
    t0 = time.perf_counter()

    _predictor_cache = NOCPredictor(
        checkpoint_path = cfg.CHECKPOINT_PATH,
        bert_model_path = cfg.BERT_MODEL_PATH,
        label_map_path  = cfg.LABEL_MAP_PATH,
        noc_titles_path = cfg.NOC_TITLES_PATH,
    )

    elapsed = time.perf_counter() - t0
    logger.info("Predictor ready in %.1f s", elapsed)
    return _predictor_cache


# ─── predict_rows ─────────────────────────────────────────────────────────────

def predict_rows(
    descriptions: list[str],
    titles:       list[str | None] | None = None,
    predictor:    object | None = None,
    top_k:        int = cfg.TOP_K,
) -> pd.DataFrame:
    """
    Predict NOC codes for a list of job-description strings.

    Parameters
    ----------
    descriptions : list[str]
        Raw job-description text.  HTML, URLs, and phone numbers are cleaned
        automatically inside con_noccai_v7's preprocessing pipeline.
    titles : list[str | None], optional
        Corresponding job titles (same length as descriptions).  Pass None or
        omit if no titles are available.
    predictor : NOCPredictor, optional
        Pre-loaded predictor.  If None, load_predictor() is called automatically
        (first call takes ~15–20 s; subsequent calls return immediately from cache).
    top_k : int
        Number of ranked NOC predictions to return per row (default 5).

    Returns
    -------
    pd.DataFrame
        One row per input description with columns:
            top1_noc_code, top1_noc_title, top1_confidence
            top1_hierarchy        e.g. "l1=2 | l2=21 | l3=213 | l4=2132 | l5=21320"
            level_confidences     e.g. "L1=98.2% | L2=95.1% | L3=89.4% | L4=82.3% | L5=71.6%"
            pred_1_noc_code … pred_5_noc_code
            pred_1_noc_title … pred_5_noc_title
            pred_1_confidence … pred_5_confidence
            short_text_flag       True if description < 25 words (low reliability)
    """
    if not descriptions:
        return pd.DataFrame()

    if predictor is None:
        predictor = load_predictor()

    if titles is None:
        titles = [None] * len(descriptions)

    if len(titles) != len(descriptions):
        raise ValueError(
            f"lengths mismatch: {len(descriptions)} descriptions but {len(titles)} titles"
        )

    # ── Short-text flagging ───────────────────────────────────────────────────
    # DistilBERT's practical useful limit is ~380 words (~512 tokens).
    # Descriptions under 25 words lack sufficient occupational specificity for
    # reliable 515-class classification.
    word_counts = [len(str(d).split()) for d in descriptions]

    # ── Run batch inference ───────────────────────────────────────────────────
    results = predictor.predict_batch(
        descriptions = descriptions,
        titles       = titles,
        top_k        = top_k,
        batch_size   = cfg.BATCH_SIZE,
    )

    # ── Assemble output DataFrame ─────────────────────────────────────────────
    records = []
    for i, (preds, wc) in enumerate(zip(results, word_counts)):
        row: dict = {}

        # Top-1 convenience columns
        top1 = preds[0]
        row["top1_noc_code"]     = top1["noc_code"]
        row["top1_noc_title"]    = top1["noc_title"]
        row["top1_confidence"]   = round(top1["confidence"], 6)
        row["top1_hierarchy"]    = top1.get("hierarchy", "")
        row["level_confidences"] = top1.get("level_confidences", "")

        # All top-K ranked predictions
        for rank, pred in enumerate(preds, start=1):
            row[f"pred_{rank}_noc_code"]    = pred["noc_code"]
            row[f"pred_{rank}_noc_title"]   = pred["noc_title"]
            row[f"pred_{rank}_confidence"]  = round(pred["confidence"], 6)

        row["short_text_flag"] = wc < cfg.SHORT_TEXT_MIN_WORDS
        records.append(row)

    return pd.DataFrame(records)


# ─── predict_csv ──────────────────────────────────────────────────────────────

def predict_csv(
    input_path:  str | Path,
    output_path: str | Path | None = None,
    desc_col:    str = "DESCRIPTION_EN",
    title_col:   str | None = "TITLE_EN",
    predictor:   object | None = None,
    top_k:       int = cfg.TOP_K,
) -> pd.DataFrame:
    """
    Read a CSV, predict NOC codes for every row, and optionally write results.

    Parameters
    ----------
    input_path : str | Path
        Path to the input CSV.  The file must contain a column named `desc_col`.
    output_path : str | Path, optional
        If provided, the enriched DataFrame is written to this path as CSV.
        If None, results are returned in memory only.
    desc_col : str
        Name of the column containing job-description text.
        Default: "DESCRIPTION_EN" (matches WICAR / CSJ training data convention).
    title_col : str | None
        Name of the optional job-title column.  Set to None if not available.
        Default: "TITLE_EN".
    predictor : NOCPredictor, optional
        Pre-loaded predictor.  Loaded automatically if None.
    top_k : int
        Number of ranked predictions per row.

    Returns
    -------
    pd.DataFrame
        Original CSV columns, plus all prediction columns from predict_rows().

    Raises
    ------
    FileNotFoundError
        If input_path does not exist.
    KeyError
        If desc_col is not found in the CSV.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    logger.info("Reading %s …", input_path)
    df = pd.read_csv(input_path, low_memory=False)

    if desc_col not in df.columns:
        available = ", ".join(df.columns.tolist())
        raise KeyError(
            f"Column '{desc_col}' not found in {input_path.name}.\n"
            f"Available columns: {available}\n"
            f"Pass the correct column name as desc_col='...' or "
            f"title_col='...' to predict_csv()."
        )

    descriptions = df[desc_col].fillna("").astype(str).tolist()
    titles: list[str | None] = (
        df[title_col].fillna("").astype(str).tolist()
        if (title_col and title_col in df.columns)
        else None
    )

    logger.info("Running inference on %d rows (GPU batch_size=%d) …", len(df), cfg.BATCH_SIZE)
    pred_df = predict_rows(descriptions, titles, predictor=predictor, top_k=top_k)

    # Merge predictions back alongside original columns
    out_df = pd.concat([df.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        logger.info("Predictions written to %s  (%d rows)", output_path, len(out_df))

    return out_df


# ─── print_summary ────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame) -> None:
    """
    Print a concise confidence-band summary to stdout.

    Parameters
    ----------
    df : pd.DataFrame
        Output of predict_csv() or predict_rows().  Must contain 'top1_confidence'.
    """
    if "top1_confidence" not in df.columns:
        logger.warning("print_summary: 'top1_confidence' column not found — skipping.")
        return

    confs  = df["top1_confidence"].astype(float)
    n      = len(confs)
    high   = (confs >= cfg.HIGH_CONF_THRESHOLD).sum()
    low    = (confs <  cfg.LOW_CONF_THRESHOLD).sum()
    mid    = n - high - low
    flagged = df["short_text_flag"].sum() if "short_text_flag" in df.columns else "n/a"

    print()
    print("┌─────────────────────────────────────────────────────────────┐")
    print("│            Top-1 Confidence Summary  (hierarchy_v7)        │")
    print("├─────────────────────────────────────────────────────────────┤")
    print(f"│  Rows predicted     : {n:<38}│")
    print(f"│  Mean confidence    : {confs.mean():<38.2%}│")
    print(f"│  Median confidence  : {confs.median():<38.2%}│")
    print(f"│  Min / Max          : {confs.min():.2%} / {confs.max():.2%}{'':<26}│")
    print("├─────────────────────────────────────────────────────────────┤")
    print(f"│  ≥ {cfg.HIGH_CONF_THRESHOLD:.0%}  high conf   : {high:<5} rows  ({high/n:.1%}){'':<18}│")
    print(f"│  {cfg.LOW_CONF_THRESHOLD:.0%}–{cfg.HIGH_CONF_THRESHOLD:.0%}  medium      : {mid:<5} rows  ({mid/n:.1%}){'':<18}│")
    print(f"│  < {cfg.LOW_CONF_THRESHOLD:.0%}  flag/review  : {low:<5} rows  ({low/n:.1%}){'':<18}│")
    print("├─────────────────────────────────────────────────────────────┤")
    print(f"│  Short-text flagged : {str(flagged):<38}│")
    print("│  (< 25 words — low occupational specificity for 515 classes)│")
    print("└─────────────────────────────────────────────────────────────┘")
    print()
