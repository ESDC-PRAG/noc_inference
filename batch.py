"""
noc_inference/batch.py
======================
Core prediction logic for hierarchy_v7 / con_noccai_v7.

Public functions
----------------
    load_predictor(force_reload)
    predict_one(description, title, predictor, top_k, show_confidence, batch_size, short_text_flag)
    predict_rows(descriptions, titles, predictor, top_k, show_confidence, batch_size, short_text_flag)
    predict_csv(input_path, output_path, desc_col, title_col, predictor, top_k, show_confidence, batch_size, short_text_flag)
    print_summary(df, high_threshold, low_threshold)

Every predict_* function accepts top_k, show_confidence, and batch_size as
explicit call-time arguments.  Omit them to use the session defaults from
config (change with noc_inference.set_defaults()).

Architecture reminder (hierarchy_v7 / con_noccai_v7):
    Backbone  : distilbert-base-uncased  (English, local_files_only=True)
    Heads     : classification_heads.level_1 ... level_5  (NOT heads.lN)
    L5 classes: 515
    EMA       : ema_state["shadow"]
    Best ckpt : checkpoint_step_42952.pt  (epoch 14, 93.98% L5 val acc)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from . import config as cfg

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy predictor singleton
# ---------------------------------------------------------------------------
_predictor_cache: Optional[object] = None


def _resolve(value, default):
    """Return value if it is not None, else default."""
    return value if value is not None else default


def _build_row(
    preds: list[dict],
    top_k: int,
    show_confidence: bool,
    word_count: int,
    short_text_flag: bool,
) -> dict:
    """
    Flatten the raw list-of-dicts from NOCPredictor into a single output row.

    Parameters
    ----------
    preds            Ranked prediction dicts from NOCPredictor (length >= top_k)
    top_k            How many ranks to include
    show_confidence  Whether to include *_confidence / level_confidences columns
    word_count       Pre-computed word count of the source text
    short_text_flag  Whether to append the short_text_flag column at all
    """
    row: dict = {}

    # Top-1 convenience block
    top1 = preds[0]
    row["top1_noc_code"]  = top1["noc_code"]
    row["top1_noc_title"] = top1["noc_title"]
    if show_confidence:
        row["top1_confidence"]   = round(top1["confidence"], 6)
        row["top1_hierarchy"]    = top1.get("hierarchy", "")
        row["level_confidences"] = top1.get("level_confidences", "")

    # Ranked top-K block
    for rank, pred in enumerate(preds[:top_k], start=1):
        row[f"pred_{rank}_noc_code"]  = pred["noc_code"]
        row[f"pred_{rank}_noc_title"] = pred["noc_title"]
        if show_confidence:
            row[f"pred_{rank}_confidence"] = round(pred["confidence"], 6)

    if short_text_flag:
        row["short_text_flag"] = word_count < cfg.SHORT_TEXT_MIN_WORDS

    return row


# ---------------------------------------------------------------------------
# load_predictor
# ---------------------------------------------------------------------------

def load_predictor(force_reload: bool = False) -> object:
    """
    Load and return the NOCPredictor, caching it for the Python session.

    The ~270 MB checkpoint is read from disk only on the first call (or when
    force_reload=True).  Subsequent calls return the cached instance instantly.

    Parameters
    ----------
    force_reload : bool
        Discard the cached predictor and reload from disk.  Use after calling
        set_paths() in a session where the model was already loaded.

    Returns
    -------
    NOCPredictor
        Ready-to-use predictor with model on GPU (or CPU if CUDA unavailable).

    Raises
    ------
    FileNotFoundError
        If any required model file is missing.
        Run ``noc_inference.config.print_paths()`` for a full diagnosis.
    """
    global _predictor_cache

    if _predictor_cache is not None and not force_reload:
        return _predictor_cache

    path_errors = cfg.validate()
    if path_errors:
        raise FileNotFoundError(
            "Cannot load predictor — missing files:\n"
            + "\n".join(f"  {e}" for e in path_errors)
            + "\n\nFix options:"
            + "\n  1. git lfs pull  (re-download LFS binaries)"
            + "\n  2. noc_inference.set_paths(checkpoint=..., distilbert=...)"
            + "\n     then load_predictor(force_reload=True)"
            + "\n  3. Set env vars NOC_CHECKPOINT_PATH and NOC_DISTILBERT_PATH"
            + "\n  4. noc_inference.config.print_paths()  for full diagnosis"
        )

    # con_noccai_v7 is installed as a sibling package after pip install —
    # no sys.path hacks are needed.
    from con_noccai_v7 import NOCPredictor  # noqa: PLC0415

    logger.info(
        "Loading hierarchy_v7 predictor ...\n"
        "  checkpoint : %s\n"
        "  backbone   : %s",
        cfg.CHECKPOINT_PATH,
        cfg.BERT_MODEL_PATH,
    )
    t0 = time.perf_counter()
    _predictor_cache = NOCPredictor(
        checkpoint_path=cfg.CHECKPOINT_PATH,
        bert_model_path=cfg.BERT_MODEL_PATH,
        label_map_path=cfg.LABEL_MAP_PATH,
        noc_titles_path=cfg.NOC_TITLES_PATH,
    )
    logger.info("Predictor ready in %.1f s", time.perf_counter() - t0)
    return _predictor_cache


# ---------------------------------------------------------------------------
# predict_one  — single string in, list of dicts out
# ---------------------------------------------------------------------------

def predict_one(
    text: str,
    title: str | None = None,
    predictor: object | None = None,
    top_k: int | None = None,
    show_confidence: bool | None = None,
    batch_size: int | None = None,
    short_text_flag: bool | None = None,
) -> list[dict]:
    """
    Predict NOC code(s) for a single job-description string.

    Parameters
    ----------
    text : str
        Raw job-description text.  HTML, URLs, and phone numbers are cleaned
        automatically by the con_noccai_v7 preprocessing pipeline.
    title : str, optional
        Job title — used as supplementary signal if the model supports it.
    predictor : NOCPredictor, optional
        Pre-loaded predictor.  Loaded and cached automatically if None.
    top_k : int, optional
        How many ranked NOC predictions to return.  1 = best prediction only.
        Default: cfg.TOP_K  (session default set via set_defaults()).
    show_confidence : bool, optional
        Include ``confidence``, ``hierarchy``, and ``level_confidences`` keys.
        Default: cfg.SHOW_CONFIDENCE.
    batch_size : int, optional
        GPU batch size.  Default: cfg.BATCH_SIZE.
        (Single prediction uses batch_size=1 internally, but this parameter
        is accepted for API consistency with predict_rows / predict_csv.)
    short_text_flag : bool, optional
        Add a ``short_text_flag`` key to each result dict.
        Default: cfg.SHORT_TEXT_FLAG.

    Returns
    -------
    list[dict]
        One dict per rank (length = top_k).  Keys always present:
            rank          integer 1 ... top_k
            noc_code      5-digit NOC 2021 string, e.g. "21232"
            noc_title     English occupational title

        Additional keys when show_confidence=True:
            confidence         softmax probability 0-1
            hierarchy          "l1=2 | l2=21 | l3=213 | l4=2132 | l5=21320"  (rank-1 only)
            level_confidences  "L1=97% | L2=93% | ..."                        (rank-1 only)

        Additional key when short_text_flag=True:
            short_text_flag    True if the input had < SHORT_TEXT_MIN_WORDS words

    Examples
    --------
    Single best prediction with confidence::

        result = predict_one(
            "Manages a team of software developers and conducts code reviews.",
            title="Software Team Lead",
            top_k=1,
        )
        print(result[0]["noc_code"], result[0]["noc_title"], result[0]["confidence"])

    Top-5, no confidence::

        results = predict_one(
            "Administers medications and monitors vital signs in a hospital unit.",
            top_k=5,
            show_confidence=False,
        )
        for r in results:
            print(r["rank"], r["noc_code"], r["noc_title"])
    """
    _top_k     = _resolve(top_k,          cfg.TOP_K)
    _show_conf = _resolve(show_confidence, cfg.SHOW_CONFIDENCE)
    _bsize     = _resolve(batch_size,      cfg.BATCH_SIZE)
    _stflag    = _resolve(short_text_flag, cfg.SHORT_TEXT_FLAG)

    if predictor is None:
        predictor = load_predictor()

    word_count = len(str(text).split())

    raw   = predictor.predict_batch(
        descriptions=[text],
        titles=[title],
        top_k=_top_k,
        batch_size=_bsize,
    )
    preds = raw[0]  # single-item batch

    output = []
    for rank, pred in enumerate(preds[:_top_k], start=1):
        entry: dict = {
            "rank":      rank,
            "noc_code":  pred["noc_code"],
            "noc_title": pred["noc_title"],
        }
        if _show_conf:
            entry["confidence"] = round(pred["confidence"], 6)
            if rank == 1:
                entry["hierarchy"]         = pred.get("hierarchy", "")
                entry["level_confidences"] = pred.get("level_confidences", "")
        if _stflag:
            entry["short_text_flag"] = word_count < cfg.SHORT_TEXT_MIN_WORDS
        output.append(entry)

    return output


# ---------------------------------------------------------------------------
# predict_rows  — list of strings in, DataFrame out
# ---------------------------------------------------------------------------

def predict_rows(
    descriptions: list[str],
    titles: list[str | None] | None = None,
    predictor: object | None = None,
    top_k: int | None = None,
    show_confidence: bool | None = None,
    batch_size: int | None = None,
    short_text_flag: bool | None = None,
) -> pd.DataFrame:
    """
    Predict NOC codes for a list of job-description strings.

    Parameters
    ----------
    descriptions : list[str]
        Raw job-description text, one string per row.
    titles : list[str | None], optional
        Job titles, same length as descriptions.  None entries are fine.
    predictor : NOCPredictor, optional
        Pre-loaded predictor.  Loaded and cached automatically if None.
    top_k : int, optional
        Ranked predictions per row.  Default: cfg.TOP_K.
        Use top_k=1 for a lightweight single-best output.
    show_confidence : bool, optional
        Include *_confidence and level_confidences columns.
        Default: cfg.SHOW_CONFIDENCE.
    batch_size : int, optional
        Rows per GPU forward pass.  Default: cfg.BATCH_SIZE (64).
        Reduce to 32 or 16 on machines with less VRAM.
    short_text_flag : bool, optional
        Append short_text_flag column.  Default: cfg.SHORT_TEXT_FLAG.

    Returns
    -------
    pd.DataFrame
        One row per input.  Columns depend on top_k / show_confidence:

        Always present:
            top1_noc_code, top1_noc_title
            pred_1_noc_code ... pred_N_noc_code   (N = top_k)
            pred_1_noc_title ... pred_N_noc_title

        When show_confidence=True (default):
            top1_confidence, top1_hierarchy, level_confidences
            pred_1_confidence ... pred_N_confidence

        When short_text_flag=True (default):
            short_text_flag

    Examples
    --------
    Default (top-5, with confidence)::

        df = predict_rows(["Software developer...", "Registered nurse..."])

    Top-1, no confidence, no short-text column::

        df = predict_rows(
            descriptions,
            top_k=1,
            show_confidence=False,
            short_text_flag=False,
        )
    """
    if not descriptions:
        return pd.DataFrame()

    _top_k     = _resolve(top_k,          cfg.TOP_K)
    _show_conf = _resolve(show_confidence, cfg.SHOW_CONFIDENCE)
    _bsize     = _resolve(batch_size,      cfg.BATCH_SIZE)
    _stflag    = _resolve(short_text_flag, cfg.SHORT_TEXT_FLAG)

    if predictor is None:
        predictor = load_predictor()

    if titles is None:
        titles = [None] * len(descriptions)

    if len(titles) != len(descriptions):
        raise ValueError(
            f"Length mismatch: {len(descriptions)} descriptions but {len(titles)} titles."
        )

    word_counts = [len(str(d).split()) for d in descriptions]

    logger.info(
        "predict_rows: %d rows | top_k=%d | show_confidence=%s | batch_size=%d",
        len(descriptions), _top_k, _show_conf, _bsize,
    )

    results = predictor.predict_batch(
        descriptions=descriptions,
        titles=titles,
        top_k=_top_k,
        batch_size=_bsize,
    )

    records = [
        _build_row(preds, _top_k, _show_conf, wc, _stflag)
        for preds, wc in zip(results, word_counts)
    ]
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# predict_csv  — CSV file in, enriched DataFrame + optional CSV file out
# ---------------------------------------------------------------------------

def predict_csv(
    input_path: str | Path,
    output_path: str | Path | None = None,
    desc_col: str = "DESCRIPTION_EN",
    title_col: str | None = "TITLE_EN",
    predictor: object | None = None,
    top_k: int | None = None,
    show_confidence: bool | None = None,
    batch_size: int | None = None,
    short_text_flag: bool | None = None,
) -> pd.DataFrame:
    """
    Read a CSV, predict NOC codes for every row, optionally write results.

    Parameters
    ----------
    input_path : str | Path
        Path to the input CSV.  Must contain a column named desc_col.
    output_path : str | Path, optional
        Write the enriched DataFrame here as UTF-8 CSV.
        Omit to keep results in memory only.
    desc_col : str
        Column with job-description text.
        Default: "DESCRIPTION_EN"  (WICAR / CSJ training convention).
    title_col : str | None
        Optional job-title column.  Set None if absent.  Default: "TITLE_EN".
    predictor : NOCPredictor, optional
        Pre-loaded predictor.  Loaded automatically if None.
    top_k : int, optional
        Ranked predictions per row.  Default: cfg.TOP_K.
    show_confidence : bool, optional
        Include confidence columns.  Default: cfg.SHOW_CONFIDENCE.
    batch_size : int, optional
        GPU batch size.  Default: cfg.BATCH_SIZE.
    short_text_flag : bool, optional
        Append short_text_flag column.  Default: cfg.SHORT_TEXT_FLAG.

    Returns
    -------
    pd.DataFrame
        Original CSV columns merged with prediction columns.

    Raises
    ------
    FileNotFoundError   If input_path does not exist.
    KeyError            If desc_col is not in the CSV.

    Examples
    --------
    Full run — top-5, confidence on, write output file::

        df = predict_csv(
            r"C:\data\jobs.csv",
            r"C:\data\jobs_noc.csv",
        )

    Top-1 only, no confidence, no output file::

        df = predict_csv(
            r"C:\data\jobs.csv",
            desc_col="Job_Description",
            title_col=None,
            top_k=1,
            show_confidence=False,
            short_text_flag=False,
        )
        print(df[["Job_Description", "top1_noc_code", "top1_noc_title"]].head())
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    logger.info("Reading %s ...", input_path)
    df = pd.read_csv(input_path, low_memory=False)

    if desc_col not in df.columns:
        raise KeyError(
            f"Column '{desc_col}' not found in {input_path.name}.\n"
            f"Available columns: {', '.join(df.columns.tolist())}\n"
            f"Pass the correct name as desc_col='...' to predict_csv()."
        )

    texts  = df[desc_col].fillna("").astype(str).tolist()
    titles_list: list[str | None] = (
        df[title_col].fillna("").astype(str).tolist()
        if (title_col and title_col in df.columns)
        else None
    )

    _top_k     = _resolve(top_k,          cfg.TOP_K)
    _show_conf = _resolve(show_confidence, cfg.SHOW_CONFIDENCE)
    _bsize     = _resolve(batch_size,      cfg.BATCH_SIZE)
    _stflag    = _resolve(short_text_flag, cfg.SHORT_TEXT_FLAG)

    logger.info(
        "Running inference on %d rows | top_k=%d | show_confidence=%s | batch_size=%d",
        len(df), _top_k, _show_conf, _bsize,
    )

    pred_df = predict_rows(
        descriptions=texts,
        titles=titles_list,
        predictor=predictor,
        top_k=_top_k,
        show_confidence=_show_conf,
        batch_size=_bsize,
        short_text_flag=_stflag,
    )

    out_df = pd.concat(
        [df.reset_index(drop=True), pred_df.reset_index(drop=True)],
        axis=1,
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        logger.info("Predictions written to %s  (%d rows)", output_path, len(out_df))

    return out_df


# ---------------------------------------------------------------------------
# print_summary
# ---------------------------------------------------------------------------

def print_summary(
    df: pd.DataFrame,
    high_threshold: float | None = None,
    low_threshold:  float | None = None,
) -> None:
    """
    Print a confidence-band summary to stdout.

    Parameters
    ----------
    df : pd.DataFrame
        Output of predict_csv() or predict_rows().  Must contain
        ``top1_confidence`` (only present when show_confidence=True).
    high_threshold : float, optional
        Override the high-confidence cutoff for this summary only.
        Default: cfg.HIGH_CONF_THRESHOLD.
    low_threshold : float, optional
        Override the flag-for-review cutoff for this summary only.
        Default: cfg.LOW_CONF_THRESHOLD.

    Notes
    -----
    If ``top1_confidence`` is not in df (i.e. show_confidence=False was used),
    a warning is printed and the function returns without raising an error.

    Examples
    --------
    Default thresholds::

        print_summary(df)

    Custom thresholds for a stricter review::

        print_summary(df, high_threshold=0.90, low_threshold=0.60)
    """
    if "top1_confidence" not in df.columns:
        logger.warning(
            "print_summary: 'top1_confidence' column not found. "
            "Re-run predict_* with show_confidence=True to enable this report."
        )
        return

    _high = _resolve(high_threshold, cfg.HIGH_CONF_THRESHOLD)
    _low  = _resolve(low_threshold,  cfg.LOW_CONF_THRESHOLD)

    confs   = df["top1_confidence"].astype(float)
    n       = len(confs)
    high    = (confs >= _high).sum()
    low     = (confs <  _low).sum()
    mid     = n - high - low
    flagged = (
        int(df["short_text_flag"].sum())
        if "short_text_flag" in df.columns
        else "n/a (short_text_flag=False)"
    )
    top_k_cols = [c for c in df.columns if c.startswith("pred_") and c.endswith("_noc_code")]
    top_k_used = len(top_k_cols)

    print()
    print("+-----------------------------------------------------------------+")
    print("|         Top-1 Confidence Summary  (hierarchy_v7)               |")
    print("+-----------------------------------------------------------------+")
    print(f"|  Rows predicted       : {n:<40}|")
    print(f"|  Top-K returned       : {top_k_used:<40}|")
    print(f"|  Mean confidence      : {confs.mean():<40.2%}|")
    print(f"|  Median confidence    : {confs.median():<40.2%}|")
    print(f"|  Min / Max            : {confs.min():.2%} / {confs.max():.2%}{'':<28}|")
    print("+-----------------------------------------------------------------+")
    print(f"|  >= {_high:.0%}  high conf     : {high:<5} rows  ({high/n:.1%}){'':<20}|")
    print(f"|  {_low:.0%}-{_high:.0%}  medium       : {mid:<5} rows  ({mid/n:.1%}){'':<20}|")
    print(f"|  <  {_low:.0%}  flag/review   : {low:<5} rows  ({low/n:.1%}){'':<20}|")
    print("+-----------------------------------------------------------------+")
    print(f"|  Short-text flagged   : {str(flagged):<40}|")
    print("|  (< 25 words = low occupational specificity for 515 classes)    |")
    print("+-----------------------------------------------------------------+")
    print()
