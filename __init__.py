"""
noc_inference
=============
Thin wrapper around con_noccai_v7 — hierarchy_v7 (DistilBERT-base-uncased, 515 L5 classes).

Quick start
-----------

Single prediction::

    import noc_inference
    result = noc_inference.predict_one(
        "Manages a team of software developers and conducts code reviews.",
        title="Software Team Lead",
        top_k=1,
    )
    print(result[0]["noc_code"], result[0]["noc_title"])

Batch — list of strings::

    df = noc_inference.predict_rows(
        descriptions=["Software developer...", "Registered nurse..."],
        top_k=5,
        show_confidence=True,
    )

Batch — CSV file::

    df = noc_inference.predict_csv(
        input_path=r"C:\data\jobs.csv",
        output_path=r"C:\data\jobs_noc.csv",
        desc_col="DESCRIPTION_EN",
        title_col="TITLE_EN",
        top_k=5,
        show_confidence=True,
    )
    noc_inference.print_summary(df)

Change session-wide defaults once::

    noc_inference.set_defaults(top_k=1, show_confidence=False, batch_size=32)

Diagnose path issues::

    noc_inference.config.print_paths()

Public API
----------
Prediction functions:
    predict_one(text, title, predictor, top_k, show_confidence, batch_size, short_text_flag)
        -> list[dict]   one prediction per rank

    predict_rows(descriptions, titles, predictor, top_k, show_confidence, batch_size, short_text_flag)
        -> pd.DataFrame   one row per input description

    predict_csv(input_path, output_path, desc_col, title_col, predictor, top_k,
                show_confidence, batch_size, short_text_flag)
        -> pd.DataFrame   original CSV merged with prediction columns

    print_summary(df, high_threshold, low_threshold)
        -> None   prints confidence-band report to stdout

Model loading:
    load_predictor(force_reload=False)
        -> NOCPredictor   cached — only loads from disk on first call

Configuration:
    set_defaults(top_k, batch_size, show_confidence, short_text_flag,
                 short_text_min_words, high_conf_threshold, low_conf_threshold)
        -> None   override session-wide inference defaults

    set_paths(checkpoint, distilbert, taxonomy, label_map)
        -> None   override model file paths (call before load_predictor)

    config.print_paths()
        -> None   print active paths + defaults; use to diagnose missing files

    config.validate()
        -> list[str]   empty = all files found; non-empty = list of missing paths
"""

from .batch import (
    load_predictor,
    predict_one,
    predict_rows,
    predict_csv,
    print_summary,
)
from .config import set_defaults, set_paths
from . import config

__all__ = [
    # Prediction
    "predict_one",
    "predict_rows",
    "predict_csv",
    "print_summary",
    # Model loading
    "load_predictor",
    # Configuration
    "set_defaults",
    "set_paths",
    "config",
]

__version__ = "0.7.0"