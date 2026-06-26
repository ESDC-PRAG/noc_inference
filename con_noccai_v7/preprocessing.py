"""
con_noccai_v7/preprocessing.py

Text cleaning for inference — mirrors the cleaning applied during V7 training.

Key differences from the old multilingual model:
  • DistilBERT-base-uncased tokeniser handles lowercasing internally;
    we do NOT lowercase here so casing information is preserved until
    the tokeniser subword-splits it.
  • Word dropout and chunking were training-only augmentations — not applied here.
  • Title composition follows TITLE_MODE_FULL (the dominant training mode,
    60 % of examples): "{title}: {description}" when both are present.
"""

from __future__ import annotations

import html
import re
import unicodedata
from typing import Optional


# Pre-compiled patterns for speed during batch processing
_RE_HTML_TAG   = re.compile(r"<[^>]+>")
_RE_URL        = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_RE_EMAIL      = re.compile(r"\S+@\S+\.\S+")
_RE_WHITESPACE = re.compile(r"\s+")
_RE_CTRL       = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_text(text: str) -> str:
    """
    Apply the same cleaning as the V7 training pipeline.

    Steps
    -----
    1. Guard against non-string input (return empty string).
    2. Strip HTML tags and unescape HTML entities (&amp; &lt; …).
    3. Remove control characters (non-printable ASCII).
    4. Normalise Unicode to NFKC so ligatures, accented chars, etc. are
       consistent (NFKC rather than NFKD so we keep non-ASCII characters).
    5. Remove bare URLs and email addresses.
    6. Collapse runs of whitespace (spaces, tabs, newlines) to a single space.
    7. Strip leading/trailing whitespace.

    Note: we do *not* lowercase because DistilBERT-base-uncased does its own
    lowercasing in the tokeniser.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. HTML
    text = _RE_HTML_TAG.sub(" ", text)
    text = html.unescape(text)

    # 2. Control characters
    text = _RE_CTRL.sub(" ", text)

    # 3. Unicode normalisation
    text = unicodedata.normalize("NFKC", text)

    # 4. URLs and emails
    text = _RE_URL.sub(" ", text)
    text = _RE_EMAIL.sub(" ", text)

    # 5. Whitespace
    text = _RE_WHITESPACE.sub(" ", text).strip()

    return text


def build_input_text(
    description: str,
    title: Optional[str] = None,
    sep: str = ": ",
) -> str:
    """
    Compose the final string fed to the tokeniser.

    Mirrors the dominant training mode (TITLE_MODE_FULL, 60 % of examples):
        "<title>: <description>"

    If title is absent or empty after cleaning, only the description is used.

    Parameters
    ----------
    description : str
        Raw job description text.
    title : str, optional
        Job title / posting title.
    sep : str
        Separator placed between title and description. Default ": ".

    Returns
    -------
    str
        Cleaned, composed input text ready for the tokeniser.
    """
    desc_clean  = clean_text(description)
    title_clean = clean_text(title) if title else ""

    if title_clean and desc_clean:
        return title_clean + sep + desc_clean
    if desc_clean:
        return desc_clean
    if title_clean:
        return title_clean
    return ""
