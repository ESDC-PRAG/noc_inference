"""
con_noccai_v7/label_map.py

Loads hierarchy_v7/label_map.json and provides all index <-> code lookups
needed at inference time.

Confirmed JSON structure (from inspection of the real file)
-----------------------------------------------------------
{
  "num_classes_per_level": {"1": 10, "2": 45, "3": 89, "4": 162, "5": 515},
  "level_to_idx": {
    "1": {"0": 0, ...},   "2": {"00": 0, ...},   "3": {"000": 0, ...},
    "4": {"0001": 0, ...}, "5": {"00010": 0, ..., "95109": 514}
  },
  "idx_to_level": {
    "1": {"0": "0", ...}, "2": {"0": "00", ...}, "3": {"0": "000", ...},
    "4": {"0": "0001", ...}, "5": {"0": "00010", ..., "514": "95109"}
  }
}

Hierarchy chain
---------------
NOC 2021 codes are strictly hierarchical by prefix truncation.
Verified against the real file: 0 broken chains across all 515 L5 codes.
    L5 "21234" -> L4 "2123" -> L3 "213" -> L2 "21" -> L1 "2"
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class LabelMap:
    """
    Index <-> NOC code lookup for all five hierarchy levels.

    Attributes
    ----------
    idx_to_code[level] : list[str]
        Position i -> NOC code string.  e.g. idx_to_code[5][0] == '00010'
    code_to_idx[level] : dict[str, int]
        NOC code string -> classifier output index.
    code_to_title : dict[str, str]
        5-digit NOC code -> English occupational title.
    parent_chain[noc5] : dict[str, str]
        e.g. {'l1': '2', 'l2': '21', 'l3': '213', 'l4': '2123', 'l5': '21234'}
    num_classes : dict[int, int]
        Number of classes per level.
    """

    def __init__(
        self,
        label_map_path,
        noc_titles_path=None,
    ):
        self._label_map_path = Path(label_map_path)
        self._noc_titles_path = Path(noc_titles_path) if noc_titles_path else None

        self.idx_to_code: Dict[int, List[str]] = {}
        self.code_to_idx: Dict[int, Dict[str, int]] = {}
        self.code_to_title: Dict[str, str] = {}
        self.parent_chain: Dict[str, Dict[str, str]] = {}
        self.num_classes: Dict[int, int] = {}

        self._load()

    def _load(self):
        if not self._label_map_path.exists():
            raise FileNotFoundError(
                f"label_map.json not found: {self._label_map_path}\n"
                "Check Config.LABEL_MAP_PATH in hierarchy_v7/config.py."
            )

        with open(self._label_map_path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)

        required = {"num_classes_per_level", "level_to_idx", "idx_to_level"}
        missing = required - set(raw.keys())
        if missing:
            raise KeyError(
                f"label_map.json missing expected top-level keys: {missing}\n"
                f"Keys found: {list(raw.keys())}"
            )

        # num_classes
        self.num_classes = {int(k): int(v) for k, v in raw["num_classes_per_level"].items()}
        logger.info(
            "label_map — classes per level: %s",
            {f"L{k}": v for k, v in sorted(self.num_classes.items())},
        )

        # idx_to_code and code_to_idx for each level
        idx_to_level = raw["idx_to_level"]
        level_to_idx = raw["level_to_idx"]

        for lvl in range(1, 6):
            key = str(lvl)
            if key not in idx_to_level:
                raise KeyError(f"idx_to_level is missing level key '{key}'")

            rev = idx_to_level[key]           # {"0": "00010", "1": "00011", ...}
            n = len(rev)
            codes: List[str] = [""] * n
            for str_idx, code in rev.items():
                codes[int(str_idx)] = str(code)

            self.idx_to_code[lvl] = codes
            self.code_to_idx[lvl] = {str(c): int(i) for c, i in level_to_idx[key].items()}
            logger.info("  L%d: %d codes  (first=%s, last=%s)", lvl, n, codes[0], codes[-1])

        # NOC titles
        if self._noc_titles_path and self._noc_titles_path.exists():
            self._load_titles()
        elif self._noc_titles_path:
            logger.warning(
                "NOC titles CSV not found at %s — predictions will show codes only.",
                self._noc_titles_path,
            )

        # Hierarchy chains
        self._build_parent_chains()

    def _load_titles(self):
        df = pd.read_csv(self._noc_titles_path, dtype=str)
        df.columns = df.columns.str.strip()

        code_col  = _find_col(df, ["NOC_Code", "NOCCode", "noc_code", "Code", "code"])
        title_col = _find_col(df, ["Level5_Name", "level5_name", "Title", "title", "Name", "name"])

        if code_col is None or title_col is None:
            logger.warning(
                "Could not identify code/title columns in %s (columns found: %s)",
                self._noc_titles_path,
                list(df.columns),
            )
            return

        loaded = 0
        for _, row in df.iterrows():
            try:
                # Handles "10010.0", "10010", 10010, etc.
                code = str(int(float(str(row[code_col])))).zfill(5)
                self.code_to_title[code] = str(row[title_col]).strip()
                loaded += 1
            except (ValueError, TypeError):
                continue

        logger.info("Loaded %d NOC titles from %s", loaded, self._noc_titles_path)

    def _build_parent_chains(self):
        for code5 in self.idx_to_code.get(5, []):
            if len(code5) == 5:
                self.parent_chain[code5] = {
                    "l1": code5[0],
                    "l2": code5[:2],
                    "l3": code5[:3],
                    "l4": code5[:4],
                    "l5": code5,
                }

    # Public API

    def idx_to_noc5(self, idx: int) -> str:
        """Classifier index -> 5-digit NOC code (e.g. 0 -> '00010')."""
        try:
            return self.idx_to_code[5][idx]
        except IndexError:
            return f"UNKNOWN_IDX_{idx}"

    def get_title(self, noc5: str) -> str:
        """English title for a 5-digit NOC code, or the code itself if not found."""
        return self.code_to_title.get(noc5, noc5)

    def get_hierarchy(self, noc5: str) -> Dict[str, str]:
        """Full 5-level hierarchy dict for a given L5 code."""
        return self.parent_chain.get(noc5, {"l5": noc5})

    def __repr__(self) -> str:
        return (
            f"LabelMap(L5={self.num_classes.get(5, '?')} classes, "
            f"titles={len(self.code_to_title)})"
        )


def _find_col(df, candidates):
    for name in candidates:
        if name in df.columns:
            return name
    return None
