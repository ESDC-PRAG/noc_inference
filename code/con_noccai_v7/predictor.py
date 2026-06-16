"""
con_noccai_v7/predictor.py

Primary inference interface for the hierarchy_v7 NOC classifier.

Quick start
-----------
    from con_noccai_v7 import NOCPredictor

    predictor = NOCPredictor(
        checkpoint_path = r"C:\\...\\NOC_Model\\checkpoints\\best_step_XXXXX.pt",
        bert_model_path = r"C:\\...\\NOC Model\\models\\distilbert-base-uncased",
        label_map_path  = r"C:\\...\\hierarchy_v7\\label_map.json",
        noc_titles_path = r"C:\\...\\training data\\noc_2021_merged_clean.csv",
    )

    results = predictor.predict(
        description = "Develops and maintains Python microservices ...",
        title       = "Senior Software Developer",
        top_k       = 5,
    )
    for r in results:
        print(r["rank"], r["noc_code"], r["noc_title"], f"{r['confidence']:.1%}")
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from .label_map import LabelMap
from .model import HierarchicalNOCModel, load_checkpoint
from .preprocessing import build_input_text

logger = logging.getLogger(__name__)


# ── Output type alias ──────────────────────────────────────────────────────────
Prediction = Dict[str, Any]
"""
Single ranked prediction:
{
    "rank":          1,                          # 1-based rank
    "noc_code":      "21234",                    # 5-digit NOC 2021 code
    "noc_title":     "Software engineers",       # English occupational title
    "confidence":    0.876,                      # softmax probability at L5
    "hierarchy": {                               # full NOC hierarchy codes
        "l1": "2",
        "l2": "21",
        "l3": "213",
        "l4": "2123",
        "l5": "21234"
    },
    "level_confidences": {                       # per-level top-1 probability
        "l1": 0.97, "l2": 0.93, "l3": 0.91,
        "l4": 0.89, "l5": 0.876
    }
}
"""


class NOCPredictor:
    """
    Inference wrapper for the hierarchy_v7 NOC 2021 classifier.

    The model produces logits at five NOC hierarchy levels simultaneously.
    For inference we primarily report L5 (unit group, 5-digit) predictions,
    but all five levels' top-1 confidences are returned for diagnostics.

    Parameters
    ----------
    checkpoint_path : str | Path
        Path to a .pt file saved by hierarchy_v7/train.py.
        EMA weights are preferred automatically if present.
    bert_model_path : str | Path
        Local directory of distilbert-base-uncased (vocab.txt, config.json, …).
    label_map_path : str | Path
        Path to hierarchy_v7/label_map.json created during training data prep.
    noc_titles_path : str | Path, optional
        Path to noc_2021_merged_clean.csv for human-readable title lookup.
        If omitted, predictions show NOC codes only.
    device : torch.device | str, optional
        Defaults to CUDA if available, else CPU.
    max_length : int
        Tokeniser truncation length. Must match training (default 256).
    prefer_ema : bool
        Prefer EMA weights over base model weights in checkpoint (default True).
    """

    MAX_LENGTH_DEFAULT: int = 256

    def __init__(
        self,
        checkpoint_path: str | Path,
        bert_model_path: str | Path,
        label_map_path: str | Path,
        noc_titles_path: Optional[str | Path] = None,
        device: Optional[torch.device | str] = None,
        max_length: int = MAX_LENGTH_DEFAULT,
        prefer_ema: bool = True,
    ) -> None:
        # ── Device ────────────────────────────────────────────────────────────
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        logger.info("NOCPredictor initialising on %s", self.device)

        self.max_length = max_length

        # ── Tokeniser ─────────────────────────────────────────────────────────
        bert_model_path = Path(bert_model_path)
        if not bert_model_path.exists():
            raise FileNotFoundError(
                f"DistilBERT model directory not found: {bert_model_path}\n"
                "Expected: distilbert-base-uncased folder with vocab.txt, config.json, etc."
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(bert_model_path),
            local_files_only=True,
        )
        logger.info("Tokeniser loaded — vocab size: %d", self.tokenizer.vocab_size)

        # ── Label map ─────────────────────────────────────────────────────────
        self.label_map = LabelMap(
            label_map_path=label_map_path,
            noc_titles_path=noc_titles_path,
        )

        # ── Model ─────────────────────────────────────────────────────────────
        self.model: HierarchicalNOCModel = load_checkpoint(
            checkpoint_path=checkpoint_path,
            bert_model_path=bert_model_path,
            device=self.device,
            prefer_ema=prefer_ema,
        )
        self.model.eval()

        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info("Model loaded — %.1fM parameters", n_params / 1e6)

    # ── Single-example prediction ──────────────────────────────────────────────

    def predict(
        self,
        description: str,
        title: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Prediction]:
        """
        Classify a single job description.

        Parameters
        ----------
        description : str
            Job posting body text (HTML is stripped automatically).
        title : str, optional
            Job title / posting title. When provided it is prepended to the
            description, mirroring the dominant training mode.
        top_k : int
            Number of ranked L5 predictions to return (default 5).

        Returns
        -------
        list[Prediction]
            Ranked from highest to lowest L5 softmax probability.
            See module-level ``Prediction`` type alias for field details.
        """
        return self.predict_batch(
            descriptions=[description],
            titles=[title] if title is not None else None,
            top_k=top_k,
        )[0]

    # ── Batch prediction ───────────────────────────────────────────────────────

    def predict_batch(
        self,
        descriptions: List[str],
        titles: Optional[List[Optional[str]]] = None,
        top_k: int = 5,
        batch_size: int = 32,
    ) -> List[List[Prediction]]:
        """
        Classify a list of job descriptions efficiently.

        Parameters
        ----------
        descriptions : list[str]
            Job description texts.
        titles : list[str | None], optional
            Parallel list of job titles. Pass None for items without a title.
            If omitted entirely, no titles are used.
        top_k : int
            Top-k L5 predictions per example (default 5).
        batch_size : int
            GPU/CPU micro-batch size (default 32). Reduce if OOM.

        Returns
        -------
        list[list[Prediction]]
            One list of predictions per input description.
        """
        if titles is None:
            titles = [None] * len(descriptions)

        if len(titles) != len(descriptions):
            raise ValueError(
                f"descriptions ({len(descriptions)}) and titles ({len(titles)}) "
                "must have the same length."
            )

        # ── Compose and clean input texts ─────────────────────────────────────
        texts = [
            build_input_text(desc, ttl)
            for desc, ttl in zip(descriptions, titles)
        ]

        # ── Sanity check for empty inputs ─────────────────────────────────────
        empty_mask = [not t for t in texts]
        if any(empty_mask):
            n_empty = sum(empty_mask)
            logger.warning(
                "%d of %d inputs produced empty text after cleaning. "
                "These will return low-confidence predictions.",
                n_empty,
                len(texts),
            )

        # ── Run inference in micro-batches ────────────────────────────────────
        all_logits_l5: List[torch.Tensor] = []
        all_level_probs: List[Dict[str, float]] = []

        t0 = time.perf_counter()
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                chunk = texts[start : start + batch_size]
                logits_dict = self._forward_batch(chunk)

                # L5 (used for ranking)
                l5_probs = F.softmax(logits_dict["l5"], dim=-1)  # [B, 515]
                all_logits_l5.append(l5_probs.cpu())

                # Per-level top-1 for diagnostics
                for i in range(len(chunk)):
                    level_probs: Dict[str, float] = {}
                    for lvl in range(1, 6):
                        key = f"l{lvl}"
                        probs = F.softmax(logits_dict[key][i], dim=-1)
                        level_probs[key] = float(probs.max().item())
                    all_level_probs.append(level_probs)

        elapsed = time.perf_counter() - t0
        logger.debug(
            "Inference: %d examples in %.3fs (%.1f ms/example)",
            len(texts),
            elapsed,
            elapsed / len(texts) * 1000,
        )

        # ── Build prediction dicts ────────────────────────────────────────────
        all_probs = torch.cat(all_logits_l5, dim=0)  # [N, 515]
        k = min(top_k, all_probs.size(-1))
        top_probs, top_idxs = torch.topk(all_probs, k=k, dim=-1)  # [N, k]

        results: List[List[Prediction]] = []
        for i in range(len(descriptions)):
            preds: List[Prediction] = []
            for rank in range(k):
                idx        = int(top_idxs[i, rank].item())
                confidence = float(top_probs[i, rank].item())
                noc5       = self.label_map.idx_to_noc5(idx)
                preds.append(
                    {
                        "rank":              rank + 1,
                        "noc_code":          noc5,
                        "noc_title":         self.label_map.get_title(noc5),
                        "confidence":        confidence,
                        "hierarchy":         self.label_map.get_hierarchy(noc5),
                        "level_confidences": all_level_probs[i],
                    }
                )
            results.append(preds)

        return results

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _forward_batch(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenise a list of texts and run one forward pass."""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids      = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        logits_dict, _ = self.model(input_ids, attention_mask)
        return logits_dict

    # ── Convenience ───────────────────────────────────────────────────────────

    def top1(self, description: str, title: Optional[str] = None) -> Prediction:
        """Return only the top-1 prediction for a single input."""
        return self.predict(description, title, top_k=1)[0]

    def __repr__(self) -> str:
        return (
            f"NOCPredictor("
            f"device={self.device}, "
            f"max_length={self.max_length}, "
            f"n_l5_classes={len(self.label_map.idx_to_code.get(5, []))})"
        )
