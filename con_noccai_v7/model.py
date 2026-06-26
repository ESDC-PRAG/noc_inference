"""
con_noccai_v7/model.py

Hierarchical NOC 2021 classification model — V7 architecture.
Key names reverse-engineered from checkpoint inspection:

classification_heads.level_1.1.weight  (10,  768)   <- shallow: Dropout → Linear
classification_heads.level_2.1.weight  (45,  768)   <- shallow: Dropout → Linear
classification_heads.level_3.1.weight  (384, 768)   <- deep:    Dropout → Linear → ReLU → Dropout → Linear
classification_heads.level_3.4.weight  (89,  384)
classification_heads.level_4.1.weight  (384, 768)
classification_heads.level_4.4.weight  (162, 384)
classification_heads.level_5.1.weight  (384, 768)
classification_heads.level_5.4.weight  (515, 384)
projection_head.0.weight               (512, 768)   <- Linear(768→512) → ReLU → Dropout → Linear(512→256)
projection_head.3.weight               (256, 512)

EMA weights: ckpt['ema_state']['shadow']  (flat state dict, same keys as above)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel

logger = logging.getLogger(__name__)

# Confirmed from checkpoint
LEVEL_SIZES   = {1: 10, 2: 45, 3: 89, 4: 162, 5: 515}
DEEP_LEVELS   = frozenset({3, 4, 5})
DEEP_HIDDEN   = 384
PROJ_HIDDEN   = 512
PROJ_OUT      = 256
DROPOUT       = 0.25


class HierarchicalNOCModel(nn.Module):
    """
    Exact mirror of hierarchy_v7 training model.

    classification_heads.level_N  (ModuleDict of Sequential)
      Shallow (L1, L2):
        0: Dropout(p)
        1: Linear(768, N)

      Deep (L3, L4, L5):
        0: Dropout(p)
        1: Linear(768, 384)
        2: ReLU()
        3: Dropout(p)
        4: Linear(384, N)

    projection_head  (Sequential)
        0: Linear(768, 512)
        1: ReLU()
        2: Dropout(p)
        3: Linear(512, 256)
    """

    def __init__(
        self,
        bert_model_path: str | Path,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()

        self.bert = DistilBertModel.from_pretrained(
            str(bert_model_path), local_files_only=True
        )
        h = self.bert.config.dim  # 768

        self.classification_heads = nn.ModuleDict({
            # Shallow heads — Dropout → Linear
            "level_1": nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(h, 10),
            ),
            "level_2": nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(h, 45),
            ),
            # Deep heads — Dropout → Linear → ReLU → Dropout → Linear
            "level_3": nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(h, DEEP_HIDDEN),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(DEEP_HIDDEN, 89),
            ),
            "level_4": nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(h, DEEP_HIDDEN),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(DEEP_HIDDEN, 162),
            ),
            "level_5": nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(h, DEEP_HIDDEN),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(DEEP_HIDDEN, 515),
            ),
        })

        # Linear(768→512) → ReLU → Dropout → Linear(512→256)
        self.projection_head = nn.Sequential(
            nn.Linear(h, PROJ_HIDDEN),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(PROJ_HIDDEN, PROJ_OUT),
        )

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0]   # CLS token [B, 768]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        cls = self.encode(input_ids, attention_mask)
        logits = {
            f"l{lvl}": self.classification_heads[f"level_{lvl}"](cls)
            for lvl in range(1, 6)
        }
        proj = F.normalize(self.projection_head(cls), p=2, dim=-1)
        return logits, proj


def load_checkpoint(
    checkpoint_path: str | Path,
    bert_model_path: str | Path,
    device: Optional[torch.device] = None,
    prefer_ema: bool = True,
) -> HierarchicalNOCModel:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info("Loading checkpoint from %s onto %s", checkpoint_path, device)
    raw = torch.load(checkpoint_path, map_location=device)

    state, source = _extract_state_dict(raw, prefer_ema)
    logger.info("Using %s weights from checkpoint", source)

    # Strip DataParallel prefix if present
    if any(k.startswith("module.") for k in state):
        state = {k[7:]: v for k, v in state.items()}

    model = HierarchicalNOCModel(bert_model_path=bert_model_path)
    missing, unexpected = model.load_state_dict(state, strict=False)

    EXPECTED_MISSING = {
        "bert.embeddings.word_embeddings.weight",
        "bert.embeddings.position_embeddings.weight",
        "bert.embeddings.LayerNorm.weight",
        "bert.embeddings.LayerNorm.bias",
    }

    true_missing   = [k for k in missing   if k not in EXPECTED_MISSING]
    frozen_missing = [k for k in missing   if k in EXPECTED_MISSING]

    if frozen_missing:
        logger.info(
            "Embedding weights (%d) loaded from pretrained DistilBERT "
            "(frozen during training, not tracked by EMA — expected).",
            len(frozen_missing),
        )
    if true_missing:
        logger.warning("Truly missing keys (%d): %s", len(true_missing), true_missing[:8])
    if unexpected:
        logger.warning("Unexpected keys (%d): %s", len(unexpected), unexpected[:8])
    if not true_missing and not unexpected:
        logger.info("State dict loaded cleanly — all keys matched.")

    for meta in ("epoch", "global_step", "best_l5_acc", "val_loss"):
        if meta in raw:
            logger.info("  checkpoint %s = %s", meta, raw[meta])

    model.to(device)
    model.eval()
    return model


def _extract_state_dict(raw: dict, prefer_ema: bool) -> Tuple[dict, str]:
    # EMA weights: ema_state['shadow'] is the flat param dict
    if prefer_ema and "ema_state" in raw:
        ema = raw["ema_state"]
        if isinstance(ema, dict) and "shadow" in ema:
            shadow = ema["shadow"]
            if isinstance(shadow, dict) and shadow:
                return shadow, "EMA shadow"

    if "model_state_dict" in raw:
        return raw["model_state_dict"], "model"

    # Flat state dict saved directly
    sample = list(raw.keys())[:5]
    if any(k.startswith(("bert.", "classification_heads.", "projection_head.")) for k in sample):
        return raw, "flat state dict"

    raise KeyError(
        f"Cannot locate model weights. Top-level keys: {list(raw.keys())}"
    )
