"""Cross-Layer Attention Probe (CLAP) for per-field extraction error detection.

Implements the probing architecture from:
    Suresh et al., "Cross-Layer Attention Probing for Fine-Grained
    Hallucination Detection", TRUST-AI @ ECAI 2025.
    https://arxiv.org/pdf/2509.09700

Adaptation for this project
----------------------------
The original paper detects hallucinations in open-domain QA by probing the
EOS token's hidden states. Here we apply the same architecture to per-field
extraction error detection:

  Input  : n_layers activation vectors per field, each shape (hidden_dim,),
           stacked into a sequence (n_layers, hidden_dim).
  Task   : Binary — is this extracted field value wrong?
  Token  : Last content token of the field value in the generated JSON
           (the "last_token" position strategy already used by the project).

Architecture (identical to the paper)
---------------------------------------
  1. Down-projection : Linear(hidden_dim → d_model) + LayerNorm
  2. CLS token       : Learnable vector prepended → sequence (1+n_layers, d_model)
  3. Encoder         : n_enc TransformerEncoderLayer blocks with full attention
  4. Classifier      : Linear(d_model → 1) on the CLS output only

The attention learns which layers matter most — no manual layer selection needed.

Why this is novel for DeepSeek-R1
-----------------------------------
DeepSeek-R1-Distill is a reasoning model. Its hidden states during JSON
generation already encode the completed chain-of-thought. Cross-layer probing
of a reasoning model for extraction trust has not been studied before.
The CLAP attention weights can show whether the trust signal lives in reasoning
layers (early/mid) or answer-writing layers (late) — a new scientific question.

Interface
----------
Mirrors linear.py exactly so stages 04 and 07 can use both probe types uniformly.
train_clap()       — full training with CV; used by 06_train_clap.py
fit_clap_for_lodo() — single-fold fit; used by 07_lodo_clap.py (no nested CV)
predict_clap_proba() — inference; used by both scripts
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


# ============================================================================
# Public data structures
# ============================================================================

@dataclass
class ClapMetrics:
    """Evaluation metrics produced by train_clap().

    Mirrors ProbeMetrics from linear.py so downstream code can treat both
    probe types uniformly.
    """
    n_train: int
    n_test: int
    n_test_errors: int
    auroc: float          # on 20% held-out split
    auprc: float          # on 20% held-out split
    per_fold_auroc: list[float] = field(default_factory=list)

    @property
    def cv_auroc_mean(self) -> float | None:
        return float(np.mean(self.per_fold_auroc)) if self.per_fold_auroc else None

    @property
    def cv_auroc_std(self) -> float | None:
        return float(np.std(self.per_fold_auroc)) if self.per_fold_auroc else None


@dataclass
class ClapProbeResult:
    """A trained CLAP probe with metrics and architecture metadata.

    Saved to disk as a .pt file via torch.save().
    The .score() method runs inference without re-importing internals.
    """
    state_dict: dict[str, Any]   # PyTorch weights (CPU tensors)
    arch_config: dict[str, Any]  # hidden_dim, n_layers, d_model, n_enc, dropout
    metrics: ClapMetrics
    layers: list[int]            # which LLM layer indices were used

    def score(self, X_multilayer: np.ndarray) -> np.ndarray:
        """Return P(error) for each field.

        Args:
            X_multilayer: float32 array shape (n_samples, n_layers, hidden_dim).

        Returns:
            (n_samples,) float32 array of P(error) in [0, 1].
        """
        model = _build_model(self.arch_config)
        model.load_state_dict(self.state_dict)
        model.eval()
        with torch.no_grad():
            t = torch.from_numpy(X_multilayer.astype(np.float32))
            return torch.sigmoid(model(t)).squeeze(-1).numpy()


# ============================================================================
# Neural network
# ============================================================================

class _CLAPModel(nn.Module):
    """CLAP neural network: (batch, n_layers, hidden_dim) → (batch, 1) logits."""

    def __init__(
        self,
        hidden_dim: int,   # LLM hidden size, e.g. 3584 for DeepSeek-R1-7B
        n_layers: int,     # number of captured layers, e.g. 14
        d_model: int,      # projection dimension, e.g. 128
        n_enc: int,        # transformer encoder layers, e.g. 1
        dropout: float,    # dropout rate
    ) -> None:
        super().__init__()

        # Step 1 — down-project each layer vector to d_model
        self.down_proj = nn.Linear(hidden_dim, d_model)
        self.proj_norm = nn.LayerNorm(d_model)

        # Step 2 — learnable CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Step 3 — transformer encoder
        # nhead must divide d_model: d_model=128 → nhead=4
        nhead = max(1, d_model // 32)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,    # (batch, seq, d_model)
            norm_first=True,     # pre-norm: more stable for small datasets
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_enc)

        # Step 4 — classify from CLS output
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_layers, hidden_dim)
        Returns:
            (batch, 1) raw logits — apply sigmoid for probabilities
        """
        x = self.proj_norm(self.down_proj(x))         # (batch, n_layers, d_model)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)                # (batch, 1+n_layers, d_model)
        x = self.encoder(x)                            # (batch, 1+n_layers, d_model)
        return self.classifier(x[:, 0, :])             # (batch, 1)


def _build_model(arch_config: dict[str, Any]) -> _CLAPModel:
    """Reconstruct a model from its saved arch_config dict."""
    return _CLAPModel(
        hidden_dim=arch_config["hidden_dim"],
        n_layers=arch_config["n_layers"],
        d_model=arch_config["d_model"],
        n_enc=arch_config["n_enc"],
        dropout=arch_config["dropout"],
    )


# ============================================================================
# Public training API
# ============================================================================

def train_clap(
    *,
    X: np.ndarray,           # (n_samples, n_layers, hidden_dim)
    y: np.ndarray,           # (n_samples,) int32
    layers: list[int],       # which LLM layers (metadata)
    d_model: int = 128,
    n_enc: int = 1,
    dropout: float = 0.1,
    lr: float = 5e-4,
    epochs: int = 50,
    batch_size: int = 64,
    weight_decay: float = 1e-2,
    warmup_epochs: int = 5,
    patience: int = 10,
    cv_folds: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
) -> ClapProbeResult:
    """Train CLAP with K-fold CV and a held-out evaluation split.

    Three training runs in total:
      1. K-fold CV (cv_folds models) → per_fold_auroc: honest cross-validated metric.
      2. 80/20 split (1 model)       → auroc, auprc on held-out set.
      3. Full dataset (1 model)      → the saved weights used for inference.

    Used by scripts/06_train_clap.py.
    """
    n_samples, n_layers, hidden_dim = X.shape
    n_pos = int(y.sum())
    n_neg = n_samples - n_pos

    arch_config: dict[str, Any] = {
        "hidden_dim": hidden_dim,
        "n_layers":   n_layers,
        "d_model":    d_model,
        "n_enc":      n_enc,
        "dropout":    dropout,
    }

    logger.info(
        "CLAP | n=%d (%d pos / %d neg) | layers=%d | hidden=%d | d_model=%d | n_enc=%d",
        n_samples, n_pos, n_neg, n_layers, hidden_dim, d_model, n_enc,
    )

    # ── 1. K-fold CV ──────────────────────────────────────────────────────
    per_fold_auroc: list[float] = []
    if n_pos >= cv_folds and n_neg >= cv_folds:
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True,
                               random_state=random_state)
        for fold_i, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
            m = _build_model(arch_config)
            _fit(m, X[tr_idx], y[tr_idx],
                 lr=lr, epochs=epochs, batch_size=batch_size,
                 weight_decay=weight_decay, warmup_epochs=warmup_epochs,
                 patience=patience, random_state=random_state + fold_i)
            proba = predict_clap_proba(m, X[val_idx])
            if y[val_idx].sum() in (0, len(val_idx)):
                continue
            per_fold_auroc.append(float(roc_auc_score(y[val_idx], proba)))

        if per_fold_auroc:
            logger.info(
                "CLAP CV AUROC: %.4f ± %.4f (%d folds)",
                float(np.mean(per_fold_auroc)),
                float(np.std(per_fold_auroc)),
                len(per_fold_auroc),
            )
    else:
        logger.warning("Skipping CV: too few positives or negatives for %d folds.",
                       cv_folds)

    # ── 2. Train/test split ───────────────────────────────────────────────
    rng = np.random.default_rng(random_state)
    idx = rng.permutation(n_samples)
    n_test = max(1, int(round(test_size * n_samples)))
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    m_eval = _build_model(arch_config)
    _fit(m_eval, X[train_idx], y[train_idx],
         lr=lr, epochs=epochs, batch_size=batch_size,
         weight_decay=weight_decay, warmup_epochs=warmup_epochs,
         patience=patience, random_state=random_state)
    test_proba = predict_clap_proba(m_eval, X[test_idx])
    n_test_errors = int(y[test_idx].sum())

    if n_test_errors in (0, n_test):
        auroc = auprc = float("nan")
        logger.warning("Split eval: single class in test set, AUROC undefined.")
    else:
        auroc = float(roc_auc_score(y[test_idx], test_proba))
        auprc = float(average_precision_score(y[test_idx], test_proba))
        logger.info("CLAP split AUROC=%.4f  AUPRC=%.4f", auroc, auprc)

    metrics = ClapMetrics(
        n_train=len(train_idx), n_test=n_test,
        n_test_errors=n_test_errors, auroc=auroc, auprc=auprc,
        per_fold_auroc=per_fold_auroc,
    )

    # ── 3. Final model on ALL data ────────────────────────────────────────
    logger.info("Fitting final CLAP model on all %d samples...", n_samples)
    m_final = _build_model(arch_config)
    _fit(m_final, X, y,
         lr=lr, epochs=epochs, batch_size=batch_size,
         weight_decay=weight_decay, warmup_epochs=warmup_epochs,
         patience=None,   # no early stopping; train full budget
         random_state=random_state)

    state_dict = {k: v.cpu() for k, v in m_final.state_dict().items()}
    return ClapProbeResult(state_dict=state_dict, arch_config=arch_config,
                           metrics=metrics, layers=layers)


def fit_clap_for_lodo(
    *,
    X_train: np.ndarray,     # (n_train_fields, n_layers, hidden_dim)
    y_train: np.ndarray,     # (n_train_fields,)
    arch_config: dict[str, Any],
    lr: float = 5e-4,
    epochs: int = 30,        # fewer epochs than full training — LODO runs 25 folds
    batch_size: int = 64,
    weight_decay: float = 1e-2,
    warmup_epochs: int = 3,
    patience: int = 5,
    random_state: int = 42,
) -> _CLAPModel:
    """Train a single CLAP model for one LODO fold.

    Used by scripts/07_lodo_clap.py. Does NOT run nested CV — just a
    single fit on the training documents and returns the trained model.
    Fewer epochs than train_clap() because we call this 25 times (once
    per document) and need the total runtime to stay under ~5 minutes.
    """
    model = _build_model(arch_config)
    _fit(model, X_train, y_train,
         lr=lr, epochs=epochs, batch_size=batch_size,
         weight_decay=weight_decay, warmup_epochs=warmup_epochs,
         patience=patience, random_state=random_state)
    return model


def predict_clap_proba(model: _CLAPModel, X: np.ndarray) -> np.ndarray:
    """Return P(error) for each sample. Works with any _CLAPModel instance."""
    model.eval()
    with torch.no_grad():
        t = torch.from_numpy(X.astype(np.float32))
        return torch.sigmoid(model(t)).squeeze(-1).numpy()


# ============================================================================
# Internal training loop (used by train_clap and fit_clap_for_lodo)
# ============================================================================

def _fit(
    model: _CLAPModel,
    X: np.ndarray,
    y: np.ndarray,
    *,
    lr: float,
    epochs: int,
    batch_size: int,
    weight_decay: float,
    warmup_epochs: int,
    patience: int | None,
    random_state: int,
) -> None:
    """One complete training run with AdamW + cosine LR + optional early stop.

    pos_weight = n_neg/n_pos mirrors class_weight='balanced' in sklearn,
    so the imbalanced 12% error rate is handled automatically.
    """
    torch.manual_seed(random_state)

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay,
    )

    def lr_lambda(ep: int) -> float:
        if warmup_epochs > 0 and ep < warmup_epochs:
            return (ep + 1) / warmup_epochs
        t = (ep - warmup_epochs) / max(epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    X_t = torch.from_numpy(X.astype(np.float32))
    y_t = torch.from_numpy(y.astype(np.float32)).unsqueeze(-1)  # (n, 1)
    n = len(y)
    best_loss, no_improve = float("inf"), 0

    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n)
        epoch_loss, n_batches = 0.0, 0

        for start in range(0, n, batch_size):
            idx = perm[start: start + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(X_t[idx]), y_t[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        if patience is not None:
            avg = epoch_loss / max(n_batches, 1)
            if avg < best_loss - 1e-4:
                best_loss, no_improve = avg, 0
            else:
                no_improve += 1
            if no_improve >= patience:
                logger.debug("Early stop at epoch %d (loss=%.5f)", epoch + 1, avg)
                break

    model.eval()