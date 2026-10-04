# ==============================================================================
# CIKM 2026 Tutorial — IDs That Mean Something
# ------------------------------------------------------------------------------
# Trains, evaluates, and saves SASRec and TIGER on the Amazon Beauty dataset.
#
#   Note: this script is NOT meant to be run live during the tutorial (training TIGER's
#   RQ-KMeans tokenizer + seq2seq model takes too long for the session).
#   It is provided for attendees who want to reproduce the pretrained
#   checkpoints used in Session 1 - Part 2, or experiment with the
#   take-home exercise afterwards.
#
# Usage:
#   python train_and_save_models.py
#
# Output:
#   saved_models/SASRec/<timestamp>.pkl
#   saved_models/TIGER/<timestamp>.pkl
# ==============================================================================

import torch
from sentence_transformers import SentenceTransformer

from cornac.data import FeatureModality
from cornac.datasets import amazon_review
from cornac.eval_methods import NextItemEvaluation
from cornac.experiment import Experiment
from cornac.metrics import MRR, NDCG, Recall
from cornac.models import SASRec, TIGER

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "saved_models"

TIGER_CONFIG = dict(
    tokenizer="rkmeans",
    feature_standardize=True,
    num_enc_layers=1,
    num_dec_layers=1,
    num_heads=1,
    dropout=0.2,
    learning_rate=5e-4,
    lr_schedule="cosine",
    warmup_steps=1000,
    weight_decay=0.035,
    batch_size=512,
    n_beams=30,
    model_selection="best",
    val_metric="ndcg",
    val_k=10,
    n_epochs=50,
    val_eval_every=10,
    val_sample=2000,
    device=DEVICE,
    verbose=True,
    seed=123,
)

SASREC_CONFIG = dict(
    embedding_dim=128,
    loss="cross-entropy",
    n_sample=512,
    batch_size=512,
    n_epochs=50,
    learning_rate=1e-2,
    max_len=20,
    num_blocks=1,
    num_heads=1,
    model_selection="best",
    val_eval_every=10,
    val_metric="ndcg",
    val_k=10,
    device=DEVICE,
    verbose=True,
    seed=123,
)

# Data
data = amazon_review.load_feedback(category="beauty")
texts, item_ids = amazon_review.load_text(category="beauty")

encoder = SentenceTransformer("sentence-t5-base", device=DEVICE)
features = encoder.encode(texts, batch_size=64, show_progress_bar=True)
del encoder
if DEVICE == "cuda":
    torch.cuda.empty_cache()

# Evaluation protocol (seeded for a reproducible split)
next_item_eval = NextItemEvaluation.leave_last_out(
    data=data,
    exclude_unknowns=True,
    verbose=True,
    item_feature=FeatureModality(features=features, ids=item_ids),
    seed=123,
)

# Models
models = [SASRec(**SASREC_CONFIG), TIGER(**TIGER_CONFIG)]
metrics = [Recall(k=10), NDCG(k=10), MRR()]

# Train and evaluate
Experiment(
    eval_method=next_item_eval, 
    models=models, metrics=metrics, 
    show_validation=False).run()

sasrec, tiger = models

# Save: models
sasrec_path = sasrec.save(save_dir=SAVE_DIR)
tiger_path = tiger.save(save_dir=SAVE_DIR)


print(f"Saved SASRec to:     {sasrec_path}")
print(f"Saved TIGER to:      {tiger_path}")
