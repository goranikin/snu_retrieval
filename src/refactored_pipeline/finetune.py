# snu_retrieval/src/refactored_pipeline/finetune.py

import json

import hydra
import numpy as np
import torch
from datasets import load_dataset
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from refactored_pipeline.datasets.data import LitSearchTripletDataset
from refactored_pipeline.losses.loss import TripletMarginLoss
from refactored_pipeline.models.base import Specter2Encoder
from refactored_pipeline.tasks.trainer import Specter2Trainer
from refactored_pipeline.utils.data_processing_utils import create_kv_pairs


@hydra.main(
    config_path="conf",
    config_name="finetune_specter2",
    version_base=None,
)
def main(cfg: DictConfig):
    # 1. Data loading
    with open(cfg.train_data_path, "r") as f:
        train_data = json.load(f)
    train_dataset = LitSearchTripletDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)

    corpus_clean_data = load_dataset(
        cfg.litsearch_dataset_path, "corpus_clean", split="full"
    )
    query_data = load_dataset(cfg.litsearch_dataset_path, "query", split="full")

    kv_pairs = create_kv_pairs(corpus_clean_data)
    documents = list(kv_pairs.keys())
    corpusid_list = list(kv_pairs.values())
    np.save(cfg.corpusid_list_path, np.array(corpusid_list))

    # 2. Prepare model, loss, optimizer, scheduler
    model = Specter2Encoder(
        model_name_or_dir=cfg.model_name_or_dir,
        freeze_body_and_docs=cfg.freeze_body_and_docs,
    )
    loss_fn = TripletMarginLoss(margin=cfg.margin)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    total_steps = len(train_loader) * cfg.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.lr,
        total_steps=total_steps,
        pct_start=cfg.warmup_ratio,
        anneal_strategy="linear",
    )

    # 3. Trainer setting
    trainer = Specter2Trainer(
        model=model,
        loss=loss_fn,
        optimizer=optimizer,
        config=cfg,
        train_loader=train_loader,
        validation_loss_loader=None,  # 필요시 설정
        validation_evaluator=None,  # 필요시 설정
        scheduler=scheduler,
    )
    trainer.set_recall_validation_data(documents, list(query_data))

    # 4. finetuning
    trainer.train()


if __name__ == "__main__":
    main()
