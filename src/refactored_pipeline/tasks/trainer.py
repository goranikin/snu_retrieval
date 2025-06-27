import json
import os

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from refactored_pipeline.tasks.base.base_trainer import BaseTrainer
from refactored_pipeline.tasks.evaluator import FaissIndexer, FaissRetriever
from refactored_pipeline.utils.retrieval_utils import mean_recall, retrieve_for_dataset


class Specter2Trainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.writer.add_text("model_name", self.config["model_name_or_dir"])
        self.nb_epochs = self.config["epochs"]
        assert "record_frequency" in self.config, (
            "need to provide record frequency for this trainer"
        )
        self.record_frequency = self.config["record_frequency"]
        assert "gradient_accumulation_steps" in self.config, (
            "need to setup gradient accumulation steps in config"
        )
        self.gradient_accumulation_steps = self.config["gradient_accumulation_steps"]

        assert "corpusid_list_path" in self.config, (
            "need to provide corpusid_list_path in config for recall validation"
        )
        self.corpusid_list_path = self.config.get("corpusid_list_path")
        assert "faiss_index_path" in self.config, (
            "need to provide faiss_index_path in config for recall validation"
        )
        self.faiss_index_path = self.config.get("faiss_index_path")
        self.recall_k = self.config.get("recall_k", 20)

        if os.path.getsize(os.path.join(self.checkpoint_dir, "training_perf.txt")) == 0:
            self.training_res_handler.write("epoch,iter,batch_ranking_loss\n")
        if self.validation:
            to_write = "epoch,iter"
            if self.validation_loss_loader is not None:
                to_write += ",val_ranking_loss"
            if self.validation_evaluator is not None:
                assert "validation_metrics" in self.config, (
                    "need to provide validation metrics"
                )
                self.validation_metrics = self.config["validation_metrics"]
                to_write += "," + ",".join(
                    [f"full_rank_{metric}" for metric in self.validation_metrics]
                )
                assert "val_full_rank_qrel_path" in self.config, (
                    "need to provide path for qrel with this loader"
                )
                self.full_rank_qrel = json.load(
                    open(self.config["val_full_rank_qrel_path"])
                )
            if (
                os.path.getsize(
                    os.path.join(self.checkpoint_dir, "validation_perf.txt")
                )
                == 0
            ):
                self.validation_res_handler.write(to_write + "\n")

    def set_recall_validation_data(self, docs, queries):
        assert docs is not None, "recall_docs must not be None"
        assert queries is not None, "recall_queries must not be None"
        self.recall_docs = docs
        self.recall_queries = queries

    def train_epochs(self):
        global_step = 0
        best_val_loss = float("inf")
        for epoch in range(self.nb_epochs):
            self.model.train()
            epoch_loss = 0.0
            progress_bar = tqdm(
                self.train_loader, desc=f"Epoch {epoch + 1}/{self.nb_epochs}"
            )

            for batch_idx, batch in enumerate(progress_bar):
                queries = batch["query"]
                pos_titles = batch["positive_title"]
                pos_abss = batch["positive_abstract"]
                neg_titles = batch["negative_title"]
                neg_abss = batch["negative_abstract"]

                query_emb = self.model.encode(queries, is_q=True)
                pos_texts = [f"{t} [SEP] {a}" for t, a in zip(pos_titles, pos_abss)]
                neg_texts = [f"{t} [SEP] {a}" for t, a in zip(neg_titles, neg_abss)]
                pos_emb = self.model.encode(pos_texts, is_q=False)
                neg_emb = self.model.encode(neg_texts, is_q=False)

                loss = self.loss(query_emb, pos_emb, neg_emb)
                loss = loss / self.gradient_accumulation_steps
                loss.backward()

                epoch_loss += loss.item() * self.gradient_accumulation_steps

                # Gradient accumulation
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad], 1.0
                    )
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.optimizer.zero_grad()

                # Logging
                global_step += 1
                if global_step % self.record_frequency == 0:
                    avg_loss = epoch_loss / (batch_idx + 1)
                    self.writer.add_scalar("train/loss", avg_loss, global_step)
                    self.training_res_handler.write(
                        f"{epoch},{global_step},{avg_loss:.6f}\n"
                    )

                # Validation
                if self.validation and global_step % self.record_frequency == 0:
                    val_loss = self.evaluate_loss(self.validation_loss_loader)
                    self.writer.add_scalar(
                        "val/loss", val_loss["val_ranking_loss"], global_step
                    )
                    log_line = (
                        f"{epoch},{global_step},{val_loss['val_ranking_loss']:.6f}"
                    )

                    # Retrieval recall@k validation logic (docs/queries passed as arguments)
                    recall = None
                    if hasattr(self, "recall_docs") and hasattr(self, "recall_queries"):
                        recall_docs = self.recall_docs
                        recall_queries = self.recall_queries
                    if (
                        recall_docs is not None
                        and recall_queries is not None
                        and self.corpusid_list_path is not None
                        and self.faiss_index_path is not None
                    ):
                        indexer = FaissIndexer(self.model)
                        indexer.build_faiss_index(recall_docs, self.faiss_index_path)
                        recall_corpusid_list = np.load(self.corpusid_list_path).tolist()
                        retriever = FaissRetriever(self.model, self.faiss_index_path)
                        results = retrieve_for_dataset(
                            retriever,
                            recall_queries,
                            recall_corpusid_list,
                            k=self.recall_k,
                        )
                        recall = mean_recall(results, k=self.recall_k)
                        self.writer.add_scalar("val/recall", recall, global_step)
                        log_line += f",{recall:.6f}"

                    self.validation_res_handler.write(log_line + "\n")
                    if val_loss["val_ranking_loss"] < best_val_loss:
                        best_val_loss = val_loss["val_ranking_loss"]
                        self.save_checkpoint(global_step, best_val_loss, is_best=True)

            self.save_checkpoint(
                global_step, epoch_loss / len(self.train_loader), is_best=False
            )

    def evaluate_loss(self, data_loader):
        self.model.eval()
        total_loss = 0.0
        n = 0
        with torch.no_grad():
            for batch in data_loader:
                queries = batch["query"]
                pos_titles = batch["positive_title"]
                pos_abss = batch["positive_abstract"]
                neg_titles = batch["negative_title"]
                neg_abss = batch["negative_abstract"]

                query_emb = self.model.encode(queries, is_q=True)
                pos_texts = [f"{t} {a}" for t, a in zip(pos_titles, pos_abss)]
                neg_texts = [f"{t} {a}" for t, a in zip(neg_titles, neg_abss)]
                pos_emb = self.model.encode(pos_texts, is_q=False)
                neg_emb = self.model.encode(neg_texts, is_q=False)

                loss = self.loss(query_emb, pos_emb, neg_emb)
                total_loss += loss.item()
                n += 1
        return {"val_ranking_loss": total_loss / n if n > 0 else 0.0}
