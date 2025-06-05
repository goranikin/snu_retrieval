import os

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from finetune_pipeline.data.data import LitSearchTripletDataset
from finetune_pipeline.models.dpr.model.loss import TripletMarginLoss


class DprTrainer:
    def __init__(self, model_wrapper):
        self.model_wrapper = model_wrapper
        self.query_encoder = model_wrapper.query_encoder
        self.paper_encoder = model_wrapper.paper_encoder
        self.device = model_wrapper.device

    def train(
        self,
        train_data,
        val_data=None,
        output_dir="./dpr_finetuned",
        lr=2e-5,
        batch_size=8,
        epochs=3,
        margin=1.0,
        eval_steps=100,
        weight_decay=0.01,
        warmup_ratio=0.1,
    ):
        train_dataset = LitSearchTripletDataset(
            train_data, self.model_wrapper.query_tokenizer
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        optimizer = AdamW(
            [p for p in self.model_wrapper.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=weight_decay,
        )

        total_steps = len(train_loader) * epochs
        scheduler = OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=total_steps,
            pct_start=warmup_ratio,
            anneal_strategy="linear",
        )

        triplet_loss = TripletMarginLoss(margin=margin)
        global_step = 0
        best_val_loss = float("inf")

        for epoch in range(epochs):
            self.query_encoder.train()
            self.paper_encoder.train()
            epoch_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

            for batch in progress_bar:
                query_emb = self.model_wrapper.encode_query(batch["query"])
                pos_emb = self.model_wrapper.encode_paper(
                    batch["positive_title"], batch["positive_abstract"]
                )
                neg_emb = self.model_wrapper.encode_paper(
                    batch["negative_title"], batch["negative_abstract"]
                )

                loss = triplet_loss(query_emb, pos_emb, neg_emb)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model_wrapper.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                progress_bar.set_postfix({"loss": loss.item()})

                global_step += 1
                if val_data is not None and global_step % eval_steps == 0:
                    val_loss = self.evaluate(val_data, batch_size)
                    print(f"Validation Loss: {val_loss:.4f}")

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        self.save_model(output_dir)
                        print(f"Model saved to {output_dir} (val_loss: {val_loss:.4f})")

                    self.query_encoder.train()
                    self.paper_encoder.train()

            avg_epoch_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{epochs} - Avg Loss: {avg_epoch_loss:.4f}")

        if val_data is None or epochs % eval_steps != 0:
            self.save_model(output_dir)

        return self.model_wrapper

    def evaluate(self, val_data, batch_size=8):
        val_dataset = LitSearchTripletDataset(
            val_data, self.model_wrapper.query_tokenizer
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        self.query_encoder.eval()
        self.paper_encoder.eval()
        triplet_loss = TripletMarginLoss(margin=1.0)
        total_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                query_emb = self.model_wrapper.encode_query(batch["query"])
                pos_emb = self.model_wrapper.encode_paper(
                    batch["positive_title"], batch["positive_abstract"]
                )
                neg_emb = self.model_wrapper.encode_paper(
                    batch["negative_title"], batch["negative_abstract"]
                )
                loss = triplet_loss(query_emb, pos_emb, neg_emb)
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def save_model(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.model_wrapper.save_model(output_dir)
        print(f"DPR 모델이 {output_dir}에 저장되었습니다.")
