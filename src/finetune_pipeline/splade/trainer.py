import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

from base.data import LitSearchTripletDataset
from base.loss import TripletMarginLoss


class SpladeTrainer:
    def __init__(self, model, tokenizer, device, regularizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.regularizer = regularizer

    def train(
        self,
        train_data,
        val_data=None,
        output_dir="./output",
        lr=2e-5,
        batch_size=8,
        epochs=3,
        kl_weight=1.0,
        margin_weight=0.05,
        eval_steps=100,
        weight_decay=0.01,
        warmup_ratio=0.1,
        max_length=128,
        margin=1.0,
    ):
        train_dataset = LitSearchTripletDataset(train_data, self.tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
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

        triplet_loss_fn = TripletMarginLoss(margin=margin)
        global_step = 0
        best_val_loss = float("inf")

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

            for batch in progress_bar:
                # batch: dict with keys ["query", "positive_title", "positive_abstract", "negative_title", "negative_abstract"]
                query = batch["query"]
                pos_title = batch["positive_title"]
                pos_abs = batch["positive_abstract"]
                neg_title = batch["negative_title"]
                neg_abs = batch["negative_abstract"]

                # Tokenize
                q_tokens = self.tokenizer(
                    query,
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(self.device)
                pos_tokens = self.tokenizer(
                    [
                        t + self.tokenizer.sep_token + a
                        for t, a in zip(pos_title, pos_abs)
                    ],
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(self.device)
                neg_tokens = self.tokenizer(
                    [
                        t + self.tokenizer.sep_token + a
                        for t, a in zip(neg_title, neg_abs)
                    ],
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(self.device)

                q_rep = self.model.encode(q_tokens, is_q=True)
                pos_rep = self.model.encode(pos_tokens, is_q=False)
                neg_rep = self.model.encode(neg_tokens, is_q=False)

                ranking_loss = triplet_loss_fn(q_rep, pos_rep, neg_rep)

                reg = self.regularizer["train"]["reg"]
                lambda_q = reg["lambdas"]["lambda_q"].get_lambda()
                lambda_d = reg["lambdas"]["lambda_d"].get_lambda()
                reg_loss_fn = reg["loss"]

                reg_loss_q = reg_loss_fn(q_rep) * lambda_q
                reg_loss_pos = reg_loss_fn(pos_rep) * lambda_d
                reg_loss_neg = reg_loss_fn(neg_rep) * lambda_d
                reg_loss = reg_loss_q + (reg_loss_pos + reg_loss_neg) / 2

                loss = ranking_loss + reg_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                progress_bar.set_postfix({"loss": loss.item()})

                del q_rep, pos_rep, neg_rep, loss
                torch.cuda.empty_cache()

                global_step += 1
                if val_data is not None and global_step % eval_steps == 0:
                    val_loss = self.evaluate(val_data, batch_size, max_length)
                    print(f"Validation Loss: {val_loss:.4f}")

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        self.save_model(output_dir)
                        print(f"Model saved to {output_dir} (val_loss: {val_loss:.4f})")

                    self.model.train()

            avg_epoch_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{epochs} - Avg Loss: {avg_epoch_loss:.4f}")

        if val_data is None or epochs % eval_steps != 0:
            self.save_model(output_dir)

        return self.model

    def evaluate(self, val_data, batch_size=8, max_length=128, margin=1.0):
        val_dataset = LitSearchTripletDataset(val_data, self.tokenizer)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        self.model.eval()
        triplet_loss_fn = TripletMarginLoss(margin=margin)
        total_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                query = batch["query"]
                pos_title = batch["positive_title"]
                pos_abs = batch["positive_abstract"]
                neg_title = batch["negative_title"]
                neg_abs = batch["negative_abstract"]

                q_tokens = self.tokenizer(
                    query,
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(self.device)
                pos_tokens = self.tokenizer(
                    [
                        t + self.tokenizer.sep_token + a
                        for t, a in zip(pos_title, pos_abs)
                    ],
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(self.device)
                neg_tokens = self.tokenizer(
                    [
                        t + self.tokenizer.sep_token + a
                        for t, a in zip(neg_title, neg_abs)
                    ],
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(self.device)

                q_rep = self.model.encode(q_tokens, is_q=True)
                pos_rep = self.model.encode(pos_tokens, is_q=False)
                neg_rep = self.model.encode(neg_tokens, is_q=False)

                ranking_loss = triplet_loss_fn(q_rep, pos_rep, neg_rep)

                reg = self.regularizer["train"]["reg"]
                lambda_q = reg["lambdas"]["lambda_q"].get_lambda()
                lambda_d = reg["lambdas"]["lambda_d"].get_lambda()
                reg_loss_fn = reg["loss"]

                reg_loss_q = reg_loss_fn(q_rep) * lambda_q
                reg_loss_pos = reg_loss_fn(pos_rep) * lambda_d
                reg_loss_neg = reg_loss_fn(neg_rep) * lambda_d
                reg_loss = reg_loss_q + (reg_loss_pos + reg_loss_neg) / 2

                loss = ranking_loss + reg_loss
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def save_model(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        # SPLADE 모델 저장 방식에 맞게 구현
        torch.save(
            self.model.state_dict(), os.path.join(output_dir, "pytorch_model.bin")
        )
        self.tokenizer.save_pretrained(output_dir)
        print(f"모델과 토크나이저가 {output_dir}에 저장되었습니다.")
