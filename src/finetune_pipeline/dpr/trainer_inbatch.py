import os

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# from base.loss import TripletMarginLoss


def dpr_inbatch_cross_entropy_loss(query_emb, ctx_emb):
    """
    query_emb: (B, D)
    ctx_emb: (B, D)  # positive contexts only
    """
    # (B, B): similarity matrix
    sim = torch.matmul(query_emb, ctx_emb.t())
    labels = torch.arange(query_emb.size(0), dtype=torch.long, device=query_emb.device)
    loss = torch.nn.CrossEntropyLoss()(sim, labels)
    return loss


class DprLitSearchInBatchDataset(Dataset):
    def __init__(self, data: list[dict[str, str]], tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "query": item["query"],
            "positive_title": item["positive_title"],
            "positive_abstract": item["positive_abstract"],
        }


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
        output_dir="./dpr_finetune_inbatch",
        lr=5e-5,
        batch_size=64,
        epochs=3,
        margin=1.0,
        eval_steps=100,
        weight_decay=0.01,
        warmup_ratio=0.1,
    ):
        train_dataset = DprLitSearchInBatchDataset(
            train_data, self.model_wrapper.query_tokenizer
        )

        def collate_fn(batch):
            return {
                "query": [item["query"] for item in batch],
                "positive_title": [item["positive_title"] for item in batch],
                "positive_abstract": [item["positive_abstract"] for item in batch],
            }

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
        )

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

        # triplet_loss = TripletMarginLoss(margin=margin)
        global_step = 0
        best_val_loss = float("inf")

        for epoch in range(epochs):
            self.query_encoder.train()
            self.paper_encoder.train()
            epoch_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

            for batch in progress_bar:
                # query_emb = self.model_wrapper.encode_query(
                #     batch["query"], no_grad=False
                # )
                # pos_emb = self.model_wrapper.encode_paper(
                #     batch["positive_title"], batch["positive_abstract"], no_grad=False
                # )
                # neg_emb = self.model_wrapper.encode_paper(
                #     batch["negative_title"], batch["negative_abstract"], no_grad=False
                # )

                # loss = triplet_loss(query_emb, pos_emb, neg_emb)
                queries = batch["query"]
                pos_titles = batch["positive_title"]
                pos_abstracts = batch["positive_abstract"]

                query_emb = self.model_wrapper.encode_query(
                    queries, no_grad=False
                )  # (B, D)
                pos_emb = self.model_wrapper.encode_paper(
                    pos_titles, pos_abstracts, no_grad=False
                )  # (B, D)

                def safe_normalize(x, p=2, dim=1, eps=1e-12):
                    return x / (x.norm(p=p, dim=dim, keepdim=True) + eps)

                query_emb = safe_normalize(query_emb, p=2, dim=1)
                pos_emb = safe_normalize(pos_emb, p=2, dim=1)

                loss = dpr_inbatch_cross_entropy_loss(query_emb, pos_emb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                progress_bar.set_postfix({"loss": loss.item()})

                del query_emb, pos_emb, loss
                torch.cuda.empty_cache()

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
        val_dataset = DprLitSearchInBatchDataset(
            val_data, self.model_wrapper.query_tokenizer
        )

        def collate_fn(batch):
            return {
                "query": [item["query"] for item in batch],
                "positive_title": [item["positive_title"] for item in batch],
                "positive_abstract": [item["positive_abstract"] for item in batch],
            }

        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, collate_fn=collate_fn
        )
        self.query_encoder.eval()
        self.paper_encoder.eval()
        # triplet_loss = TripletMarginLoss(margin=1.0)
        total_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                queries = batch["query"]
                pos_titles = batch["positive_title"]
                pos_abstracts = batch["positive_abstract"]

                query_emb = self.model_wrapper.encode_query(
                    queries, no_grad=False
                )  # (B, D)
                pos_emb = self.model_wrapper.encode_paper(
                    pos_titles, pos_abstracts, no_grad=False
                )  # (B, D)

                loss = dpr_inbatch_cross_entropy_loss(query_emb, pos_emb)
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def save_model(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.model_wrapper.save_model(output_dir)
        print(f"DPR 모델이 {output_dir}에 저장되었습니다.")
