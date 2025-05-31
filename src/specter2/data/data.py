import random

from torch.utils.data import Dataset


class LitSearchDataset(Dataset):
    def __init__(self, data: list[dict[str, str]], tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item["question"]

        # 포지티브 샘플 (1개)
        pos_ctx = item["positive_ctxs"][0]
        pos_text = pos_ctx["title"] + self.tokenizer.sep_token + pos_ctx["text"]

        # 네거티브 샘플 (3개 중 랜덤 선택)
        neg_ctx = random.choice(item["negative_ctxs"])
        neg_text = neg_ctx["title"] + self.tokenizer.sep_token + neg_ctx["text"]

        # 토크나이징
        query_tokens = self.tokenizer(
            question,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        pos_tokens = self.tokenizer(
            pos_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        neg_tokens = self.tokenizer(
            neg_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # 배치 차원 제거
        query_tokens = {k: v.squeeze(0) for k, v in query_tokens.items()}
        pos_tokens = {k: v.squeeze(0) for k, v in pos_tokens.items()}
        neg_tokens = {k: v.squeeze(0) for k, v in neg_tokens.items()}

        return {
            "query": query_tokens,
            "positive": pos_tokens,
            "negative": neg_tokens,
        }
