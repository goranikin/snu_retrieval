from torch.utils.data import Dataset


class LitSearchTripletDataset(Dataset):
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
            "negative_title": item["negative_title"],
            "negative_abstract": item["negative_abstract"],
        }
