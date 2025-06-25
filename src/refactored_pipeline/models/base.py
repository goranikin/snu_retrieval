from abc import ABC, abstractmethod

import torch
from adapters import AutoAdapterModel
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.tokenization_auto import AutoTokenizer


class TransformerRep(torch.nn.Module):
    def __init__(self, model_name_or_dir: str) -> None:
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name_or_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_dir)

    def forward(self, **tokens):
        hidden_states = self.transformer(**tokens)[
            0
        ]  # [batch_size, seq_len, hidden_size]
        return hidden_states[:, 0, :]  # CLS token representation


class SiameseBase(torch.nn.Module, ABC):
    def __init__(
        self,
        d_model_name_or_dir,
        q_model_name_or_dir,
        match="dot_product",
        freeze_d_model=False,
    ):
        super().__init__()

        self.transformer_rep_docs = TransformerRep(d_model_name_or_dir)
        self.transformer_rep_query = TransformerRep(q_model_name_or_dir)
        self.match = match

        assert not (freeze_d_model and q_model_name_or_dir is None)
        self.freeze_d_model = freeze_d_model
        if self.freeze_d_model:
            self.transformer_rep_docs.requires_grad_(False)

    @abstractmethod
    def encode(self, kwargs, is_q) -> torch.Tensor:
        pass

    def _encode(self, tokens, is_q=False) -> torch.Tensor:
        transformer = self.transformer_rep_docs
        if is_q and self.transformer_rep_query is not None:
            transformer = self.transformer_rep_query
        return transformer(
            **tokens
        )  # It returns the CLS token [batch_size, hidden_size]

    def train(self, mode=True):
        if self.transformer_rep_docs is None:
            self.transformer_rep_docs.train(mode)
        else:
            self.transformer_rep_query.train(mode)
            mode_d = False if not mode else self.freeze_d_model
            self.transformer_rep_docs.train(mode_d)

    def forward(self, **kwargs):
        """
        forward takes as inputs 1 or 2 dict
        "d_kwards" => contains all inputs for document encoding
        "q_kwargs" => contains all inputs for query encoding
        "nb_negatives" => number of negatives for the document representation

        if nb_negatives is provided, the dim of d_kwargs change from [batch_size * 2, hidden_size] to [batch_size * nb_negatives, hidden_size]
        """
        out = {}
        do_d, do_q = "d_kwargs" in kwargs, "q_kwargs" in kwargs
        if do_d:
            d_rep = self.encode(kwargs["d_kwargs"], is_q=False)
            out.update({"d_rep": d_rep})
        if do_q:
            q_rep = self.encode(kwargs["q_kwargs"], is_q=True)
            out.update({"q_rep": q_rep})
        if do_d and do_q:
            bs = q_rep.shape[0]
            d_rep = d_rep.reshape(bs, kwargs["nb_negatives"], -1)
            q_rep = q_rep.unsqueeze(1)
            score = torch.sum(q_rep * d_rep, dim=-1)
            out.update({"score": score})

        return out


class Siamese(SiameseBase):
    """
    standard dense encoder class
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def encode(self, tokens, is_q):
        return self._encode(tokens=tokens, is_q=is_q)


"""
Specter2 has an adapter architecture, which can change the representation of documents and queries.
It use just one body model for both documents and queries so that it have to avoid to use Siamese class.
"""


class Specter2Base(torch.nn.Module, ABC):
    def __init__(self, model_name_or_dir: str, freeze_body_and_docs=True):
        super().__init__()
        self.model = AutoAdapterModel.from_pretrained(model_name_or_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_dir)
        self.model.load_adapter("allenai/specter2", source="hf", load_as="paper")
        self.model.load_adapter(
            "allenai/specter2_adhoc_query", source="hf", load_as="query"
        )

        self.model.set_active_adapters("paper")

        if freeze_body_and_docs:  # Freeze the body model
            for param in self.model.parameters():
                param.requires_grad = False

            for name, param in self.model.named_parameters():
                if "adapters.query" in name:
                    param.requires_grad = True

    @abstractmethod
    def encode(self, text, is_q) -> torch.Tensor:
        pass

    def _encode(self, text, is_q) -> torch.Tensor:
        adapter_type = "query" if is_q else "paper"
        self.model.set_active_adapters(adapter_type)
        tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        outputs = self.model(**tokens)
        return outputs.last_hidden_state[:, 0, :]  # CLS token representation


class Specter2Encoder(Specter2Base):
    def __init__(
        self,
        model_name_or_dir: str,
        freeze_body_and_docs=True,
    ):
        super().__init__(model_name_or_dir, freeze_body_and_docs)

    def encode(self, text, is_q):
        return self._encode(text=text, is_q=is_q)
