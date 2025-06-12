import torch
import hydra

from omegaconf import DictConfig

from splade.utils.utils import (
    set_seed,
)
from splade.models.models_utils import get_model
from transformers import AutoTokenizer

import random
import json
from splade.losses.regularization import RegWeightScheduler, SparsityRatio, FLOPS, L0
from splade.trainer import SpladeTrainer


def setup_regularizers(model, config):
    output_dim = model.output_dim if hasattr(model, "output_dim") else 30522

    regularizer = {
        "eval": {
            "L0": {"loss": L0()},
            "sparsity_ratio": {"loss": SparsityRatio(output_dim=output_dim)},
        },
        "train": {},
    }

    if config["use_regularization"]:
        reg_loss = FLOPS() if config["reg_type"] == "FLOPS" else L0()

        temp = {
            "loss": reg_loss,
            "targeted_rep": "rep",
            "lambdas": {},
        }

        temp["lambdas"]["lambda_q"] = RegWeightScheduler(
            lambda_=float(config["lambda_q"]),
            T=int(config["total_steps"] // 2),
        )

        temp["lambdas"]["lambda_d"] = RegWeightScheduler(
            lambda_=float(config["lambda_d"]),
            T=int(config["total_steps"] // 2),
        )

        regularizer["train"]["reg"] = temp

    return regularizer


@hydra.main(config_path="conf", config_name="splade_conf", version_base="1.2")
def main(config: DictConfig):
    # initialize
    set_seed(config.seed)

    # load dataset
    file_path = "./data/triplet_data.json"
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    random.shuffle(data)

    n_total = len(data)
    n_train = int(n_total * 0.8)
    train_data = data[:n_train]
    val_data = data[n_train:]

    print(f"# of train: {len(train_data)}")
    print(f"# of val: {len(val_data)}")

    tokenizer = AutoTokenizer.from_pretrained(config.model_type_or_dir)

    # load model
    init_dict = {
        "model_type_or_dir": config.model_type_or_dir,
        "model_type_or_dir_q": config.model_type_or_dir_q,
        "agg": "max",  # Aggregation method for SPLADE
        "fp16": config.fp16,  # Use the fp16 setting from config
    }

    print("model 호출")
    model = get_model(config, init_dict)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    reg_config = {
        "use_regularization": True,
        "reg_type": config.flops_loss,
        "lambda_q": config.lambda_q,
        "lambda_d": config.lambda_d,
        "total_steps": config.total_steps,
    }

    regularizer = setup_regularizers(model, reg_config)

    print("trainer 호출")
    trainer = SpladeTrainer(
        model=model,
        device=device,
        tokenizer=tokenizer,
        regularizer=regularizer,
    )

    print("train 호출")
    trainer.train(
        train_data=train_data,
        val_data=val_data,
        output_dir="./output",
        lr=config.lr,
        batch_size=config.train_batch_size,
        epochs=config.epochs,
        margin=config.margin,
        eval_steps=config.eval_steps,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
    )
    print("Fine-tuning complete and model saved!")


if __name__ == "__main__":
    main()  # type:ignore
