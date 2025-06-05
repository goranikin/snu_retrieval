import json
import random

import hydra
from model.model import Dpr
from model.trainer import DprTrainer
from omegaconf import DictConfig


@hydra.main(config_path="./conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    file_path = "../../data/datasets/finetuning_ㅋtriplet.json"

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    random.shuffle(data)

    n_total = len(data)
    n_train = int(n_total * 0.8)
    train_data = data[:n_train]
    val_data = data[n_train:]

    print(f"# of train: {len(train_data)}")
    print(f"# of val: {len(val_data)}")

    # 모델 및 트레이너 생성
    model = Dpr()
    trainer = DprTrainer(model)

    # 학습 및 저장
    trainer.train(
        train_data=train_data,
        val_data=val_data,
        output_dir=cfg.train.output_dir,
        lr=cfg.train.lr,
        batch_size=cfg.train.batch_size,
        epochs=cfg.train.epochs,
        margin=cfg.train.margin,
        eval_steps=cfg.train.eval_steps,
        weight_decay=cfg.train.weight_decay,
        warmup_ratio=cfg.train.warmup_ratio,
    )
    print("Fine-tuning complete and model saved!")


if __name__ == "__main__":
    main()  # type: ignore
