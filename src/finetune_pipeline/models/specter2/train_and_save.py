import json
import random

import hydra
from model.model import Specter2
from model.trainer import Specter2Trainer
from omegaconf import DictConfig


@hydra.main(config_path="./conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    file_path = "../../data/datasets/final_generating_query_data.json"

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    merged_data_dict = {
        cfg.train.train_key: [],
        cfg.train.val_key: [],
        cfg.train.test_key: [],
    }

    for key, value in data_dict.items():
        for split in merged_data_dict.keys():
            if split in key:
                merged_data_dict[split].extend(value)

    print(f"# of train: {len(merged_data_dict[cfg.train.train_key])}")
    print(f"# of val: {len(merged_data_dict[cfg.train.val_key])}")
    print(f"# of test: {len(merged_data_dict[cfg.train.test_key])}")

    # 각 세트 섞기
    random.shuffle(merged_data_dict[cfg.train.train_key])
    random.shuffle(merged_data_dict[cfg.train.val_key])
    random.shuffle(merged_data_dict[cfg.train.test_key])

    # 모델 및 트레이너 생성
    model = Specter2()
    trainer = Specter2Trainer(model)

    # 학습 및 저장
    trainer.train(
        train_data=merged_data_dict[cfg.train.train_key],
        val_data=merged_data_dict[cfg.train.val_key],
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
