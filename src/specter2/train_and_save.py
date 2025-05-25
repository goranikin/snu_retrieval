import glob
import json
import os
import random

import hydra
from omegaconf import DictConfig

from src.specter2.model import SPECTER2QueryAdapterFinetuner
from src.specter2.trainer import SPECTER2Trainer


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    datasets_dir = cfg.train.datasets_dir

    json_files = glob.glob(os.path.join(datasets_dir, "*.json"))

    data_dict = {}
    for file_path in json_files:
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            data_dict[file_name] = data

    for name, data in data_dict.items():
        print(
            f"{name}: {type(data)}, the number of samples: {len(data) if hasattr(data, '__len__') else 'N/A'}"
        )

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
    model = SPECTER2QueryAdapterFinetuner()
    trainer = SPECTER2Trainer(model, model.tokenizer, model.device)

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
