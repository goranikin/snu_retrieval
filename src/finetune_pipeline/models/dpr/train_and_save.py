import json
import random

# import hydra
# from omegaconf import DictConfig
from .model.model import Dpr
from .model.trainer import DprTrainer


# @hydra.main(config_path="./conf", config_name="config", version_base=None)
def main():
    file_path = "src/finetune_pipeline/data/triplet_data.json"

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
    # trainer.train(
    #     train_data=train_data,
    #     val_data=val_data,
    #     output_dir=cfg.train.output_dir,
    #     lr=cfg.train.lr,
    #     batch_size=cfg.train.batch_size,
    #     epochs=cfg.train.epochs,
    #     margin=cfg.train.margin,
    #     eval_steps=cfg.train.eval_steps,
    #     weight_decay=cfg.train.weight_decay,
    #     warmup_ratio=cfg.train.warmup_ratio,
    # )
    trainer.train(
        train_data=train_data,
        val_data=val_data,
        output_dir="/output",
        lr=2e-4,
        batch_size=256,
        epochs=5,
        margin=1.0,
        eval_steps=50,
        weight_decay=0.01,
        warmup_ratio=0.1,
    )
    print("Fine-tuning complete and model saved!")


if __name__ == "__main__":
    main()  # type: ignore
