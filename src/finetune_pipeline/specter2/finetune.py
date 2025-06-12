import json
import random

# import hydra
# from omegaconf import
from .model import Specter2
from .trainer import Specter2Trainer


# @hydra.main(config_path="./conf", config_name="config", version_base=None)
def main():
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

    print("Specter2 생성 시작")
    model = Specter2()
    print("Specter2 생성 완료")
    print("Specter2Trainer 생성 시작")
    trainer = Specter2Trainer(model)
    print("Specter2Trainer 생성 완료")
    print("train 호출")
    trainer.train(
        train_data=train_data,
        val_data=val_data,
        output_dir="./output",
        lr=2e-5,
        batch_size=8,
        epochs=5,
        margin=1.0,
        eval_steps=50,
        weight_decay=0.01,
        warmup_ratio=0.1,
    )
    print("Fine-tuning complete and model saved!")


if __name__ == "__main__":
    main()  # type: ignore
