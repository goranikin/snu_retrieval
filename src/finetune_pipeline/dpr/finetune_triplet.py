import json
import random

from .model import Dpr
from .trainer_triplet import DprTrainer


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

    # 모델 및 트레이너 생성
    model = Dpr()
    trainer = DprTrainer(model)

    trainer.train(
        train_data=train_data,
        val_data=val_data,
        output_dir="./output",
        lr=5e-5,
        batch_size=4,
        epochs=10,
        margin=1.0,
        eval_steps=50,
        weight_decay=0.01,
        warmup_ratio=0.1,
    )
    print("Fine-tuning complete and model saved!")


if __name__ == "__main__":
    main()  # type: ignore
